#include "configs.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++ i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__
int get_num_bytes_per_rdma_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4)));
}

__host__ __device__ __forceinline__
std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_rdma_ranks, int num_rdma_recv_buffer_tokens, int num_sms) {
    // Return `int32_t` offset and count to clean
    return {
        (get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_sms) / sizeof(int),
        (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_sms
    };
}

__host__ __device__ __forceinline__
std::pair<int, int> get_nvl_clean_meta(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_rdma_ranks, int num_nvl_ranks, int num_nvl_recv_buffer_tokens, int num_sms) {
    // Return `int32_t` offset and to clean
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");
    return {
        (num_nvl_recv_buffer_tokens * (hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float) + sizeof(SourceMeta)) * num_nvl_ranks * num_sms) / sizeof(int),
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_sms,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
    /*
        * 用于inter-node 的同步
        1. low-latency mode，use nvshmem_sync() to synchronize only within the current NVSHMEM team (subset of ranks)
        2. else, Uses `nvshmem_sync_all()` for global synchronization across all ranks
        * TODO: what's rdma_team here ? 
    */
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void
notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                const bool* is_token_in_rank, int num_tokens, int num_channels, int expert_alignment,
                const int rdma_clean_offset, const int rdma_num_int_clean,
                const int nvl_clean_offset, const int nvl_num_int_clean,
                int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                void* rdma_buffer_ptr,
                void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32; // 8

    /*
        * num_tokens_per_rank [64] # 在第一步构建数据时统计了要发到各rank的token数量
        * moe_recv_counter_mapped [1] # 该节点(该rdma_rank?)最终会收到的token数，用于分配 recv_x 显存
        * num_ranks(64)
        * num_tokens_per_rdma_rank[num_rdma_rank(8)] # 发送到rdma节点的token数
        * moe_recv_rdma_counter_mapped [1] # 从rdma收到的token数
        * num_tokens_per_expert [256] # 发送到各expert的token数
        * moe_recv_expert_counter_mapped # [1024] TODO: 只会用头部的4个，统计当前rank各expert收到的token数
        * is_token_in_rank, #[4096, 64]
        * num_channels, # 10
        * expert_alignment, #1
        * rdma_clean_offset,  # rdma_clean_meta.first, 83686400, 0.0836864e9 # TODO: 为什么 rdma_buffer 地址偏移从 这个数字开始 ?? 
        * rdma_num_int_clean, # rdma_clean_meta.second, 6400  # TODO: what's for ?
        * nvl_clean_offset,   # nvl_clean_meta.first, 85985280
        * nvl_num_int_clean,  # nvl_clean_meta.second,2880 
        * rdma_channel_prefix_matrix, #[num_rdma_ranks, num_channels], [8,10]
        * recv_rdma_rank_prefix_sum, #[num_rdma_ranks], [8]
        * gbl_channel_prefix_matrix, #[num_ranks, num_channels], [64, 10]
        * recv_gbl_rank_prefix_sum, #[num_ranks], [64]
        * rdma_buffer_ptr, #rdma cache, rdma_buffer_ptr, (byte)[1e9]
        * barrier_signal_ptrs[NUM_MAX_NVL_PEERS][NUM_MAX_NVL_PEERS] 用于 tblock 内对等通知彼此，自己ready
    */

    auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;
    /*
        * num_rdma_experts(256/8) ，per rdma_rank 上的 experts 数目
        * num_nvl_experts(32/8), per nvlink 上的 experts 数目 
    */

    if (sm_id == 0) {
    /*
        sm0 用于远端通讯
    */
        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        /*
            thread_id==32, 即 warp1 的 1st thread，用来做 nvshmem 全局同步
            1. 如果是 lowLatencyMode, 在 rdma_team 内的 gpu 之间同步
            2. highThroughtMode, 在所有 rdma_teams(即集群层面)同步
        */
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);
        /*
            * 用于 intra-node 内的GPU之间同步， barrier_signal_ptrs[8][8] 为`对等通信` a.k.a pair-wise communication，即 rank_i, rank_j 都告知对方 自己ready
            * 注意这里是 节点内 ipc 通讯
        */

        // Send numbers of tokens per rank/expert to RDMA ranks
        // 1. 创建对等buffer，all rdma_ranks 可见 
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);
        /*
            * 构建用于rdma通讯的对等buffer
            SymBuffer(void* &gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1)
            * num_elems=NUM_MAX_NVL_PEERS + num_rdma_experts + 1=8+32+1=41，即这个对等buffer里，存了41个整型指针，分别指向 每个 rdma_rank 上的 buffer 地址; 每个rdma_rank 上的 nexperts
            * kNumRDMARanks=8
            * 注意，这里使用 sm_id==0, num_sm==1 来创建 SymBuffer，对所有 rdma_rank 可见 
        */

        // Clean up for later data dispatch
        // 2. 初始化 rdma_buffer 
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
        /*
            * rdma_clean_offset，rdma_buffer 起始地址
            * rdma_num_int_clean(6400)，需要清零的buffer_size
            * 注意这个操作是per thread 执行的，故 step=num_threads
        */

        // Copy to send buffer
        /* 
            3. 填充 send_buffer，统计自己(rank)会往远端rdma_ranks 发送的tokens 统计信息，包括:
                1. 发送给远端 rdma_rank 对应 8个nvl_peer 的tokens 数, e.g.  num_tokens_per_rank[i]
                2. 发送给远端 rdma_rank 上32个experts 的tokens 数，e.g. num_tokens_per_expert[i]
                3. 发送给远端 rdma_rank 上 总共的tokens 数，e.g. num_tokens_per_rdma_rank[thread_id];
        */ 
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];
        /*
            rdma_rank_idx = i / NUM_MAX_NVL_PEERS
            local_rank_idx = i % NUM_MAX_NVL_PEERS，即当前rank/gpu，在节点内 local 索引
            实际效果，当前 rdma_rank 向 symBuffer 实例 rdma_recv_num_tokens_mixed 的 rdma_rank_idx 子缓存区域的 local_rank_idx 位置，写入 num_tokens_per_rank[i]； 
            写完成后，由于是 symBuffer实例，所有远端 rdma_rank 都能同步access该位置的vals
            
            注意，i+=num_threads, 即这个loop 都是以 per thread-block 为单位进行的，亦即物理上对应sm0
        */
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] = num_tokens_per_expert[i];
        /*
            * rdma_rank_idx =  i / num_rdma_experts，即当前 i-th expert 落在哪个 rdma_rank 上
            * local_expert_idx = i % num_rdma_experts，即当前 i-th expert 在当前 rdma_rank 上的局部索引
            实际效果：当前 rdma_rank 向 symBuffer 实例 rdma_recv_num_tokens_mixed 的 rdma_rank_idx 子缓冲区域中的 local_expert_idx 位置，写入 num_tokens_per_expert[i]；
            写完后，所有远端 rdma_rank 都能同步access 该位置的vales
        */
        if (thread_id < kNumRDMARanks)
            rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
        __syncthreads();
        /*
            实际效果：当前 rdma_rank 向 symBuffer 实例 rdma_recv_num_tokens_mixed[thread_idx] 子缓冲区域中的第 40(NUM_MAX_NVL_PEERS + num_rdma_experts)个位置，
            写入  num_tokens_per_rdma_rank[thread_id]; 
        */

        /*
            TODO: 不是 symBuffer 就是所有 rdma_rank 可见的嘛？为什么还需要再 send_buffer(i) 写 recv_buffer(rdma_rank) ??
        */

        // Issue send
        // TODO: more light fence or barrier or signaling
        // TODO: overlap EP barrier and NVL cleaning
        for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
            if (i != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                                reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                                (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),
                                                translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank), 0, lane_id, 0);
            } else { 
                UNROLLED_WARP_COPY(1, lane_id, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, 
                                    rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank), 
                                    rdma_recv_num_tokens_mixed.send_buffer(i), 
                                    ld_volatile_global, st_na_global);
            }
        }
        __syncthreads();

        // Wait previous operations to be finished。 有 i!=rdma_rank 对应，等待跨节点 同步完成
        if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);
        __syncthreads();

        // Barrier， 与 i==rdma_rank 对应，等待同一个rdma_team内同步完成
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        /*
            在rdma_team 中，每个rank只收到了同rdma_team的8个节点信息，其他56个rank的信息，则是通过ipc 从 同rdma_rank 上的 rank 上获取。
            这里所有通讯都是双向的，从而每个节点都可以获取全局发送给到自己的tokens相关信息
        */
        __syncthreads();

        // NVL buffers
        auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        /*
        这些变量定义了 comm_buffers 用于在节点内gpu 之间交换 tokens 统计信息
            * nvl_send_buffer , 只有 thread0-7 使用
            * nvl_recv_buffer ，每个nvl_rank 接收buffer 指针
            * nvl_reduced_num_tokens_per_expert ，per expert 上 aggregated tokens 数目
            * nvl_send_num_tokens_per_rank ，per rdma_rank 发送出去的 tokens 数目
            * nvl_send_num_tokens_per_expert , per expert 发送出去的 tokens 数目
            * nvl_recv_num_tokens_per_rank ， per rdma_rank 接收到的 tokens 数目
            * nvl_recv_num_tokens_per_expert ， per expert 接收到的 tokens 数目
        * 注意， AsymBuffer.buffer() 仅用于 single rdma_rank case 
        */

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes <= nvl_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
        if (thread_id < num_rdma_experts) { // num_rdma_experts(32)，为本地节点上experts 数目；前 32个 threads, per thread 为 一个 local expert 统计计数
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i)
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
            nvl_reduced_num_tokens_per_expert[thread_id] = sum; // 跟新 expert_i 上 reduce_num_tokens，其来自所有 rdma_rank 发给该expert 的tokens 总数
        }
        /*
            * per thread per expert counter
            * sm0 上 thread_i 去collect 从各个 rdma_rank 要发送给 expert_i 的tokens，并 reduce sum
            * TODO: 这里需要可视化理解下，前述 send_buffer(),  nvshmem_ibdga_put() 等操作与这里 recv_buffer() 的对应关系 
        */
        __syncthreads();

        // Reduce RDMA received tokens ，通过rdma 接收的tokens 统计
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
                recv_rdma_rank_prefix_sum[i] = sum; // 存的是per rdma_rank 上的tokens 总数
            }
            while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1);
            *moe_recv_rdma_counter_mapped = sum;
        }
        /*
            这里，只使用 thread0 去读取 rdma_recv_num_tokens_mixed symBuffer 中第 40(NUM_MAX_NVL_PEERS + num_rdma_experts)个位置上的 num_tokens_per_rdma_rank[thread_id]; 
            并写入 moe_recv_rdma_counter_mapped 指针，其为 mapped memory，即 cpu/gpu 都可以访问            
        */

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
        if (thread_id < NUM_MAX_NVL_PEERS) { // 使用前8条lanes 处理 xx 
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i)
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
            /*
            AsymBuffer.buffer(nvl_rank) ::  return (ptrs[0] + num_bytes * nvl_rank)
            nvl_send_num_tokens_per_rank.buffer() 是一个 8x8 指针，per nvl_rank dim，包含指向8个rdma_rank 上 tokens_统计的指针。如下：
            * rdma_rank0 上 nvl_rank0 
                * 其 bufffer[0] 写入来自 rdma_rank(0) 上 num_tokens_per_rdma_rank
                * 其 bufffer[1] 写入来自 rdma_rank(1) 上 num_tokens_per_rdma_rank
                * 。。
            * 注意，这里是节点内 ipc 通信，并不是当前 nvl_rank 走rdma从外部 rdma_rank 上取值，而是从与当前nvl_rank 同节点上 属于 其他 rdma_rank_id 的gpu 上获取这些tokens 统计值
            最终，nvl_send_num_tokens_per_rank 是以单个 rdma_rank 为视角，其上每个 nvl_rank 将向其他 rdma_rank 发送的tokens数目，组成的 8x8 指针
            */
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++ i)
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
            /*
                nvl_send_num_tokens_per_expert.buffer() 类似一个 8x4 指针，per nvl_rank dim, 包含指向4个 expert 上的tokens 统计的指针。如下:
                *  同理，ipc 通信，从 当前 nvl_rank 所对应 rdma_rank 发送到所有其他 rdma_rank 上相同 local gpu_idx 上的 4x experts, each expert 上的tokens 数目
            */
        }
        /*
            见 https://zhuanlan.zhihu.com/p/1890067712996270654
            * 在写 nvl_send_num_tokens_per_rank ， nvl_send_num_tokens_per_expert ， 对应的本节点的 nvl_recv_num_tokens_per_rank ， nvl_recv_num_tokens_per_expert 也被更新
        */
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
        /*
            用于 intra-node 内的GPU之间同步， barrier_signal_ptrs[8][8] 为`对等通信`，即 rank_i, rank_j 都告知对方 自己ready 
        */

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++ i) {
                int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
                recv_gbl_rank_prefix_sum[i] = sum;
            }
            while (ld_volatile_global(moe_recv_counter_mapped) != -1);
            *moe_recv_counter_mapped = sum;
        }
        /*
            * src_rdma_rank 当前 gpu_rank_idx 所在 rdma_rank 组 
            * src_nvl_rank 当前gpu_rank 在 当前 rdma_rank 组内的local idx 
            * nvl_recv_num_tokens_per_rank 类似一个 8x8 指针，表示一个 rdma_rank 组内 的 nvl_rank 分别从其他所有rdma_rank 获取的 tokens 统计
            * sum 是在所有gpu_ranks 上做的，即最终获取的是当前 rdma_rank 分组上的8个 nvl_ranks，将从其他所有 rdma_ranks(包含自身所在rdma_rank) 接收的tokens 总数
            * 注意，nvl_recv_num_tokens_per_rank 也是一个 symBuffer，即其地址和值，对所有gpu可见。
            * moe_recv_counter_mapped，其描述当前 rdma_rank 将接收到的tokens 总数，且其是 mapped_memory，即 host/device 都可见
        */
        if (thread_id < num_nvl_experts) { // 前4个threads 一对一到 local nvl_experts
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1);
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        /*
            nvl_recv_num_tokens_per_expert 类似一个8x4 指针，当前 nvl_rank 从其同一个rdma_rank组内其他 rdma_rank_idx 标记的 gpu_ranks，其每个上接收的 4x experts 分别的tokens 数目
            * sum 表示对于当前expert_id，其从所有同rdma_rank分组内，同gpu_id的8个跨节点gpu上接收到的tokens 总数
            * moe_recv_expert_counter_mapped 和 nvl_expert_id 一一对应。其表示当前 nvl_rank 上 per expert 将通过rdma 接收到的tokens 总数
        */
        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else { 
        /*
            * 见 https://zhuanlan.zhihu.com/p/1890067712996270654
            * 在 internode::notify_dispatch 中启用了9个 SM block计算, SM0 用 rdma通信和ipc通信来统计要接受和发送的信息。sm1~8 则统计channel（通道）粒度的内容
                * 使用8个sm block，每个sm block使用8个warp, 256个线程, 统计10个通道的发送信息，最终填充到rdma_channel_prefix_matrix， gbl_channel_prefix_matrix两个矩阵中。
            * 在 internode::dispatch 使用20个SM， 分成10个channel。 此时将待发送tokens顺序切分成10份，计算每份（通道）发送到远端rank, rdma_rank的token数量
        */
        // Calculate meta data
        /*
            num_channels 10
            num_warps 8
            warp_id = thread_id // 32 
            TODO: 这里 thread_id  vs lane_id ？ 
                * thread_id 是 tblock level 的线程 idx
                * lane_id 是 intra-warp 内的线程 idx ?    

            is_token_in_rank[ntokens, nranks]
        */
        int dst_rdma_rank = sm_id - 1;   // sm_id 1~8 ，每个sm 对应一个 rdma_rank 
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx); // 获取到当前warp需要处理token的[起始地址, 结束地址)
            /*
                * ntokens 维度上拆分成 10x channels，每个 channel 对应由一个 warp 负责发送/接收
                per warp 对应 per channel 上 tokens 处理。per warp 需要处理 1个或多个 channel
                per lane 对应 per token 处理
            */
            // Iterate over tokens
            int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                auto is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                /*
                    * is_token_in_rank + token_i * num_ranks  为当前token_i 在 is_token_in_rank 的首地址。
                    注意当前 token_i 理论上是映射到所有8个 rdma_rank 分组上。而每个 rdma_rank 分组由一个 sm 管理，
                    故 dst_rdma_rank * NVL_PEERS 由当前 dst_rdma_rank 负责cover 的 8个 位置(gpu_rank)，亦即在首地址基础上的偏移
                */
                #pragma unroll
                for (int j = 0; j < NUM_MAX_NVL_PEERS; ++ j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                /*
                    由当前 dst_rdma_rank 负责cover 的 8个GPU上，是否有当前 token_i；累计当前 rdma_rank上包含当前token_i 的gpu 个数
                */
                total_count += (is_token_in_rank_uint64 != 0);  
                /*
                    * per lane 处理的tokens，这些tokens 都会映射到 dst_rdma_rank 上的某些 gpu_rank， count 是具体哪些 gpu_ranks 上存在该 token
                    * total_count 是当前 lane 统计的 将发送到各个 rdma_rank 上tokens 总数 
                */
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count); // warp(channel)-level，亦即 当前 rdma_rank 将发送到各个 rdma_rank 上的 tokens 数统计，在per channel 维度上
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);
            /*
                当前 rdma_rank 其上各个 nvl_rank 将要发送出去的 tokens 统计
            */

            // Write into channel matrix
            if (lane_id == 0) { // 使用 local warp lane_id0 写，即每个 warp 上的 local lane0 执行
                #pragma unroll
                for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
            }
            /*
                gbl_channel_prefix_matrix[num_ranks, num_channels] # [64, 10]
                    * (dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels 当前 channel_id 的首地址
                * 基本就是 per sm_id 处理 per dst_rdma_rank，其对应 gbl_channel_prefix_matrix 中连续8行。每行又细分成 10x channels 
                rdma_channel_prefix_matrix[num_rdma_ranks, num_channels] #[8,10]
            */
        }

        // Calculate prefix sum
        __syncthreads();
        if (thread_id == 0) {  // global 0 号线程
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;   // rdma_channel_prefix_matrix 上的行首地址
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                prefix_row[i] += prefix_row[i - 1];  // 计算 prefix-sum
        }

        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                prefix_row[i] += prefix_row[i - 1]; // 计算 prefix-sum
        }
    }
}
/*  总结：
        notify_dispatch() 就是在登记各种token的发送/接受信息， 是一个比较轻量级的任务。
        之后， 在host cpu侧代码会基于 mapping内存 moe_recv_counter 得到节点接受的token数量，然后基于这个token数量分配recv_x的内存, 而后开始internode::dispatch工作。
*/

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
    auto notify_dispatch_func = low_latency_mode ? \
        notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
    LAUNCH_KERNEL(&cfg, notify_dispatch_func, \
                  num_tokens_per_rank, moe_recv_counter_mapped, num_ranks, \
                  num_tokens_per_rdma_rank, moe_recv_rdma_counter_mapped, \
                  num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, \
                  is_token_in_rank, num_tokens, num_channels, expert_alignment, \
                  rdma_clean_meta.first, rdma_clean_meta.second, \
                  nvl_clean_meta.first, nvl_clean_meta.second, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                  rdma_buffer_ptr, \
                  buffer_ptrs, barrier_signal_ptrs, rank, \
                  cpu_rdma_team); } break

    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode,
          int kNumDispatchRDMASenderWarps, int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)  
/*
    * kNumDispatchRDMASenderWarps=7
    * kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS = 16 ，即 dispatch() kernel 使用 16x warps launch
    * nblocks=1 ，即在一个物理sm留存一个软件block。 
    * 再加上 <<<20， >>>，即使用 20个block, 分配20个物理sm用于通信的目的。
*/
dispatch(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, SourceMeta* recv_src_meta,
         const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
         int* send_rdma_head, int* send_nvl_head,
         int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
         const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
         const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
         const bool* is_token_in_rank,
         int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
         int scale_token_stride, int scale_hidden_stride,
         void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
         void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
         int rank, int num_ranks) {
    /*
        * recv_x,  // [m, 7168]，用于接受发送到本节点的token, 给后续的expert层计算, m来自由 notify_dispatch moe_recv_counter
        * recv_x_scales,  // [m,56]
        * recv_topk_idx, // [m,8]
        * recv_src_meta, //[m, 8] （8byte, 2int, src_rdma_rank, is_token_in_nvl_rank_bits）
        * x, // [4096, 7168], 用于本节点发送出去的token
        * x_scales, //[4096,56]
        * topk_idx, //[4096, 8]
        * send_rdma_head, //[m, 8] // TODO ?
        * send_nvl_head,  //[m, 8] // TODO ? 
        * recv_rdma_channel_prefix_matrix, //[8, 10] 远端rdma_rank发送给自己的信息
            * 理解：分别从所有 8个 rdma_rank 接收的tokens 数分成 10x channels 从远端发送而来
        * recv_gbl_channel_prefix_matrix, //[64, 10], 远端rank各通道发送给自己的信息
            * 理解：分别从所有 64个 gpu_rank 接受的tokens 数分成 10x channels 发送而来
        * rdma_channel_prefix_matrix, //[8, 10] 通道发送给rdma的累加信息
            * TODO，跟 recv_rdma_channel_prefix_matrix 什么区别??
        * recv_rdma_rank_prefix_sum, //[8] 远端 rdma发送给自己的累加信息
            * 远端8个 rdma_rank，分别发送到当前 rdma_rank 上的tokens 总数

        * gbl_channel_prefix_matrix,  //[64, 10] 通道发送给rank的累加信息
        * recv_gbl_rank_prefix_sum, //[64], 远端rank发送给自己的累加信息
        * num_tokens, // 4096
        * hidden_int4, // 448
        * num_scales, // 56
        * num_topk, // 8
        * num_experts, // 256
        * is_token_in_rank, //[4096, 64]
        * rdma_buffer_ptr, //用于rdma通信的缓存 1e9 byte
        * num_max_rdma_chunked_send_tokens, //28
        * num_max_rdma_chunked_recv_tokens, //140
        * buffer_ptrs, // 用于ipc通信的缓存 1e9 byte
        * num_max_nvl_chunked_send_tokens, //20
        * num_max_nvl_chunked_recv_tokens, //288
        * num_ranks //64
    */
    enum class WarpRole {
        kRDMASender,
        kRDMASenderCoordinator,
        kRDMAAndNVLForwarder,
        kForwarderCoordinator,
        kNVLReceivers
    };
    /*  Warp Role 解释: 
            * kRDMASender，处理跨节点的rdma 数据发送
                * 管理 rdma buffer 元数据 (NUM_MAX_NVL_PEERS*2+2个int)
                * 将 token 分发信息写入 symBuffer 
                * 执行 rdma put 操作(nvshmem_ibgda_put_nbi_warp)
                * 处理 rdma channel 的 head/tail 指针同步 ??
            * kRDMASenderCoordinator, 协调多个 rdmaSender
                * 维护共享内存中的锁和窗口状态(rdma_send_channel_lock/tail/window)
                * 批处理rdma事务(每个事务最多32x tokens)
                * 处理跨channel的令牌发送协调
                * 执行最终的nvshmem同步
            * kRDMAAndNVLForwarder (rdma/nvl 转发器)，桥接 rdma 和 nvlink 通信
                * 从 rdma buffer 读取数据(nvl_channel_x/ nvl_channel_src_meta等)
                * 将 rdma 数据转化为 nvl 格式 ？？
            * kForwarderCoordinator 转发协调器
                * 维护 forward_channel_head 矩阵(num_max_nvl_peers * kNumRDMARanks)
                * 执行跨nvl_rank 的head 指针同步
                * 处理超时等
            * kNVLReceivers，处理nvlink 接收逻辑
                * 从 nvlink buffer 读取数据 (nvl_channel_x)
                * 执行数据聚合 combine_token()
                * 管理 combined_x, combined_topk_weights 输出buffer
        TODO 
    */

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = sm_id % 2 == 0;  // sm 分成 奇、偶 两组
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    /*
        num_sms 20 
        num_threads 512
        num_warps 16 // 注意，这里是 per sm 含 16 warps。
        num_channels 10 
    */

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

    // role_meta 是节点内的role meta
    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
            }
        /*
            * is_forwarder 即 偶数编号的 sms 上，其中 
                * warp0~7 为 kRDMAAndNVLForwarder，即负责节点内 rdma_rank 标记的 gpu 与其他gpu 之间的 nvl 转发，亦即 从rdma_recv_buffer 转写节点内 nvl_buffer 
                * warp8~15 为 kForwarderCoordinator, 向远端rdma_rank 确认接收
        */
        } 
        /*
            奇数编号的sms 上 :
                1. warp0~6 为 kRDMASender，从 x 写入 rdma_send_buffer 
                2. warp7 为 kRDMASenderCoordinator，将 rdma_send_buffer 写入 远端  rdma_rank 的 recv_buffer 中
                3. warp8~15 为 kNVLReceivers，从 nvl_buffer 写入 recv_x 
        */
        else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    /*
        * role_meta 返回 <warpRole，target_rank> 序列
        * TODO: 每个 warpRole 对应的 target_ranks 
    */
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second; // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto num_bytes_per_rdma_token = get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk, num_topk);
    auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    /*
        * num_bytes_per_rdma_token
            * token hidden_bytes 7168
            * scale 7168/128 * 4 =224
            * topk_idx: 8 * 4 = 32
            * topk_weight: 8 * 4 = 32
            * sourceMeta 8
            * 16byte对齐后为 7472 bytes
        
        * num_max_rdma_chunked_recv_tokens, //140
        * SymBuffer(gbl_ptr, num_elems, num_ranks, sm_id=0, nums_sm=1)
        * rdma_channel_data[channel_i].send_ptr 首地址:  gbl_ptr + num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token * kNumRDMARanks * channel_i 
        * rdma_channel_data[channel_i].recv_ptr 首地址:  gbl_ptr + num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token * kNumRDMARanks * ( channel_i + num_channels)
        * rdma_channel_meta[channel_i].send_ptr 首地址: gbl_ptr + (NUM_MAX_NVL_PEERS * 2 + 2) * kNumRDMARanks * channel_id 
        * rdma_channel_meta[channel_i].recv_ptr 首地址: gbl_ptr + (NUM_MAX_NVL_PEERS * 2 + 2) * kNumRDMARanks * ( channel_id + num_channels)
        * rdma_channel_head[channel_i].send_ptr 首地址： gbl_ptr + 1*kNumRDMARanks*channel_id 
        * rdma_channel_head[channel_i].recv_ptr 首地址： gbl_ptr + 1*kNumRDMARanks* ( channel_id + num_channels)
        * rdma_channel_tail[channel_i].send_ptr 首地址： gbl_ptr + 1*kNumRDMARanks*channel_id 
        * rdma_channel_tail[channel_i].recv_ptr 首地址： gbl_ptr + 1*kNumRDMARanks* ( channel_id + num_channels)        

        TODO: head/tail 指针怎么用的？？ 
    */

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for Receivers"
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    /*
        * 同一个指针，可能对 Forwarder 是“读”，但对 Receiver 是“写”。
        * buffer_ptrs, 是节点内8个gpu的共享内存地址，由 ipc handle 映射的一块buffer: 1e9 byte，节点内8个gpu 均可见。
    */
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
    /*
        * 对于 kRDMAAndNVLForwarder 角色的 warps(warp0~7 on each even sm)， 其负责节点内从 local rdma buffer 读取信息填入到 remote nvl_buffer
        * 当前 nvl_rank 从 buffer_pts[nvl_rank] 读取，通过nvl 通信，写入peers remote GPU 的 nvl_buffer，即 buffer_ptrs[target_rank]
        * rs_wr_buffer_ptr 是共享buffer 上nvl_rank 读取地址 , [source] local rdma send_buffer
        * ws_rr_buffer_ptr 是共享buffer上 target_rank 写出地址，[destination] remote nvl buffer 
    */
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;
    /*
        * 对于 kNVLReceivers 角色的warps(warp8~15 on each odd sm)，其负责从 nvl_buffer 读取信息填入 自己(nvl_rank) 的 recv_x 
        * rs_wr_buffer_ptr 是共享buffer 上 target_rank 读取地址，(其上数据即forwarder 阶段 target_rank 写出的数据)， [source] remote nvl_buffer
        * ws_rr_buffer_ptr 是共享buffer 上 写入 nvl_rank(当前rank)的 recv_buffer 地址，[destination] local rdma recv_buffer 
    */

    /*
        总结目前理解，
            1) 当warpRole 是 rdma&nvlForwarder，则从当前 nvl_rank 对应的 rdma_buffer.send_buffer 中取值，填入 remote target_ranks(即其余peer gpu)的 nvl_buffer
            2）当warpRole 是 nvlReceivers，则从remote target_ranks 的nvl_buffer 中取值， 写入当前 nvl_rank 对应的 rdma_buffer.recv_buffer中 
    */

    // Allocate buffers
    auto nvl_channel_x = AsymBuffer<int4>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * hidden_int4, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_x_scales = AsymBuffer<float>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_scales, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_topk_idx = AsymBuffer<int>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_topk_weights = AsymBuffer<float>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head = AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail = AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    /*
        * ws_rr_buffer_ptr 指向 target_rank ipc， 即 ipc ranks  
        * rs_wr_buffer_ptr 指向 src_rank buffer ，即 current nvl_rank 读取(read)自己的 send_buffer 以发送(send)到远端 target_ranks 的 recv_buffer 

        * AsymBuffer(gbl_ptr, num_elems, num_ranks, sm_id = 0, num_sms = 1, offset = 0)

        * nvl_channel_x[channel_id][rs_wr_rank].ptrs[0] 首地址， ws_rr_buffer_ptr[0] + num_bytes * NUM_MAX_NVL_PEERS * channel_id + num_bytes * rs_wr_rank 
        * nvl_channel_x[channel_id][rs_wr_rank].ptrs[1] 首地址， ws_rr_buffer_ptr[1] + num_bytes * NUM_MAX_NVL_PEERS * channel_id + num_bytes * rs_wr_rank 
        * ..
        * nvl_channel_x[channel_id][rs_wr_rank].ptrs[7] 首地址， ws_rr_buffer_ptr[7] + num_bytes * NUM_MAX_NVL_PEERS * channel_id + num_bytes * rs_wr_rank 
        * 同理 for  nvl_channel_src_meta, x_scales, topk_idx, topk_weights
        
        * TODO: prefix_start, prefix_end, head, tail 怎么用??
    */

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("bar.sync 0, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("bar.sync 1, %0;" :: "r"((NUM_MAX_NVL_PEERS + 1) * 32)); };
    /*
        TODO: 
    */

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
        /*
            * ntokens 维度上拆分成 10x channels，每个 channel 对应由一个 warp 负责发送/接收
            per warp 对应 per channel 上 tokens 处理。per warp 需要处理 1个或多个 channel
            per lane 对应 per token 处理
        */

        // Send number of tokens in this channel by `-value - 1`
        /*
            * 由8个warp，每个warp处理一个远端 rdma_rank。

            1.  rdma_channel_meta 的 buffer 选择
                * 当目标rank 就是本rank，则使用 recv_buffer(避免自发送)
                * 当目标rank 是其他Rank，则使用 send_buffer 

            * gbl_channel_prefix_matrix[64, 10] # nranks * nchannels 
            * rdma_channel_prefix_matrix[8, 10] # nrdma_ranks * nchannels 

            2. 每个warp 处理一个 rdma_rank，并按 lane_id 划分工作
                * lane0~7，编码nvl peers 的前缀和, gbl_channel_prefix_matrix。本channel 在远端8个 rdma_rank  上的起始 index 取负数
                * lane8~15，编码当前channel 的 nvl peers 结束位置。本channel 在远端8个 rdma_rank 上的结束 index 取负数
                * lane16，编码rdma 通道的起始位置，rdma_channel_prefix_matrix。本channel 在远端 rdma_rank 上起始index 取负 
                * lane17，编码rdma 通道的结束位置
        */
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // Issue RDMA for non-local ranks, 通过 nvshmem 发送 send_buffer 到远端 rdma_rank 节点
            /*
                nvshmemx_int_put_nbi_warp是一个warp级别函数，需要当前warp全部线程参与
            */
            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem(); // 同步 kRDMASender 和 kRDMASenderCoordinator ，所有8个warp 完成后，再进入下一步

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        /*
            * 填充 send_buffer：
                1. 如果是 rdma_rank 本身，直接将 rdma_channel_data.recv_buffer() 设置为 send_buffer
                2. 否则，就是当前channel 的 send_buffer + lane_id 偏移
        */
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++ token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 = __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp(); // 确保8条lane 都完成读取，在后续指向
            /*
                * 读取跨节点分发信息，关键优化：
                    1. 使用 __ldg() 指令 进行常量缓存读取，减少dram访问延迟
                    2. 将 8x bool package 成 uint64 读取。 per bool 偏移计算: token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS
                * uint64 表示当前token 在 target rdma_rank 上8个gpu 上的分布情况 
                * global_rdma_tail_idx 统计该token需要发送给的RDMA_ranks 的数量
                * 这里 per lane 处理 (per token 是否要发送到) 某个 rdma_rank
            */

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            /*
                每个 warp 处理 的token list:   warp_id, warp_id+7, warp_id+14, ...
                这里 bypass 不属于当前warp 要处理的 tokens
            */
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;
            /*
                 * 若当前token 不需要发送给 当前 rdma_rank， 则 tail_idx = -1 (无效索引)
                 * 若当前 token 需要发送给 当前 rdma_rank，则 tail_idx 为实际尾指针(glb_rdma_tail_idx-1)
            */
            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));

                // Timeout check
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            /*
                1. 通过使用 RingBuffer，发送端只维护 tail_ptr, 而接收端只维护 head_ptr，即可实现无锁设计
                2. 通过在发送端维护 head_ptr 的副本 cached_rdma_channel_head，减少频繁 head_ptr 读取。
                只有当`tail_ptr - cached_head_ptr >= 缓冲区大小`时，才更新该 cached_rdma_channel_head
            */
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;
            /*
                * rdma_tail_idx，当前token 在 target rdma_rank rdma_buffer 中的写入位置
                * send_rdma_head [ntokens, num_rdma_ranks] # 这里 lane_id 与 n_rdma_rank_idx 等价
                所以，这里实际上是 token_idx 在 lane_id 个 rdma_rank 上的 rdma_head 指针更新?

            */

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++ i) if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                topk_ranks[num_topk_ranks] = i;
                auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                if (lane_id == num_topk_ranks)
                    src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                dst_send_buffers[num_topk_ranks ++] = reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_rdma_token;
            }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);
            /*
                
            */

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {
                // Acquire lock first
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        // Synchronize shared memory
        sync_rdma_sender_smem();

        // Get number of tokens to send for each RDMA rank
        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        // Iterate all RDMA ranks
        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                trap();
            }

            // TODO: try thread-level `put_nbi`?
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++ i) {
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                auto processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                // Issue RDMA send
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_rdma_token * num_tokens_to_issue;
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_rdma_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_rdma_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
                } else {
                    // Lighter fence for local RDMA rank
                    memory_fence();
                }
                __syncwarp();

                // Update tails
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;
        const auto dst_rank = rdma_rank * NUM_MAX_NVL_PEERS + dst_nvl_rank;
        const auto dst_rank_expert_begin = dst_rank * (num_experts / num_ranks);
        const auto dst_rank_expert_end = dst_rank_expert_begin + (num_experts / num_ranks);

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, meta: %d, %d, %d, %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank, meta_0, meta_1, meta_2, meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();
            while (lane_id == 0) {
                int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = ld_volatile_global(nvl_channel_head.buffer());

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, ld_volatile_global(nvl_channel_head.buffer()), cached_nvl_channel_tail);
                    trap();
                }
            }
            __syncwarp();

            // Find next source RDMA rank (round-robin)
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, head: %d, tail: %d, expected: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_rdma_channel_head, cached_rdma_channel_tail, num_tokens_to_recv_from_rdma);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++ i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                void* shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(static_cast<int8_t*>(shifted) + hidden_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    if (not kCachedMode)
                        send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
                }
                if (not is_in_dst_nvl_rank)
                    continue;

                // Get an empty slot
                int dst_slot_idx = (cached_nvl_channel_tail ++) % num_max_nvl_chunked_recv_tokens;

                // Copy data
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                                   nvl_channel_x.buffer() + dst_slot_idx * hidden_int4,
                                   reinterpret_cast<int4*>(shifted),
                                   ld_nc_global, st_na_global);
                shifted = static_cast<int4*>(shifted) + hidden_int4;

                // Copy source meta
                if (lane_id == 0)
                    st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx, src_meta);
                shifted = static_cast<SourceMeta*>(shifted) + 1;

                // Copy `x_scales`
                UNROLLED_WARP_COPY(1, lane_id, num_scales,
                                   nvl_channel_x_scales.buffer() + dst_slot_idx * num_scales,
                                   reinterpret_cast<float*>(shifted),
                                   ld_nc_global, st_na_global);
                shifted = static_cast<float*>(shifted) + num_scales;

                // Copy `topk_idx` and `topk_weights`
                // NOTES: do not use `shifted` after this `if`, because only several lanes are shifted
                if (lane_id < num_topk) {
                    // Read
                    auto idx_value = ld_nc_global(static_cast<int*>(shifted) + lane_id);
                    shifted = static_cast<int*>(shifted) + num_topk;
                    auto weight_value = ld_nc_global(static_cast<float*>(shifted) + lane_id);

                    // Transform and write
                    idx_value = (idx_value >= dst_rank_expert_begin and idx_value < dst_rank_expert_end) ? idx_value - dst_rank_expert_begin : -1;
                    st_na_global(nvl_channel_topk_idx.buffer() + dst_slot_idx * num_topk + lane_id, idx_value);
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    st_na_global(nvl_channel_topk_weights.buffer() + dst_slot_idx * num_topk + lane_id, weight_value);
                }

                // In case of insufficient NVL buffers, early stopping
                if ((++ num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (lane_id == 0)
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (lane_id == 0)
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        // Extra warps for forwarder coordinator should exit directly
        if (target_rank > 0)
            return;

        // Forward warp coordinator
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Clean shared memory
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i) if (not forward_channel_retired[i])
                min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            // Update remote head
            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank), channel_id + num_channels, lane_id == rdma_rank);
                last_head = min_head;
            }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        int src_nvl_rank = target_rank, total_offset = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks and not kCachedMode)
            recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Check channel status by lane 0
            start_time = clock64();
            while (lane_id == 0) {
                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer());

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
                    trap();
                }
            }

            // Sync queue tail
            cached_channel_tail_idx = __shfl_sync(0xffffffff, cached_channel_tail_idx, 0);

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++ chunk_idx, -- num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx ++) % num_max_nvl_chunked_recv_tokens;
                auto meta = ld_nc_global(nvl_channel_src_meta.buffer() + token_idx_in_buffer);
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

                // Copy data
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                                   recv_x + recv_token_idx * hidden_int4,
                                   nvl_channel_x.buffer() + token_idx_in_buffer * hidden_int4,
                                   ld_nc_global, st_na_global);

                // Copy source meta
                if (lane_id == 0 and not kCachedMode)
                    st_na_global(recv_src_meta + recv_token_idx, meta);

                // Copy scales
                UNROLLED_WARP_COPY(1, lane_id, num_scales,
                                   recv_x_scales + recv_token_idx * num_scales,
                                   nvl_channel_x_scales.buffer() + token_idx_in_buffer * num_scales,
                                   ld_nc_global, st_na_global);

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    auto recv_idx = recv_token_idx * num_topk + lane_id;
                    auto buffer_idx = token_idx_in_buffer * num_topk + lane_id;
                    st_na_global(recv_topk_idx + recv_idx, static_cast<int64_t>(ld_nc_global(nvl_channel_topk_idx.buffer() + buffer_idx)));
                    st_na_global(recv_topk_weights + recv_idx, ld_nc_global(nvl_channel_topk_weights.buffer() + buffer_idx));
                }
            }

            // Move queue
            __syncwarp();
            if (lane_id == 0)
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }
    }
}

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              int scale_token_stride, int scale_hidden_stride,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              cudaStream_t stream, int num_channels, bool low_latency_mode) {
    constexpr int kNumDispatchRDMASenderWarps = 7;

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
    auto dispatch_func = low_latency_mode ? \
        (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumDispatchRDMASenderWarps> : dispatch<true, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>) : \
        (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumDispatchRDMASenderWarps> : dispatch<false, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>); \
    LAUNCH_KERNEL(&cfg, dispatch_func, \
                  reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx, recv_topk_weights, reinterpret_cast<SourceMeta*>(recv_src_meta), \
                  reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights, \
                  send_rdma_head, send_nvl_head, \
                  recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                  is_token_in_rank, \
                  num_tokens, hidden_int4, num_scales, num_topk, num_experts, \
                  scale_token_stride, scale_hidden_stride, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_ptrs, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    EP_HOST_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    /*
        分配20个 block做 dispatch() kernel 计算，该kernel 固定 per sm per block，故这里 20个 tblock, 对应到 20个 sm 上。用于通信
    */
    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode>
__global__ void cached_notify(const int rdma_clean_offset, const int rdma_num_int_clean,
                              const int nvl_clean_offset, const int nvl_num_int_clean,
                              int* combined_rdma_head, int num_combined_tokens, int num_channels,
                              const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs, int** barrier_signal_ptrs, int rank, int num_ranks,
                              bool is_cached_dispatch, const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();

    auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Using two SMs, which clean the RDMA/NVL buffer respectively
    if (sm_id == 0) {
        // Barrier for RDMA
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        __syncthreads();

        // Clean
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
        __syncthreads();

        // Barrier again
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    } else if (sm_id == 1) {
        // Barrier for NVL
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Clean
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

        // Barrier again
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else if (sm_id == 2) {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        // Iterate in reverse order
        if (lane_id < num_rdma_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

            // NOTES: `1 << 25` is a heuristic large number
            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; -- token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                } else {
                    last_head = current_head;
                }
            }
        }
    } else {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        if (lane_id < NUM_MAX_NVL_PEERS and warp_id < num_channels) {
            for (int dst_rdma_rank = sm_id - 3; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 3) {
                // Iterate in reverse order
                int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
                int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
                int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                token_start_idx += shift, token_end_idx += shift;

                // NOTES: `1 << 25` is a heuristic large number
                int last_head = 1 << 25;
                #pragma unroll
                for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; -- token_idx)  {
                    auto current_head = __ldg(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                    if (current_head < 0) {
                        combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
                    } else {
                        last_head = current_head;
                    }
                }
            }
        }
    }
}

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode) {
    const int num_threads = std::max(128, 32 * num_channels);
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_channels * 2 > 3);

    // Launch kernel
    auto cached_notify_func = low_latency_mode ? cached_notify<true> : cached_notify<false>;
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
    LAUNCH_KERNEL(&cfg, cached_notify_func,
                  rdma_clean_meta.first, rdma_clean_meta.second,
                  nvl_clean_meta.first, nvl_clean_meta.second,
                  combined_rdma_head, num_combined_tokens, num_channels,
                  rdma_channel_prefix_matrix, rdma_rank_prefix_sum, combined_nvl_head,
                  rdma_buffer_ptr,
                  buffer_ptrs, barrier_signal_ptrs, rank, num_ranks,
                  is_cached_dispatch, cpu_rdma_team);
}

template <int kNumRanks, bool kMaybeWithBias, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank, int head_idx,
                             int lane_id, int hidden_int4, int num_topk,
                             int4* combined_row, float* combined_topk_weights,
                             const int4* bias_0_int4, const int4* bias_1_int4,
                             int num_max_recv_tokens, const ReceiveFn& recv_fn, const ReceiveTWFn& recv_tw_fn) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++ i) if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
        slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
        topk_ranks[num_topk_ranks ++] = i;
    }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

    // Reduce data
    #pragma unroll
    for (int i = lane_id; i < hidden_int4; i += 32) {
        // Read bias
        // TODO: make it as a finer-grained template
        int4 bias_0_value_int4, bias_1_value_int4;
        if (kMaybeWithBias) {
            bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
            bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
        }

        // Read buffers
        // TODO: maybe too many registers here
        int4 recv_value_int4[kMaxNumRanks];
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j)
            recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);
        
        // Clean
        // Reduce bias
        float values[kDtypePerInt4] = {0};
        if (kMaybeWithBias) {
            auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
            auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++ j)
                values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
        }

        // Reduce all-to-all results
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j) {
            auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
            #pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++ k)
                values[k] += static_cast<float>(recv_value_dtypes[k]);
        }

        // Cast back to `dtype_t` and write
        int4 out_int4;
        auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
        #pragma unroll
        for (int j = 0; j < kDtypePerInt4; ++ j)
            out_dtypes[j] = static_cast<dtype_t>(values[j]);
        st_na_global(combined_row + i, out_int4);
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++ i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        st_na_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

template<bool kLowLatencyMode,
         int kNumRDMARanks, typename dtype_t,
         int kNumCombineForwarderWarps,
         int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
         int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
         int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
         int kNumRDMAReceivers = kNumForwarders + NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((NUM_MAX_NVL_PEERS + 1 + kNumForwarders) * 32, 1)
combine(int4* combined_x, float* combined_topk_weights,
        const bool* is_combined_token_in_rank,
        const int4* x, const float* topk_weights,
        const int4* bias_0, const int4* bias_1,
        const int* combined_rdma_head, const int* combined_nvl_head,
        const SourceMeta* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
        int num_tokens, int num_combined_tokens, int hidden, int num_topk,
        void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
        void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
        int rank, int num_ranks) {
    enum class WarpRole {
        kNVLSender,
        kNVLAndRDMAForwarder,
        kRDMAReceiver,
        kCoordinator
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_rdma_receiver_sm = sm_id % 2 == 1;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_rdma_receiver_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
                auto shuffled_warp_id = warp_id - NUM_MAX_NVL_PEERS;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
                return {WarpRole::kRDMAReceiver, warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == NUM_MAX_NVL_PEERS + kNumForwarders + 1);
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    if (warp_role == WarpRole::kNVLSender) {
        // NVL producers
        const auto dst_nvl_rank = warp_id;

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<int4>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * hidden_int4, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);
        auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);
        auto nvl_channel_topk_weights = AsymBuffer<float>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank).advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
        }
        __syncwarp();

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Iterate over all tokens and send by chunks
        while (true) {
            // Exit if possible
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: %d, tail: %d, start: %d, end: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, ld_volatile_global(nvl_channel_head.buffer() + lane_id), cached_channel_tail_idx,
                           token_start_idx, token_end_idx);
                    trap();
                }
            }

            // Sync token start index and count
            for (int current_rdma_idx = 0; current_rdma_idx < kNumRDMARanks; ++ current_rdma_idx) {
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                // Sync token start index
                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int num_tokens_in_chunk = __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++ chunk_idx, ++ token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx ++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    // Copy data
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                    // Copy source meta
                    if (lane_id == 0)
                        st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx, ld_nc_global(src_meta + token_idx));

                    // Copy `topk_weights`
                    if (lane_id < num_topk)
                        st_na_global(nvl_channel_topk_weights.buffer() + dst_slot_idx * num_topk + lane_id, ld_nc_global(topk_weights + token_idx * num_topk + lane_id));
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            // Move queue tail
            __syncwarp();
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto hidden_bytes = hidden_int4 * sizeof(int4);
        auto num_bytes_per_rdma_token = get_num_bytes_per_rdma_token(hidden_int4, 0, 0, num_topk);
        auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x = AsymBuffer<int4>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * hidden_int4, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_topk_weights = AsymBuffer<float>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_topk, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head = AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Combiner warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        auto sync_forwarder_smem = [=]() { asm volatile("bar.sync 0, %0;" :: "r"((kNumForwarders + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("bar.sync 1, %0;" :: "r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" :: "r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // Advance to the corresponding NVL buffer
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * hidden_int4);
            nvl_channel_src_meta.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma);
            nvl_channel_topk_weights.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_topk);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            // Clean shared memory and sync
            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

            // Iterate over all tokens and combine by chunks
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: %d, chunked: %d\n",
                               channel_id, rdma_rank, nvl_rank, dst_rdma_rank, ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)), token_start_idx, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                // Combine and write to the RDMA buffer
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);

                    // Wait lanes to be ready
                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        // Timeout check
                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            printf("DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                   channel_id, rdma_rank, nvl_rank, lane_id, dst_rdma_rank, cached_nvl_channel_tail_idx, token_idx, num_tokens_to_combine, sub_warp_id, kNumWarpsPerForwarder, expected_head);
                            trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_rdma_token;
                    auto recv_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * hidden_int4 + hidden_int4_idx); };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(nvl_channel_topk_weights.buffer(src_nvl_rank) + slot_idx * num_topk + topk_idx); };
                    combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS>(expected_head >= 0,
                                                                                 expected_head, lane_id,
                                                                                 hidden_int4, num_topk,
                                                                                 static_cast<int4*>(shifted),
                                                                                 reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                                                                                 nullptr, nullptr, num_max_nvl_chunked_recv_tokens_per_rdma, recv_fn, recv_tw_fn);

                    // Update head
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1) : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_rdma_token;
                        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token);
                        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
                    } else {
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    if (lane_id == 0) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                    }
                }
            }

            // Retired
            __syncwarp();
            if (lane_id == 0)
                forwarder_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1) : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, expect: %d\n",
                               channel_id, rdma_rank, nvl_rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Combine current token
                auto recv_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(reinterpret_cast<const int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_rdma_token) + hidden_int4_idx);};
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_rdma_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx);};
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks>(expected_head >= 0,
                                                                         expected_head, lane_id,
                                                                         hidden_int4, num_topk,
                                                                         combined_x + token_idx * hidden_int4,
                                                                         combined_topk_weights + token_idx * num_topk,
                                                                         bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
                                                                         bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
                                                                         num_max_rdma_chunked_recv_tokens, recv_fn, recv_tw_fn);
            }

            // Retired
            __syncwarp();
            if (lane_id == 0)
                rdma_receiver_retired[warp_id] = true;
        } else {
            // Coordinator
            // Sync shared memory status
            is_rdma_receiver_sm ? sync_rdma_receiver_smem() : sync_forwarder_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                if (is_rdma_receiver_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (not is_rdma_receiver_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                if (is_rdma_receiver_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++ i) if (not rdma_receiver_retired[i])
                        min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id + num_channels, dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    // Find minimum head for NVL ranks
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks; ++ i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++ j) if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                            min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                    }
                }

                // Nanosleep and let other warps work
                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }
}

void combine(cudaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int rank, int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode) {
    constexpr int kNumCombineForwarderWarps = 16;

#define COMBINE_LAUNCH_CASE(num_rdma_ranks) { \
    auto combine_func = low_latency_mode ? \
        combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps> : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps>; \
    LAUNCH_KERNEL(&cfg, combine_func, \
                  reinterpret_cast<int4*>(combined_x), combined_topk_weights, is_combined_token_in_rank, \
                  reinterpret_cast<const int4*>(x), topk_weights, \
                  reinterpret_cast<const int4*>(bias_0), reinterpret_cast<const int4*>(bias_1), \
                  combined_rdma_head, combined_nvl_head, \
                  reinterpret_cast<const SourceMeta*>(src_meta), rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, \
                  num_tokens, num_combined_tokens, hidden, num_topk, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_ptrs, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    EP_HOST_ASSERT(num_forwarder_warps > 0 and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks > std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(type == CUDA_R_16BF);

    SETUP_LAUNCH_CONFIG(num_channels * 2, (NUM_MAX_NVL_PEERS + num_forwarder_warps + 1) * 32, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode

} // namespace deep_ep
