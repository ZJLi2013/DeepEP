#include <vector>
#include <cstring>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#endif

namespace deep_ep {

namespace intranode {

template<int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks) \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {

#ifndef DISABLE_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    /*
        通过 root_unique_id 来构建nvshmem的通信组
        root_unique_id 是 rdma_rank=0 的 8张显卡会各自产生 unique_id, 
        然后每台机器上的显卡使用 rdma_group = rank % 8 来获取 rdma_rank=0的8个显卡对应的第rdma_group显卡的unique id
        也就是说，rdma_rank 分组不是同一个node中的8张gpu，而是 所有 nodes上，相同 local_gpu_id 的分组
        然后通过nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr)完成网络组构建
    */

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS, NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS, &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    return nvshmem_align(alignment, size);
}
/*
    aligned allocations for inter-node communication buffers, used for : 
        * RDMA operations in the `internode` namepsace
        * buffer for gpu-to-gpu communication across nodes
        * preparing memory used fornvshmem Put/Get operations and atomics
    * must be freed with `nvshmem_free()`
*/

void free(void* ptr) {
    nvshmem_free(ptr);
}

void barrier() {
    nvshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}
#endif

} // namespace internode

} // namespace deep_ep
