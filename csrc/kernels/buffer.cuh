#pragma once

#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep {

template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    int total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void* &gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void* &gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer() {
        return reinterpret_cast<dtype_t*>(ptr);
    }

    __device__ __forceinline__ dtype_t& operator[](int idx) {
        return buffer()[idx];
    }
};

template <typename dtype_t, int kNumRanks = 1>
struct AsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    int num_bytes;

public:
    int total_bytes;

    __device__ __forceinline__ AsymBuffer(void* &gbl_ptr, int num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        ptrs[0] = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ AsymBuffer(void** gbl_ptrs, int num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        for (int i = 0; i < kNumRanks; ++ i) {
            ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * sm_id + num_bytes * offset;
            gbl_ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    __device__ __forceinline__ void advance(int shift) {
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++ i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    __device__ __forceinline__ AsymBuffer advance_also(void* &gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    template<int kNumAlsoRanks>
    __device__ __forceinline__ AsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++ i)
            gbl_ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* buffer_by(int rank_idx, int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
/*

*/
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t* send_ptr;
    uint8_t* recv_ptr;
    int num_bytes;

public:
    int total_bytes;

    __device__ __forceinline__ SymBuffer(void* &gbl_ptr, int num_elems, int num_ranks,
                                         int sm_id = 0, int num_sms = 1) {
        num_bytes = num_elems * sizeof(dtype_t);

        int per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        send_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
        recv_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    } // 创建用于nvshmem 通讯的 对等buffer，前提是 rdma enable on all nodes

    __device__ __forceinline__ dtype_t* send_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* recv_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);
    }
    /*
        当 Decoupled=True，双向通讯中使用独立的 send/recv_buffer() 
        * send_buffer(idx), 返回第 idx-th rdma_rank 的 send_ptr 指针，由  num_bytes * idx 偏移得到 
        * recv_buffer(idx)，返回第 idx-th rdma_rank 的 recv_ptr 指针，由 num_bytes * idx 偏移得到
        send_ptr 与 recv_ptr memory layout: [SM0_send, SM1_send, ...,  SM0_recv, SM1_recv ... ]
        TODO: 对等buffer中，是谁发送/发往 idx-th ? 
    */

    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` is only available for decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }
    /*
        当 Decoupled=False，双向通讯中使用统一的buffer() api, 即做send()，又做 recv() 
    */
};

} // namespace deep_ep
