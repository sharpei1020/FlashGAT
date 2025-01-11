#include <cuda_runtime.h>
#include <cstdint>
#include "agnn.cuh"

#define FULL_MASK 0xffffffff

__global__ void agnn_kernel_16x8_32(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    __shared__ float dense_X[2][8][32];
    __shared__ float softmax[2][16][2];
    __shared__ float D[16][32];

    int cur_addr = __cvta_generic_to_shared(&D[threadIdx.x>>3][(lane_id&7)*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[(bid * 16 + (threadIdx.x>>3)) * 32 + (lane_id&7) * 4]));
    cur_addr = __cvta_generic_to_shared(&dense_X[0][(threadIdx.x>>4)][(lane_id&3)*2]);
}