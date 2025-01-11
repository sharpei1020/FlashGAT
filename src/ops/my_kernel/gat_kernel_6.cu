#include <cuda_runtime.h>
#include <cstdint>
#include "gat.cuh"

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float leaky_relu(float x) {
    return x - 0.99f * min(0.f, x);
}

__global__ void __launch_bounds__(128, 12) gat_kernel_16x8(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    if(block_end == block_start) return;

    int lane_0_4 = lane_id&15;
    int thread_4_7 = threadIdx.x>>4;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float dense[2][8][64];
    __shared__ float softmax[2][2][16];
    
    int i = block_start;
    SparseAToX_idx[i&1][thread_4_7] = SparseAToX[i * 8 + thread_4_7];
    if (warp_id == 0) {
        softmax[thread_4_7][0][lane_0_4] = std::numeric_limits<float>::lowest();
        softmax[thread_4_7][1][lane_0_4] = 0.f;
    }
    int cur_addr = __cvta_generic_to_shared(&dense[i&1][thread_4_7][lane_0_4*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[SparseAToX_idx[i&1][thread_4_7]*64+4*lane_0_4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};

    for (i = block_start; i < (block_end - 1); i++) {
        float B[4];
        // load and init
        SparseAToX_idx[(i+1)&1][thread_4_7] = SparseAToX[(i+1)*8 + thread_4_7];
        int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][thread_4_7][lane_0_4*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[SparseAToX_idx[(i+1)&1][thread_4_7]*64+4*lane_0_4]));
        asm volatile("cp.async.commit_group;\n"::);
        for (int k = 0; k < 4; k++) {
            B[k] = dense[i&1][k+(lane_id&1)*4][warp_id*16+(lane_id>>1)];
        }
        // softmax and Matmul
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            int col_id = lane_id&7;
            float C[2] = {0.f};
            for (int iter = 0; iter < ((row_end-row_start+3)>>2); iter++) {
                int j = (lane_id>>3) + iter*4;
                int row_id = max(0, __fns(col_mask, 0, 1 + j));
                float alpha = std::numeric_limits<float>::lowest();
                int mask = (BitRowMask[row_start + j] & (1 << col_id)) > 0 && ((row_start + j) < row_end);
                if (mask)
                    alpha = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
                float max_alpha = alpha;
                for (int k = 1; k < 8; k *= 2) {
                    max_alpha = max(max_alpha, __shfl_xor_sync(FULL_MASK, max_alpha, 2 * k - 1, 8));
                }
                max_alpha = max(max_alpha, softmax[i&1][0][row_id]);
                softmax[(i+1)&1][0][row_id] = max_alpha;
                float sum_alpha = mask * __expf(alpha - max_alpha);
                float upper = sum_alpha;
                for (int k = 1; k < 8; k *= 2) {
                    sum_alpha += __shfl_xor_sync(FULL_MASK, sum_alpha, 2 * k - 1, 8);
                }
                float D_update = __expf(softmax[i&1][0][row_id] - max_alpha);
                sum_alpha += softmax[i&1][1][row_id] * D_update;
                softmax[(i+1)&1][1][row_id] = sum_alpha;
                D_update *= (softmax[i&1][1][row_id] + 1e-16f) / (sum_alpha + 1e-16f);
                float A = upper / (sum_alpha + 1e-16f);
                for (int k = 0; k < 2; k++) {
                    C[k] = __shfl_sync(FULL_MASK, D[(col_id&3)*2+k], row_id*2+(col_id>>2)) * D_update;
                }
                for (int k = 0; k < 16; k++)
                    C[k>>3] += __shfl_sync(FULL_MASK, A, (lane_id&24)+(k&7)) * __shfl_sync(FULL_MASK, B[k&3], col_id*4+(k>>2));
                int invite_group_id = __popc(col_mask&((1<<(lane_id>>1))-1));
                float is_set = (col_mask&(1<<(lane_id>>1)))&&(invite_group_id<4*(iter+1))&&(invite_group_id>=4*iter); 
                for (int k = 0; k < 8; k++)
                    D[k] = is_set * __shfl_sync(FULL_MASK, C[k&1], (k>>1)+(lane_id&1)*4+(invite_group_id&3)*8) + (1-is_set) * D[k];
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end - 1;
    float B[4];
    // load
    for (int k = 0; k < 4; k++) {
        B[k] = dense[i&1][k+(lane_id&1)*4][warp_id*16+(lane_id>>1)];
    }
    // softmax and Matmul
    {
        int row_start = BitMaskRowOffset[i];
        int row_end = BitMaskRowOffset[i+1];
        uint32_t col_mask = BitColMask[i];
        int col_id = lane_id&7;
        float C[2] = {0.f};
        for (int iter = 0; iter < ((row_end-row_start+3)>>2); iter++) {
            int j = (lane_id>>3) + iter*4;
            int row_id = max(0, __fns(col_mask, 0, 1 + j));
            float alpha = std::numeric_limits<float>::lowest();
            int mask = (BitRowMask[row_start + j] & (1 << col_id)) > 0 && ((row_start + j) < row_end);
            if (mask)
                alpha = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
            float max_alpha = alpha;
            for (int k = 1; k < 8; k *= 2) {
                max_alpha = max(max_alpha, __shfl_xor_sync(FULL_MASK, max_alpha, 2 * k - 1, 8));
            }
            max_alpha = max(max_alpha, softmax[i&1][0][row_id]);
            softmax[(i+1)&1][0][row_id] = max_alpha;
            float sum_alpha = mask * __expf(alpha - max_alpha);
            float upper = sum_alpha;
            for (int k = 1; k < 8; k *= 2) {
                sum_alpha += __shfl_xor_sync(FULL_MASK, sum_alpha, 2 * k - 1, 8);
            }
            float D_update = __expf(softmax[i&1][0][row_id] - max_alpha);
            sum_alpha += softmax[i&1][1][row_id] * D_update;
            softmax[(i+1)&1][1][row_id] = sum_alpha;
            D_update *= (softmax[i&1][1][row_id] + 1e-16f) / (sum_alpha + 1e-16f);
            float A = upper / (sum_alpha + 1e-16f);
            for (int k = 0; k < 2; k++) {
                C[k] = __shfl_sync(FULL_MASK, D[(col_id&3)*2+k], row_id*2+(col_id>>2)) * D_update;
            }
            for (int k = 0; k < 16; k++)
                C[k>>3] += __shfl_sync(FULL_MASK, A, (lane_id&24)+(k&7)) * __shfl_sync(FULL_MASK, B[k&3], col_id*4+(k>>2));
            int invite_group_id = __popc(col_mask&((1<<(lane_id>>1))-1));
            float is_set = col_mask&(1<<(lane_id>>1))&&(invite_group_id<4*(iter+1))&&(invite_group_id>=4*iter);
            for (int k = 0; k < 8; k++)
                D[k] = is_set * __shfl_sync(FULL_MASK, C[k&1], (k>>1)+(lane_id&1)*4+(invite_group_id&3)*8) + (1-is_set) * D[k];
        }
    }
    for (int k = 0; k < 2; k++) 
        *(float4*)(&out[(bid * 16 + (lane_id>>1)) * 64 + warp_id * 16 + (lane_id&1) * 8 + k*4]) = *(float4*)(&D[k*4]);
}

__global__ void gat_kernel_16x8_64(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    if(block_end == block_start) return;

    int lane_0_4 = lane_id&15;
    int thread_4_7 = threadIdx.x>>4;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense[2][8][64];
    __shared__ float softmax[2][2][16];
    
    int i = block_start;
    SparseAToX_idx[i&1][thread_4_7] = SparseAToX[i * 8 + thread_4_7];
    if (warp_id == 0) {
        softmax[thread_4_7][0][lane_0_4] = std::numeric_limits<float>::lowest();
        softmax[thread_4_7][1][lane_0_4] = 0.f;
    }
    int cur_addr = __cvta_generic_to_shared(&dense[i&1][thread_4_7][lane_0_4*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[SparseAToX_idx[i&1][thread_4_7]*64+4*lane_0_4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    uint32_t B[4];
    uint32_t A[4];

    for (i = block_start; i < (block_end - 1); i++) {
        // load and init
        SparseAToX_idx[(i+1)&1][thread_4_7] = SparseAToX[(i+1)*8 + thread_4_7];
        int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][thread_4_7][lane_0_4*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[SparseAToX_idx[(i+1)&1][thread_4_7]*64+4*lane_0_4]));
        asm volatile("cp.async.commit_group;\n"::);
        for (int k = 0; k < 4; k++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][warp_id*16+(k&1)*8+(lane_id>>2)]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
        }
        // softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            float alpha[4] = {std::numeric_limits<float>::lowest()};
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int k = 0; k < 4; k++) {
                int row_id = (lane_id>>2)+(k&1)*8;
                int col_id = (lane_id&3)+(k&2)*2;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                if (mask[k])
                    alpha[k] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
            }
            for (int k = 0; k < 2; k++) {
                alpha_max[k] = max(alpha[k], alpha[2+k]);
                for (int j = 1; j < 4; j*=2) {
                    alpha_max[k] = max(alpha_max[k], __shfl_xor_sync(FULL_MASK, alpha_max[k], 2*j-1, 4));
                }
                alpha_max[k] = max(alpha_max[k], softmax[i&1][0][(lane_id>>2)+k*8]);
                softmax[(i+1)&1][0][(lane_id>>2)+k*8] = alpha_max[k];
                alpha[k] = mask[k] * __expf(alpha[k] - alpha_max[k]);
                alpha[2+k] = mask[2+k] * __expf(alpha[2+k] - alpha_max[k]);
                alpha_sum[k] = alpha[k] + alpha[2+k];
                for (int j = 1; j < 4; j*=2) {
                    alpha_sum[k] += __shfl_xor_sync(FULL_MASK, alpha_sum[k], 2*j-1, 4);
                }
                alpha_sum[k] += softmax[i&1][1][(lane_id>>2)+k*8] * __expf(softmax[i&1][0][(lane_id>>2)+k*8] - alpha_max[k]);
                softmax[(i+1)&1][1][(lane_id>>2)+k*8] = alpha_sum[k];
                float rcp = 1.f/(alpha_sum[k]+1e-16f);
                alpha[k] *= rcp;
                alpha[2+k] *= rcp;
            }
            for (int k = 0; k < 4; k++) 
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[k]) : "f"(alpha[k]));
        }
        // Matmul
        {
            for (int k = 0; k < 2; k++) {
                float update = __expf(softmax[i&1][0][(lane_id>>2)+k*8]-softmax[(i+1)&1][0][(lane_id>>2)+k*8])*(softmax[i&1][1][(lane_id>>2)+k*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+k*8]+1e-16f);
                for (int j = 0; j < 4; j++) {
                    D[k*2+(j&1)+(j&2)*2] *= update;
                }
            }
            for (int k = 0; k < 2; k++) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                    : "=f"(D[k*4]), "=f"(D[k*4+1]), "=f"(D[k*4+2]), "=f"(D[k*4+3]) 
                    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                    "r"(B[k]), "r"(B[2+k])
                    "f"(D[k*4]), "f"(D[k*4+1]), "f"(D[k*4+2]), "f"(D[k*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end - 1;
    for (int k = 0; k < 4; k++) {
        if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][warp_id*16+(k&1)*8+(lane_id>>2)]));
        else
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
    }
    // softmax
    {
        int row_start = BitMaskRowOffset[i];
        int row_end = BitMaskRowOffset[i+1];
        float alpha[4] = {std::numeric_limits<float>::lowest()};
        uint32_t col_mask = BitColMask[i];
        float mask[4], alpha_max[2], alpha_sum[2];
        for (int k = 0; k < 4; k++) {
            int row_id = (lane_id>>2)+(k&1)*8;
            int col_id = (lane_id&3)+(k&2)*2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
            if (mask[k])
                alpha[k] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
        }
        for (int k = 0; k < 2; k++) {
            alpha_max[k] = max(alpha[k], alpha[2+k]);
            for (int j = 1; j < 4; j*=2) {
                alpha_max[k] = max(alpha_max[k], __shfl_xor_sync(FULL_MASK, alpha_max[k], 2*j-1, 4));
            }
            alpha_max[k] = max(alpha_max[k], softmax[i&1][0][(lane_id>>2)+k*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+k*8] = alpha_max[k];
            alpha[k] = mask[k] * __expf(alpha[k] - alpha_max[k]);
            alpha[2+k] = mask[2+k] * __expf(alpha[2+k] - alpha_max[k]);
            alpha_sum[k] = alpha[k] + alpha[2+k];
            for (int j = 1; j < 4; j*=2) {
                alpha_sum[k] += __shfl_xor_sync(FULL_MASK, alpha_sum[k], 2*j-1, 4);
            }
            alpha_sum[k] += softmax[i&1][1][(lane_id>>2)+k*8] * __expf(softmax[i&1][0][(lane_id>>2)+k*8] - alpha_max[k]);
            softmax[(i+1)&1][1][(lane_id>>2)+k*8] = alpha_sum[k];
            float rcp = 1.f/(alpha_sum[k]+1e-16f);
            alpha[k] *= rcp;
            alpha[2+k] *= rcp;
        }
        for (int k = 0; k < 4; k++) 
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[k]) : "f"(alpha[k]));
    }
    // Matmul
    {
        for (int k = 0; k < 2; k++) {
            float update = __expf(softmax[i&1][0][(lane_id>>2)+k*8]-softmax[(i+1)&1][0][(lane_id>>2)+k*8])*(softmax[i&1][1][(lane_id>>2)+k*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+k*8]+1e-16f);
            for (int j = 0; j < 4; j++) {
                D[k*2+(j&1)+(j&2)*2] *= update;
            }
        }
        for (int k = 0; k < 2; k++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                : "=f"(D[k*4]), "=f"(D[k*4+1]), "=f"(D[k*4+2]), "=f"(D[k*4+3]) 
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(B[k]), "r"(B[2+k])
                "f"(D[k*4]), "f"(D[k*4+1]), "f"(D[k*4+2]), "f"(D[k*4+3]));
        }
    }
    for (int k = 0; k < 4; k++) 
        *(float2*)(&out[(bid*16+(lane_id>>2)+(k&1)*8)*64 + warp_id*16 + (lane_id&3)*2]) = *(float2*)(&D[k*2]);
}

__global__ void gat_kernel_16x16_64(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint16_t* __restrict__ BitRowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    if(block_end == block_start) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float dense[2][16][64];
    __shared__ float softmax[2][2][16];

    int i = block_start;
    SparseAToX_idx[i&1][threadIdx.x>>3] = SparseAToX[i * 16 + (threadIdx.x>>3)];
    if (warp_id == 0) {
        softmax[lane_id>>4][0][lane_id&15] = std::numeric_limits<float>::lowest();
        softmax[lane_id>>4][1][lane_id&15] = 0.f;
    }
    for (int j = 0; j < 2; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense[i&1][j+(threadIdx.x>>4)*2][(lane_id&15)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[i&1][j+(threadIdx.x>>4)*2], node_num-1)*64+(lane_id&15)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    uint32_t A[8], B[8];
    for (int i = block_start; i < (block_end - 1); i++) {
        // load and init
        SparseAToX_idx[(i+1)&1][threadIdx.x>>3] = SparseAToX[(i + 1) * 16 + (threadIdx.x>>3)];
        for (int j = 0; j < 2; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j+(threadIdx.x>>4)*2][(lane_id&15)*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j+(threadIdx.x>>4)*2], node_num-1)*64+(lane_id&15)*4]));
        }
        for (int j = 0; j < 8; j++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(j&6)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense[i&1][(lane_id&3)+(j&6)*2][warp_id*16+(j&1)*8+(lane_id>>2)]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
        }
        // softmax
        {
            int row_start = BitMaskRowOffset[i];
            // int row_end = BitMaskRowOffset[i+1];
            float alpha[8] = {std::numeric_limits<float>::lowest()};
            uint32_t col_mask = BitColMask[i];
            float mask[8], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 8; j++) {
                int row_id = (lane_id>>2)+(j&1)*8;
                int col_id = (lane_id&3)+(j&2)*2;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                mask[j] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                if (mask[j])
                    alpha[j] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
            }
            for (int j = 0; j < 2; j++) {
                alpha_max[j] = max(alpha[j], alpha[2+j]);
                alpha_max[j] = max(alpha_max[j], max(alpha[4+j], alpha[6+j]));
                for (int k = 1; k < 4; k*=2) {
                    alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
                }
                alpha_max[j] = max(alpha_max[j], softmax[i&1][0][(lane_id>>2)+j*8]);
                softmax[(i+1)&1][0][(lane_id>>2)+j*8] = alpha_max[j];
                alpha[j] = mask[j] * __expf(alpha[j] - alpha_max[j]);
                alpha[2+j] = mask[2+j] * __expf(alpha[2+j] - alpha_max[j]);
                alpha[4+j] = mask[4+j] * __expf(alpha[4+j] - alpha_max[j]);
                alpha[6+j] = mask[6+j] * __expf(alpha[6+j] - alpha_max[j]);
                alpha_sum[j] = alpha[j] + alpha[2+j] + alpha[4+j] + alpha[6+j];
                for (int k = 1; k < 4; k*=2) {
                    alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
                }
                alpha_sum[j] += softmax[i&1][1][(lane_id>>2)+j*8] * __expf(softmax[i&1][0][(lane_id>>2)+j*8] - alpha_max[j]);
                softmax[(i+1)&1][1][(lane_id>>2)+j*8] = alpha_sum[j];
                float rcp = 1.f/(alpha_sum[j]+1e-16f);
                alpha[j] *= rcp;
                alpha[2+j] *= rcp;
                alpha[4+j] *= rcp;
                alpha[6+j] *= rcp;
            }
            for (int j = 0; j < 8; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
            }
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id>>2)+j*8]-softmax[(i+1)&1][0][(lane_id>>2)+j*8])*(softmax[i&1][1][(lane_id>>2)+j*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) {
                    D[j*2+(j&1)+(j&2)*2] *= update;
                }
            }
            for (int j = 0; j < 4; j++) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                    : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                    "r"(B[j]), "r"(B[2+j])
                    "f"(D[(j&1)*4]), "f"(D[(j&1)*4+1]), "f"(D[(j&1)*4+2]), "f"(D[(j&1)*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    } 
    i = block_end - 1;
    for (int j = 0; j < 8; j++) {
        if (SparseAToX_idx[i&1][(lane_id&3)+(j&6)*2] < node_num)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense[i&1][(lane_id&3)+(j&6)*2][warp_id*16+(j&1)*8+(lane_id>>2)]));
        else
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
    }
    // softmax
    {
        int row_start = BitMaskRowOffset[i];
        // int row_end = BitMaskRowOffset[i+1];
        float alpha[8] = {std::numeric_limits<float>::lowest()};
        uint32_t col_mask = BitColMask[i];
        float mask[8], alpha_max[2], alpha_sum[2];
        for (int j = 0; j < 8; j++) {
            int row_id = (lane_id>>2)+(j&1)*8;
            int col_id = (lane_id&3)+(j&2)*2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            mask[j] = (col_mask>>row_id)&(row_mask>>col_id)&1;
            if (mask[j])
                alpha[j] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
        }
        for (int j = 0; j < 2; j++) {
            alpha_max[j] = max(alpha[j], alpha[2+j]);
            alpha_max[j] = max(alpha_max[j], max(alpha[4+j], alpha[6+j]));
            for (int k = 1; k < 4; k*=2) {
                alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
            }
            alpha_max[j] = max(alpha_max[j], softmax[i&1][0][(lane_id>>2)+j*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+j*8] = alpha_max[j];
            alpha[j] = mask[j] * __expf(alpha[j] - alpha_max[j]);
            alpha[2+j] = mask[2+j] * __expf(alpha[2+j] - alpha_max[j]);
            alpha[4+j] = mask[4+j] * __expf(alpha[4+j] - alpha_max[j]);
            alpha[6+j] = mask[6+j] * __expf(alpha[6+j] - alpha_max[j]);
            alpha_sum[j] = alpha[j] + alpha[2+j] + alpha[4+j] + alpha[6+j];
            for (int k = 1; k < 4; k*=2) {
                alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
            }
            alpha_sum[j] += softmax[i&1][1][(lane_id>>2)+j*8] * __expf(softmax[i&1][0][(lane_id>>2)+j*8] - alpha_max[j]);
            softmax[(i+1)&1][1][(lane_id>>2)+j*8] = alpha_sum[j];
            float rcp = 1.f/(alpha_sum[j]+1e-16f);
            alpha[j] *= rcp;
            alpha[2+j] *= rcp;
            alpha[4+j] *= rcp;
            alpha[6+j] *= rcp;
        }
        for (int j = 0; j < 8; j++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
    }
    // Matmul
    {
        for (int j = 0; j < 2; j++) {
            float update = __expf(softmax[i&1][0][(lane_id>>2)+j*8]-softmax[(i+1)&1][0][(lane_id>>2)+j*8])*(softmax[i&1][1][(lane_id>>2)+j*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+j*8]+1e-16f);
            for (int k = 0; k < 4; k++) {
                D[j*2+(j&1)+(j&2)*2] *= update;
            }
        }
        for (int j = 0; j < 4; j++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(B[j]), "r"(B[2+j])
                "f"(D[(j&1)*4]), "f"(D[(j&1)*4+1]), "f"(D[(j&1)*4+2]), "f"(D[(j&1)*4+3]));
        }
    }
    for (int j = 0; j < 4; j++) 
        *(float2*)(&out[(bid*16+(lane_id>>2)+(j&1)*8)*64 + warp_id*16 + (lane_id&3)*2]) = *(float2*)(&D[j*2]);
}

at::Tensor GAT_short(
    at::Tensor feature,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor lin_weight,
    at::Tensor att_i, 
    at::Tensor att_j,
    int num_heads,
    int out_feats
) {
    int num_nodes = feature.size(0);
    auto feats = torch::mm(feature, lin_weight.t());
    auto alpha_i = torch::matmul(feats, att_i.squeeze());
    auto alpha_j = torch::matmul(feats, att_j.squeeze());
    auto output = at::empty({num_nodes, out_feats}, feature.options());

    int threads = 128;
    int blocks = (num_nodes + 15) / 16;
    gat_kernel_16x16_64<<<blocks, threads>>>(
        RowWindowOffsets.data_ptr<int>(),
        SparseAToX.data_ptr<int>(),
        BitMaskRowOffset.data_ptr<int>(),
        (uint16_t*)BitColMask.data_ptr<uint8_t>(),
        (uint16_t*)BitRowMask.data_ptr<uint8_t>(),
        feats.data_ptr<float>(),
        alpha_i.data_ptr<float>(),
        alpha_j.data_ptr<float>(),
        output.data_ptr<float>(),
        num_nodes);
    return output;
}