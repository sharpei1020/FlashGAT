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
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense_X[2][8][32];
    __shared__ float softmax[2][2][16];

    uint32_t D[8], A[2], B[4], E[1] = {0xeeee4444};
    float C[8] = {0.f};
    for (int i=0; i<8; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+(threadIdx.x>>2), node_num-1)*32+i*4+(lane_id&3)]/x_norm[min(bid*16+(threadIdx.x>>2), node_num-1)]));
    }
    int i = block_start;
    int cur_addr;
    SparseAToX_idx[i&1][(threadIdx.x>>3)] = SparseAToX[i*8+(threadIdx.x>>3)];
    cur_addr = __cvta_generic_to_shared(&dense_X[i&1][threadIdx.x>>3][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i&1][threadIdx.x>>3], node_num-1)*32+(lane_id&7)*4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (; i<(block_end-1); i++) {
        SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)] = SparseAToX[(i+1)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][threadIdx.x>>3][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)&1][threadIdx.x>>3], node_num-1)*32+(lane_id&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[4] = {0.f};
        for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 4; l++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                    dense_X[i&1][lane_id>>2][((lane_id&1)*8+((lane_id&2)>>1)+l*2+k*16+(lane_id>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][lane_id>>2], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum;
            int row_id = (lane_id>>2)+warp_id*8;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 2; k++) {
                int col_id = (lane_id&3)*2+k;
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[k] = min(-1.f+2*mask[k], alpha[k]+alpha[k+2])*beta[0];
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][(lane_id>>2)+warp_id*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+warp_id*8] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 2; k++) {
                alpha[k] = mask[k] * __expf(alpha[k]-alpha_max);
                alpha_sum += alpha[k];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][(lane_id>>2)+warp_id*8] * __expf(softmax[i&1][0][(lane_id>>2)+warp_id*8] - alpha_max);
            softmax[(i+1)&1][1][(lane_id>>2)+warp_id*8] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 2; k++) {
                alpha[k] *= rcp;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id&3)*2+j+warp_id*8] - softmax[(i+1)&1][0][(lane_id&3)*2+j+warp_id*8])
                        *(softmax[i&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int l = 0; l < 4; l++) {
                    // if (SparseAToX_idx[i&1][(lane_id&3)*2+(l>>1)] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][(lane_id&3)*2+(l>>1)][((lane_id>>2)+(l>>1)*4+((l&1)+(lane_id&3))*8+j*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "r"(A[0]), "r"(A[1]), 
                    "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-1)) {
        // SDDMM
        float alpha[4] = {0.f};
        for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 4; l++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                    dense_X[i&1][lane_id>>2][((lane_id&1)*8+((lane_id&2)>>1)+l*2+k*16+(lane_id>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][lane_id>>2], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum;
            int row_id = (lane_id>>2)+warp_id*8;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 2; k++) {
                int col_id = (lane_id&3)*2+k;
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[k] = min(-1.f+2*mask[k], alpha[k]+alpha[k+2])*beta[0];
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][(lane_id>>2)+warp_id*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+warp_id*8] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 2; k++) {
                alpha[k] = mask[k] * __expf(alpha[k]-alpha_max);
                alpha_sum += alpha[k];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][(lane_id>>2)+warp_id*8] * __expf(softmax[i&1][0][(lane_id>>2)+warp_id*8] - alpha_max);
            softmax[(i+1)&1][1][(lane_id>>2)+warp_id*8] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 2; k++) {
                alpha[k] *= rcp;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id&3)*2+j+warp_id*8] - softmax[(i+1)&1][0][(lane_id&3)*2+j+warp_id*8])
                        *(softmax[i&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int l = 0; l < 4; l++) {
                    // if (SparseAToX_idx[i&1][(lane_id&3)*2+(l>>1)] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][(lane_id&3)*2+(l>>1)][((lane_id>>2)+(l>>1)*4+((l&1)+(lane_id&3))*8+j*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "r"(A[0]), "r"(A[1]), 
                    "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
            }
        }
    }
    for (int j = 0; j < 8; j++)
        if (bid*16+(lane_id&3)*2+(j&1)+warp_id*8<node_num)
            output[(bid*16+(lane_id&3)*2+(j&1)+warp_id*8)*32+(j>>1)*8+(lane_id>>2)] = C[j];
}

__global__ void agnn_kernel_16x8_32_3(
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
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[3][8];
    __shared__ float dense_X[3][8][32];
    __shared__ float softmax[2][2][16];
    __shared__ float sparse_A[2][16][8];

    int lane_front = lane_id>>2;
    int lane_end = lane_id&3;

    uint32_t D[8], A[4], B[4];
    float C[8] = {0.f};
    for (int i=0; i<8; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+(threadIdx.x>>2), node_num-1)*32+i*4+lane_end]/x_norm[min(bid*16+(threadIdx.x>>2), node_num-1)]));
    }
    int i = block_start;
    SparseAToX_idx[i%3][threadIdx.x>>3] = SparseAToX[i*8+(threadIdx.x>>3)];
    int cur_addr = __cvta_generic_to_shared(&dense_X[i%3][threadIdx.x>>3][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i%3][threadIdx.x>>3], node_num-1)*32+(lane_id&7)*4]));
    asm volatile("cp.async.commit_group;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if ((i+1) < block_end) {
        SparseAToX_idx[(i+1)%3][threadIdx.x>>3] = SparseAToX[(i+1)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)%3][threadIdx.x>>3][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)%3][threadIdx.x>>3], node_num-1)*32+(lane_id&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
    }
    float alpha[4] = {0.f};
    uint32_t E[1] = {0xeeee4444};
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 4; k++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_X[i%3][lane_front][(j*16+lane_front*4+k*2+(lane_end>>1)+(lane_end&1)*8)&31]/x_norm[min(SparseAToX_idx[i%3][lane_front], node_num-1)]));
        }
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
            : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
            : "r"(D[j*4]), "r"(D[j*4+2]), "r"(D[j*4+1]), "r"(D[j*4+3]), 
            "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
    }
    for (int j = 0; j < 2; j++) {
        sparse_A[i&1][lane_front+warp_id*8][(lane_end*2+(j&1)+(lane_front>>2))&7] = alpha[j]+alpha[j+2];
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (; i<(block_end-2); i++) {
        SparseAToX_idx[(i+2)%3][threadIdx.x>>3] = SparseAToX[(i+2)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+2)%3][threadIdx.x>>3][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+2)%3][threadIdx.x>>3], node_num-1)*32+(lane_id&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
        float alpha[4] = {0.f};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_X[(i+1)%3][lane_front][(j*16+lane_front*4+k*2+(lane_end>>1)+(lane_end&1)*8)&31]/x_norm[min(SparseAToX_idx[(i+1)%3][lane_front], node_num-1)]));
            }
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[j*4]), "r"(D[j*4+2]), "r"(D[j*4+1]), "r"(D[j*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        for (int j = 0; j < 2; j++) {
            sparse_A[(i+1)&1][lane_front+warp_id*8][(lane_end*2+(j&1)+(lane_front>>2))&7] = alpha[j]+alpha[j+2];
        }
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 2; j++) {
                int row_id = lane_front+j*8;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                for (int k = 0; k < 2; k++) {
                    int col_id = lane_end*2+k;
                    mask[2*j+k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                    alpha[2*j+k] = min(-1.f+2*mask[2*j+k], sparse_A[i&1][lane_front+j*8][(lane_end*2+k+(lane_front>>2))&7])*beta[0];
                }
                alpha_max[j] = max(alpha[j*2], alpha[j*2+1]);
                for (int k = 1; k < 4; k<<=1) {
                    alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
                }
                alpha_max[j] = max(alpha_max[j], softmax[i&1][0][lane_front+j*8]);
                softmax[(i+1)&1][0][lane_front+j*8] = alpha_max[j];
                alpha[j*2] = mask[j*2] * __expf(alpha[j*2]-alpha_max[j]);
                alpha[j*2+1] = mask[j*2+1] * __expf(alpha[j*2+1]-alpha_max[j]);
                alpha_sum[j] = alpha[j*2] + alpha[j*2+1];
                for (int k = 1; k < 4; k<<=1) {
                    alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
                }
                alpha_sum[j] += softmax[i&1][1][lane_front+j*8] * __expf(softmax[i&1][0][lane_front+j*8] - alpha_max[j]);
                softmax[(i+1)&1][1][lane_front+j*8] = alpha_sum[j];
                float rcp = 1.f/(alpha_sum[j]+1e-16f);
                alpha[j*2] *= rcp;
                alpha[j*2+1] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
                        *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j*2+(k&2)*2+(k&1)] *= update;
            }
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < 2; j++) {
                    // if (SparseAToX_idx[i%3][lane_end*2+j] < node_num) 
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i%3][lane_end*2+j][(lane_front+(lane_end*2+j)*4+k*8+warp_id*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));            
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                    : "r"(A[0]), "r"(A[2]), "r"(A[1]), "r"(A[3]), 
                    "r"(B[0]), "r"(B[1]), 
                    "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-2))
    {
        float alpha[4] = {0.f};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_X[(i+1)%3][lane_front][(j*16+lane_front*4+k*2+(lane_end>>1)+(lane_end&1)*8)&31]/x_norm[min(SparseAToX_idx[(i+1)%3][lane_front], node_num-1)]));
            }
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[j*4]), "r"(D[j*4+2]), "r"(D[j*4+1]), "r"(D[j*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        for (int j = 0; j < 2; j++) {
            sparse_A[(i+1)&1][lane_front+warp_id*8][(lane_end*2+(j&1)+(lane_front>>2))&7] = alpha[j]+alpha[j+2];
        }
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 2; j++) {
                int row_id = lane_front+j*8;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                for (int k = 0; k < 2; k++) {
                    int col_id = lane_end*2+k;
                    mask[2*j+k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                    alpha[2*j+k] = min(-1.f+2*mask[2*j+k], sparse_A[i&1][lane_front+j*8][(lane_end*2+k+(lane_front>>2))&7])*beta[0];
                }
                alpha_max[j] = max(alpha[j*2], alpha[j*2+1]);
                for (int k = 1; k < 4; k<<=1) {
                    alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
                }
                alpha_max[j] = max(alpha_max[j], softmax[i&1][0][lane_front+j*8]);
                softmax[(i+1)&1][0][lane_front+j*8] = alpha_max[j];
                alpha[j*2] = mask[j*2] * __expf(alpha[j*2]-alpha_max[j]);
                alpha[j*2+1] = mask[j*2+1] * __expf(alpha[j*2+1]-alpha_max[j]);
                alpha_sum[j] = alpha[j*2] + alpha[j*2+1];
                for (int k = 1; k < 4; k<<=1) {
                    alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
                }
                alpha_sum[j] += softmax[i&1][1][lane_front+j*8] * __expf(softmax[i&1][0][lane_front+j*8] - alpha_max[j]);
                softmax[(i+1)&1][1][lane_front+j*8] = alpha_sum[j];
                float rcp = 1.f/(alpha_sum[j]+1e-16f);
                alpha[j*2] *= rcp;
                alpha[j*2+1] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
                        *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j*2+(k&2)*2+(k&1)] *= update;
            }
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < 2; j++) {
                    // if (SparseAToX_idx[i%3][lane_end*2+j] < node_num) 
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i%3][lane_end*2+j][(lane_front+(lane_end*2+j)*4+k*8+warp_id*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));            
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                    : "r"(A[0]), "r"(A[2]), "r"(A[1]), "r"(A[3]), 
                    "r"(B[0]), "r"(B[1]), 
                    "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
            }
        }
        i++;
        __syncthreads();
    }
    if (i == (block_end-1)) {
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 2; j++) {
                int row_id = lane_front+j*8;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                for (int k = 0; k < 2; k++) {
                    int col_id = lane_end*2+k;
                    mask[2*j+k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                    alpha[2*j+k] = min(-1.f+2*mask[2*j+k], sparse_A[i&1][lane_front+j*8][(lane_end*2+k+(lane_front>>2))&7])*beta[0];
                }
                alpha_max[j] = max(alpha[j*2], alpha[j*2+1]);
                for (int k = 1; k < 4; k<<=1) {
                    alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
                }
                alpha_max[j] = max(alpha_max[j], softmax[i&1][0][lane_front+j*8]);
                softmax[(i+1)&1][0][lane_front+j*8] = alpha_max[j];
                alpha[j*2] = mask[j*2] * __expf(alpha[j*2]-alpha_max[j]);
                alpha[j*2+1] = mask[j*2+1] * __expf(alpha[j*2+1]-alpha_max[j]);
                alpha_sum[j] = alpha[j*2] + alpha[j*2+1];
                for (int k = 1; k < 4; k<<=1) {
                    alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
                }
                alpha_sum[j] += softmax[i&1][1][lane_front+j*8] * __expf(softmax[i&1][0][lane_front+j*8] - alpha_max[j]);
                softmax[(i+1)&1][1][lane_front+j*8] = alpha_sum[j];
                float rcp = 1.f/(alpha_sum[j]+1e-16f);
                alpha[j*2] *= rcp;
                alpha[j*2+1] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
                        *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j*2+(k&2)*2+(k&1)] *= update;
            }
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < 2; j++) {
                    // if (SparseAToX_idx[i%3][lane_end*2+j] < node_num) 
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i%3][lane_end*2+j][(lane_front+(lane_end*2+j)*4+k*8+warp_id*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));            
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                    : "r"(A[0]), "r"(A[2]), "r"(A[1]), "r"(A[3]), 
                    "r"(B[0]), "r"(B[1]), 
                    "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
            }
        }
    }
    for (int j = 0; j < 4; j++)
        if (bid*16+lane_front+(j&1)*8<node_num) 
            *(float2*)(&output[(bid*16+lane_front+(j&1)*8)*32+warp_id*16+(j&2)*4+lane_end*2]) = *(float2*)(&C[j*2]);
}

__global__ void agnn_kernel_16x16_32(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint16_t* __restrict__ BitRowMask,
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
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float dense_X[2][16][32];
    __shared__ float softmax[2][2][16];

    uint32_t D[8], A[4], B[4], E[1] = {0xeeee4444};
    float C[8] = {0.f};
    for (int i=0; i<8; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+(threadIdx.x>>2), node_num-1)*32+i*4+(lane_id&3)]/x_norm[min(bid*16+(threadIdx.x>>2), node_num-1)]));
    }
    int i = block_start;
    int cur_addr;
    for (int j = 0; j < 2; j++) {
        SparseAToX_idx[i&1][(threadIdx.x>>3)+j*8] = SparseAToX[i*16+(threadIdx.x>>3)+j*8];
        cur_addr = __cvta_generic_to_shared(&dense_X[i&1][(threadIdx.x>>3)+j*8][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i&1][(threadIdx.x>>3)+j*8], node_num-1)*32+(lane_id&7)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (; i<(block_end-1); i++) {
        for (int j = 0; j < 2; j++) {
            SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*8] = SparseAToX[(i+1)*16+(threadIdx.x>>3)+j*8];
            cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][(threadIdx.x>>3)+j*8][(((lane_id&7)+(threadIdx.x>>3))*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*8], node_num-1)*32+(lane_id&7)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[8] = {0.f};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 4; l++)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                        dense_X[i&1][(lane_id>>2)+j*8][((lane_id&1)*8+((lane_id&2)>>1)+l*2+k*16+(lane_id>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(lane_id>>2)+j*8], node_num-1)]));
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                    "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                    : "=f"(alpha[j*4]), "=f"(alpha[j*4+1]), "=f"(alpha[j*4+2]), "=f"(alpha[j*4+3])
                    : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                    "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "f"(alpha[j*4]), "f"(alpha[j*4+1]), "f"(alpha[j*4+2]), "f"(alpha[j*4+3]), "r"(E[0]));
            }
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max, alpha_sum;
            int row_id = (lane_id>>2)+warp_id*8;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 4; k++) {
                int col_id = (lane_id&3)*2+(k&2)*4+(k&1);
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[(k&2)*2+(k&1)] = min(-1.f+2*mask[k], alpha[(k&2)*2+(k&1)]+alpha[(k&2)*2+(k&1)+2])*beta[0];
            }
            alpha_max = max(max(alpha[0], alpha[1]), max(alpha[4], alpha[5]));
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][(lane_id>>2)+warp_id*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+warp_id*8] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] = mask[k] * __expf(alpha[(k&2)*2+(k&1)]-alpha_max);
                alpha_sum += alpha[(k&2)*2+(k&1)];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][(lane_id>>2)+warp_id*8] * __expf(softmax[i&1][0][(lane_id>>2)+warp_id*8] - alpha_max);
            softmax[(i+1)&1][1][(lane_id>>2)+warp_id*8] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[(j&2)*2+(j&1)]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id&3)*2+j+warp_id*8] - softmax[(i+1)&1][0][(lane_id&3)*2+j+warp_id*8])
                        *(softmax[i&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 4; l++) {
                        // if (SparseAToX_idx[i&1][k*8+(lane_id&3)*2+(l>>1)] < node_num)
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][k*8+(lane_id&3)*2+(l>>1)][((lane_id>>2)+(l>>1)*4+((l&1)+(lane_id&3))*8+j*16)&31]));
                        // else
                        //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                        : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                        "r"(A[k*2]), "r"(A[k*2+1]), 
                        "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-1)) {
        // SDDMM
        float alpha[8] = {0.f};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 4; l++)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                        dense_X[i&1][(lane_id>>2)+j*8][((lane_id&1)*8+((lane_id&2)>>1)+l*2+k*16+(lane_id>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(lane_id>>2)+j*8], node_num-1)]));
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                    "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                    : "=f"(alpha[j*4]), "=f"(alpha[j*4+1]), "=f"(alpha[j*4+2]), "=f"(alpha[j*4+3])
                    : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                    "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "f"(alpha[j*4]), "f"(alpha[j*4+1]), "f"(alpha[j*4+2]), "f"(alpha[j*4+3]), "r"(E[0]));
            }
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max, alpha_sum;
            int row_id = (lane_id>>2)+warp_id*8;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 4; k++) {
                int col_id = (lane_id&3)*2+(k&2)*4+(k&1);
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[(k&2)*2+(k&1)] = min(-1.f+2*mask[k], alpha[(k&2)*2+(k&1)]+alpha[(k&2)*2+(k&1)+2])*beta[0];
            }
            alpha_max = max(max(alpha[0], alpha[1]), max(alpha[4], alpha[5]));
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][(lane_id>>2)+warp_id*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+warp_id*8] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] = mask[k] * __expf(alpha[(k&2)*2+(k&1)]-alpha_max);
                alpha_sum += alpha[(k&2)*2+(k&1)];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][(lane_id>>2)+warp_id*8] * __expf(softmax[i&1][0][(lane_id>>2)+warp_id*8] - alpha_max);
            softmax[(i+1)&1][1][(lane_id>>2)+warp_id*8] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[(j&2)*2+(j&1)]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id&3)*2+j+warp_id*8] - softmax[(i+1)&1][0][(lane_id&3)*2+j+warp_id*8])
                        *(softmax[i&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id&3)*2+j+warp_id*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 4; l++) {
                        // if (SparseAToX_idx[i&1][k*8+(lane_id&3)*2+(l>>1)] < node_num)
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][k*8+(lane_id&3)*2+(l>>1)][((lane_id>>2)+(l>>1)*4+((l&1)+(lane_id&3))*8+j*16)&31]));
                        // else
                        //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                        : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                        "r"(A[k*2]), "r"(A[k*2+1]), 
                        "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
                }
            }
        }
    }
    for (int j = 0; j < 8; j++)
        if (bid*16+(lane_id&3)*2+(j&1)+warp_id*8<node_num)
            output[(bid*16+(lane_id&3)*2+(j&1)+warp_id*8)*32+(j>>1)*8+(lane_id>>2)] = C[j];
}


__global__ void agnn_kernel_8x16_32(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint16_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float dense_X[2][16][32];
    __shared__ float softmax[2][2][8];

    uint32_t D[8], A[4], B[4], E[1] = {0xeeee4444};
    float C[8] = {0.f};
    for (int i=0; i<8; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*8+(threadIdx.x>>2), node_num-1)*32+i*4+(threadIdx.x&3)]/x_norm[min(bid*8+(threadIdx.x>>2), node_num-1)]));
    }
    int i = block_start;
    int cur_addr;
    for (int j = 0; j < 4; j++) {
        SparseAToX_idx[i&1][(threadIdx.x>>3)+j*4] = SparseAToX[i*16+(threadIdx.x>>3)+j*4];
        cur_addr = __cvta_generic_to_shared(&dense_X[i&1][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i&1][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (; i<(block_end-1); i++) {
        for (int j = 0; j < 4; j++) {
            SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*4] = SparseAToX[(i+1)*16+(threadIdx.x>>3)+j*4];
            cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[8] = {0.f};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 4; l++)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                        dense_X[i&1][(threadIdx.x>>2)+j*8][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+l*2+k*16+(threadIdx.x>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(threadIdx.x>>2)+j*8], node_num-1)]));
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                    "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                    : "=f"(alpha[j*4]), "=f"(alpha[j*4+1]), "=f"(alpha[j*4+2]), "=f"(alpha[j*4+3])
                    : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                    "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "f"(alpha[j*4]), "f"(alpha[j*4+1]), "f"(alpha[j*4+2]), "f"(alpha[j*4+3]), "r"(E[0]));
            }
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max, alpha_sum;
            int row_id = threadIdx.x>>2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 4; k++) {
                int col_id = (threadIdx.x&3)*2+(k&2)*4+(k&1);
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[(k&2)*2+(k&1)] = min(-1.f+2*mask[k], alpha[(k&2)*2+(k&1)]+alpha[(k&2)*2+(k&1)+2])*beta[0];
            }
            alpha_max = max(max(alpha[0], alpha[1]), max(alpha[4], alpha[5]));
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][threadIdx.x>>2]);
            softmax[(i+1)&1][0][threadIdx.x>>2] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] = mask[k] * __expf(alpha[(k&2)*2+(k&1)]-alpha_max);
                alpha_sum += alpha[(k&2)*2+(k&1)];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>2] * __expf(softmax[i&1][0][threadIdx.x>>2] - alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>2] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 4; k++) {
                alpha[(k&2)*2+(k&1)] *= rcp;
            }
            for (int j = 0; j < 4; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[(j&2)*2+(j&1)]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j] - softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 4; l++) {
                        // if (SparseAToX_idx[i&1][k*8+(threadIdx.x&3)*2+(l>>1)] < node_num)
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][k*8+(threadIdx.x&3)*2+(l>>1)][((threadIdx.x>>2)+(l>>1)*4+((l&1)+(threadIdx.x&3))*8+j*16)&31]));
                        // else
                        //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                        : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                        "r"(A[k*2]), "r"(A[k*2+1]), 
                        "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-1)) {
       // SDDMM
       float alpha[8] = {0.f};
       for (int j = 0; j < 2; j++) {
           for (int k = 0; k < 2; k++) {
               for (int l = 0; l < 4; l++)
                   asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                       dense_X[i&1][(threadIdx.x>>2)+j*8][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+l*2+k*16+(threadIdx.x>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(threadIdx.x>>2)+j*8], node_num-1)]));
               asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                   "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                   : "=f"(alpha[j*4]), "=f"(alpha[j*4+1]), "=f"(alpha[j*4+2]), "=f"(alpha[j*4+3])
                   : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                   "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                   "f"(alpha[j*4]), "f"(alpha[j*4+1]), "f"(alpha[j*4+2]), "f"(alpha[j*4+3]), "r"(E[0]));
           }
       }
       // Softmax
       {
           int row_start = BitMaskRowOffset[i];
           int row_end = BitMaskRowOffset[i+1];
           uint32_t col_mask = BitColMask[i];
           float mask[4], alpha_max, alpha_sum;
           int row_id = threadIdx.x>>2;
           uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
           for (int k = 0; k < 4; k++) {
               int col_id = (threadIdx.x&3)*2+(k&2)*4+(k&1);
               mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
               alpha[(k&2)*2+(k&1)] = min(-1.f+2*mask[k], alpha[(k&2)*2+(k&1)]+alpha[(k&2)*2+(k&1)+2])*beta[0];
           }
           alpha_max = max(max(alpha[0], alpha[1]), max(alpha[4], alpha[5]));
           for (int k = 1; k < 4; k<<=1) {
               alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
           }
           alpha_max = max(alpha_max, softmax[i&1][0][threadIdx.x>>2]);
           softmax[(i+1)&1][0][threadIdx.x>>2] = alpha_max;
           alpha_sum = 0;
           for (int k = 0; k < 4; k++) {
               alpha[(k&2)*2+(k&1)] = mask[k] * __expf(alpha[(k&2)*2+(k&1)]-alpha_max);
               alpha_sum += alpha[(k&2)*2+(k&1)];
           }
           for (int k = 1; k < 4; k<<=1) {
               alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
           }
           alpha_sum += softmax[i&1][1][threadIdx.x>>2] * __expf(softmax[i&1][0][threadIdx.x>>2] - alpha_max);
           softmax[(i+1)&1][1][threadIdx.x>>2] = alpha_sum;
           float rcp = 1.f/(alpha_sum+1e-16f);
           for (int k = 0; k < 4; k++) {
               alpha[(k&2)*2+(k&1)] *= rcp;
           }
           for (int j = 0; j < 4; j++)
               asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[(j&2)*2+(j&1)]));
       }
       // Matmul
       {
           for (int j = 0; j < 2; j++) {
               float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j] - softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                       *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
               for (int k = 0; k < 4; k++) 
                   C[j+k*2] *= update;
           }
           for (int j = 0; j < 2; j++) {
               for (int k = 0; k < 2; k++) {
                   for (int l = 0; l < 4; l++) {
                    //    if (SparseAToX_idx[i&1][k*8+(threadIdx.x&3)*2+(l>>1)] < node_num)
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][k*8+(threadIdx.x&3)*2+(l>>1)][((threadIdx.x>>2)+(l>>1)*4+((l&1)+(threadIdx.x&3))*8+j*16)&31]));
                    //    else
                    //        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                   }
                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                       : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                       : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                       "r"(A[k*2]), "r"(A[k*2+1]), 
                       "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
               }
           }
       } 
    }
    for (int j = 0; j < 8; j++)
        if (bid*8+(threadIdx.x&3)*2+(j&1)<node_num)
            output[(bid*8+(threadIdx.x&3)*2+(j&1))*32+(j>>1)*8+(threadIdx.x>>2)] = C[j];
}

__global__ void agnn_kernel_8x8_32_3(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[3][8];
    __shared__ float dense_X[3][8][32];
    __shared__ float softmax[2][2][8];
    __shared__ float sparse_A[2][16][8];

    uint32_t D[4], A[4], B[4], E[1] = {0xeeee4444};
    float C[4];
    float scale = beta[0];
    for (int i=0; i<4; i++) {
        C[i] = x[min(bid*8+((threadIdx.x>>2)&7), node_num-1)*32+i*4+(threadIdx.x>>5)*16]/x_norm[min(bid*8+((threadIdx.x>>2)&7), node_num-1)];
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(C[i]));
        C[i] = 0.f;
    }
    int i = block_start;
    SparseAToX_idx[i%3][threadIdx.x>>3] = SparseAToX[i*8+(threadIdx.x>>3)];
    int cur_addr = __cvta_generic_to_shared(&dense_X[i%3][threadIdx.x>>3][(((threadIdx.x&7)+(threadIdx.x>>3))*4)&31]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i%3][threadIdx.x>>3], node_num-1)*32+(threadIdx.x&7)*4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if ((i+1)<block_end) {
        SparseAToX_idx[(i+1)%3][threadIdx.x>>3] = SparseAToX[(i+1)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)%3][threadIdx.x>>3][(((threadIdx.x&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)%3][threadIdx.x>>3], node_num-1)*32+(threadIdx.x&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
    }
    {
        float alpha[4] = {0.f};
        for (int k = 0; k < 4; k++)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(
                dense_X[i%3][((threadIdx.x>>2)&7)][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+k*2+4*((threadIdx.x>>2)&7)+16*(threadIdx.x>>5))&31]/x_norm[min(SparseAToX_idx[i%3][((threadIdx.x>>2)&7)], node_num-1)]));
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
            : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
            : "r"(D[0]), "r"(D[2]), "r"(D[1]), "r"(D[3]), 
            "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        for (int k = 0; k < 2; k++)
            sparse_A[i&1][(threadIdx.x>>3)+8*k][threadIdx.x&7] = alpha[k]+alpha[k+2];
    }
    softmax[i&1][0][threadIdx.x>>3] = -1.0f * scale;
    softmax[i&1][1][threadIdx.x>>3] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (; i<(block_end-2); i++) {
        SparseAToX_idx[(i+2)%3][threadIdx.x>>3] = SparseAToX[(i+2)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+2)%3][threadIdx.x>>3][(((threadIdx.x&7)+(threadIdx.x>>3))*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+2)%3][threadIdx.x>>3], node_num-1)*32+(threadIdx.x&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
        {
            float alpha[4] = {0.f};
            for (int k = 0; k < 4; k++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(
                    dense_X[(i+1)%3][((threadIdx.x>>2)&7)][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+k*2+4*((threadIdx.x>>2)&7)+16*(threadIdx.x>>5))&31]/x_norm[min(SparseAToX_idx[(i+1)%3][((threadIdx.x>>2)&7)], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[0]), "r"(D[2]), "r"(D[1]), "r"(D[3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
            for (int k = 0; k < 2; k++)
                sparse_A[(i+1)&1][(threadIdx.x>>3)+k*8][threadIdx.x&7] = alpha[k]+alpha[k+2];
        }
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum, alpha[2];
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<((threadIdx.x>>2)&7))-1))];
            for (int k = 0; k < 2; k++) {
                mask[k] = (float)(((col_mask>>((threadIdx.x>>2)&7))&(row_mask>>((threadIdx.x&3)*2+k)))&1);
                alpha[k] = min(-1.f+2*mask[k], sparse_A[i&1][((threadIdx.x>>3)&3)+k*8][threadIdx.x&7]+sparse_A[i&1][((threadIdx.x>>3)&3)+4+k*8][threadIdx.x&7])*scale;
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][((threadIdx.x>>2)&7)]);
            softmax[(i+1)&1][0][((threadIdx.x>>2)&7)] = alpha_max;
            alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
            alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
            alpha_sum = alpha[0] + alpha[1];
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1);
            }
            alpha_sum += softmax[i&1][1][((threadIdx.x>>2)&7)]*__expf(softmax[i&1][0][((threadIdx.x>>2)&7)]-alpha_max);
            softmax[(i+1)&1][1][((threadIdx.x>>2)&7)] = alpha_sum;
            float rcp = 1.f / (alpha_sum+1e-16f);
            alpha[0] *= rcp;
            alpha[1] *= rcp;
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(alpha[j]));
        }
        for (int j=0; j<2; j++) {
            float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j]-softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
            C[j] *= update;
            C[j+2] *= update;
        }
        for (int j=0; j<2; j++) {
            // if (SparseAToX_idx[i%3][(threadIdx.x&3)*2+j]<node_num) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+((threadIdx.x&3)*2+j)*4)&31]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+8+((threadIdx.x&3)*2+j)*4)&31]));
            // } else {
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(0.f));
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(0.f));
            // }
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-2)) {
        {
            float alpha[4] = {0.f};
            for (int k = 0; k < 4; k++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(
                    dense_X[(i+1)%3][((threadIdx.x>>2)&7)][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+k*2+4*((threadIdx.x>>2)&7)+16*(threadIdx.x>>5))&31]/x_norm[min(SparseAToX_idx[(i+1)%3][((threadIdx.x>>2)&7)], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[0]), "r"(D[2]), "r"(D[1]), "r"(D[3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
            for (int k = 0; k < 2; k++)
                sparse_A[(i+1)&1][(threadIdx.x>>3)+k*8][threadIdx.x&7] = alpha[k]+alpha[k+2];
        }
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum, alpha[2];
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<((threadIdx.x>>2)&7))-1))];
            for (int k = 0; k < 2; k++) {
                mask[k] = (float)(((col_mask>>((threadIdx.x>>2)&7))&(row_mask>>((threadIdx.x&3)*2+k)))&1);
                alpha[k] = min(-1.f+2*mask[k], sparse_A[i&1][((threadIdx.x>>3)&3)+k*8][threadIdx.x&7]+sparse_A[i&1][((threadIdx.x>>3)&3)+4+k*8][threadIdx.x&7])*scale;
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][((threadIdx.x>>2)&7)]);
            softmax[(i+1)&1][0][((threadIdx.x>>2)&7)] = alpha_max;
            alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
            alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
            alpha_sum = alpha[0] + alpha[1];
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1);
            }
            alpha_sum += softmax[i&1][1][((threadIdx.x>>2)&7)]*__expf(softmax[i&1][0][((threadIdx.x>>2)&7)]-alpha_max);
            softmax[(i+1)&1][1][((threadIdx.x>>2)&7)] = alpha_sum;
            float rcp = 1.f / (alpha_sum+1e-16f);
            alpha[0] *= rcp;
            alpha[1] *= rcp;
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(alpha[j]));
        }
        for (int j=0; j<2; j++) {
            float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j]-softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
            C[j] *= update;
            C[j+2] *= update;
        }
        for (int j=0; j<2; j++) {
            // if (SparseAToX_idx[i%3][(threadIdx.x&3)*2+j]<node_num) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+((threadIdx.x&3)*2+j)*4)&31]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+8+((threadIdx.x&3)*2+j)*4)&31]));
            // } else {
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(0.f));
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(0.f));
            // }
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        i++;
    }
    __syncthreads();
    if (i == (block_end-1)) {
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum, alpha[2];
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<((threadIdx.x>>2)&7))-1))];
            for (int k = 0; k < 2; k++) {
                mask[k] = (float)(((col_mask>>((threadIdx.x>>2)&7))&(row_mask>>((threadIdx.x&3)*2+k)))&1);
                alpha[k] = min(-1.f+2*mask[k], sparse_A[i&1][((threadIdx.x>>3)&3)+k*8][threadIdx.x&7]+sparse_A[i&1][((threadIdx.x>>3)&3)+4+k*8][threadIdx.x&7])*scale;
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][((threadIdx.x>>2)&7)]);
            softmax[(i+1)&1][0][((threadIdx.x>>2)&7)] = alpha_max;
            alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
            alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
            alpha_sum = alpha[0] + alpha[1];
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1);
            }
            alpha_sum += softmax[i&1][1][((threadIdx.x>>2)&7)]*__expf(softmax[i&1][0][((threadIdx.x>>2)&7)]-alpha_max);
            softmax[(i+1)&1][1][((threadIdx.x>>2)&7)] = alpha_sum;
            float rcp = 1.f / (alpha_sum+1e-16f);
            alpha[0] *= rcp;
            alpha[1] *= rcp;
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(alpha[j]));
        }
        for (int j=0; j<2; j++) {
            float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j]-softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
            C[j] *= update;
            C[j+2] *= update;
        }
        for (int j=0; j<2; j++) {
            // if (SparseAToX_idx[i%3][(threadIdx.x&3)*2+j]<node_num) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+((threadIdx.x&3)*2+j)*4)&31]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(dense_X[i%3][(threadIdx.x&3)*2+j][(((threadIdx.x>>2)&7)+(threadIdx.x>>5)*16+8+((threadIdx.x&3)*2+j)*4)&31]));
            // } else {
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2]) : "f"(0.f));
            //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j*2+1]) : "f"(0.f));
            // }
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    }
    for (int j=0; j<4; j++)
        if (bid*8+(threadIdx.x&3)*2+(j&1) < node_num)
            output[(bid*8+(threadIdx.x&3)*2+(j&1))*32+((threadIdx.x&32)>>1)+((threadIdx.x>>2)&7)+(j&2)*4] = C[j];
}

__global__ void agnn_kernel_8x8_32(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense_X[2][8][32];
    __shared__ float softmax[2][2][8];

    uint32_t D[8], A[2], B[4], E[1] = {0xeeee4444};
    float C[8] = {0.f};
    for (int i=0; i<8; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*8+(threadIdx.x>>2), node_num-1)*32+i*4+(threadIdx.x&3)]/x_norm[min(bid*8+(threadIdx.x>>2), node_num-1)]));
    }
    int i = block_start;
    int cur_addr;
    for (int j = 0; j < 2; j++) {
        SparseAToX_idx[i&1][(threadIdx.x>>3)+j*4] = SparseAToX[i*8+(threadIdx.x>>3)+j*4];
        cur_addr = __cvta_generic_to_shared(&dense_X[i&1][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i&1][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (; i<(block_end-1); i++) {
        for (int j = 0; j < 2; j++) {
            SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*4] = SparseAToX[(i+1)*8+(threadIdx.x>>3)+j*4];
            cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)&1][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[4] = {0.f};
        for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 4; l++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                    dense_X[i&1][(threadIdx.x>>2)][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+l*2+k*16+(threadIdx.x>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(threadIdx.x>>2)], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum;
            int row_id = threadIdx.x>>2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 2; k++) {
                int col_id = (threadIdx.x&3)*2+k;
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[k] = min(-1.f+2*mask[k], alpha[k]+alpha[k+2])*beta[0];
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][threadIdx.x>>2]);
            softmax[(i+1)&1][0][threadIdx.x>>2] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 2; k++) {
                alpha[k] = mask[k] * __expf(alpha[k]-alpha_max);
                alpha_sum += alpha[k];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>2] * __expf(softmax[i&1][0][threadIdx.x>>2] - alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>2] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 2; k++) {
                alpha[k] *= rcp;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j] - softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int l = 0; l < 4; l++) {
                    // if (SparseAToX_idx[i&1][(threadIdx.x&3)*2+(l>>1)] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][(threadIdx.x&3)*2+(l>>1)][((threadIdx.x>>2)+(l>>1)*4+((l&1)+(threadIdx.x&3))*8+j*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "r"(A[0]), "r"(A[1]), 
                    "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-1)) {
        // SDDMM
        float alpha[4] = {0.f};
        for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 4; l++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(
                    dense_X[i&1][(threadIdx.x>>2)][((threadIdx.x&1)*8+((threadIdx.x&2)>>1)+l*2+k*16+(threadIdx.x>>2)*4)&31]/x_norm[min(SparseAToX_idx[i&1][(threadIdx.x>>2)], node_num-1)]));
            asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[k*4]), "r"(D[k*4+2]), "r"(D[k*4+1]), "r"(D[k*4+3]), 
                "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]), "r"(E[0]));
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[2], alpha_max, alpha_sum;
            int row_id = threadIdx.x>>2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 2; k++) {
                int col_id = (threadIdx.x&3)*2+k;
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[k] = min(-1.f+2*mask[k], alpha[k]+alpha[k+2])*beta[0];
            }
            alpha_max = max(alpha[0], alpha[1]);
            for (int k = 1; k < 4; k<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*k-1, 4));
            }
            alpha_max = max(alpha_max, softmax[i&1][0][threadIdx.x>>2]);
            softmax[(i+1)&1][0][threadIdx.x>>2] = alpha_max;
            alpha_sum = 0;
            for (int k = 0; k < 2; k++) {
                alpha[k] = mask[k] * __expf(alpha[k]-alpha_max);
                alpha_sum += alpha[k];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*k-1, 4);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>2] * __expf(softmax[i&1][0][threadIdx.x>>2] - alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>2] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            for (int k = 0; k < 2; k++) {
                alpha[k] *= rcp;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(threadIdx.x&3)*2+j] - softmax[(i+1)&1][0][(threadIdx.x&3)*2+j])
                        *(softmax[i&1][1][(threadIdx.x&3)*2+j]+1e-16f)/(softmax[(i+1)&1][1][(threadIdx.x&3)*2+j]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j+k*2] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int l = 0; l < 4; l++) {
                    // if (SparseAToX_idx[i&1][(threadIdx.x&3)*2+(l>>1)] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(dense_X[i&1][(threadIdx.x&3)*2+(l>>1)][((threadIdx.x>>2)+(l>>1)*4+((l&1)+(threadIdx.x&3))*8+j*16)&31]));
                    // else
                    //     asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l]) : "f"(0.f));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[j*4]), "=f"(C[j*4+1]), "=f"(C[j*4+2]), "=f"(C[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
                    "r"(A[0]), "r"(A[1]), 
                    "f"(C[j*4]), "f"(C[j*4+1]), "f"(C[j*4+2]), "f"(C[j*4+3]));
            }
        }
    }
    for (int j = 0; j < 8; j++)
        if (bid*8+(threadIdx.x&3)*2+(j&1)<node_num)
            output[(bid*8+(threadIdx.x&3)*2+(j&1))*32+(j>>1)*8+(threadIdx.x>>2)] = C[j];
}

__global__ void agnn_kernel_4x8_32_3(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[3][8];
    __shared__ float dense_X[3][8][32];
    __shared__ float softmax[2][2][4];
    __shared__ float sparse_A[2][4][8];

    float D[4], C[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j=0; j<4; j++) {
        D[j] = x[min(bid*4+(threadIdx.x>>3), node_num-1)*32+(threadIdx.x&7)*4+j]/x_norm[min(bid*4+(threadIdx.x>>3), node_num-1)];
    }
    int i = block_start;
    for (int j=0; j<2; j++) {
        SparseAToX_idx[i%3][(threadIdx.x>>3)+j*4] = SparseAToX[i*8+(threadIdx.x>>3)+j*4];
        int cur_addr = __cvta_generic_to_shared(&dense_X[i%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if ((i+1)<block_end) {
        for (int j=0; j<2; j++) {
            SparseAToX_idx[(i+1)%3][(threadIdx.x>>3)+j*4] = SparseAToX[(i+1)*8+(threadIdx.x>>3)+j*4];
            int cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        float alpha = 0.f, norm = 1.f/x_norm[min(SparseAToX_idx[i%3][threadIdx.x&7], node_num-1)];
        for (int j=0; j<32; j++) {
            alpha += __shfl_sync(FULL_MASK, D[j&3], (j>>2), 8) * dense_X[i%3][threadIdx.x&7][((threadIdx.x&7)*4+j)&31];
        }
        sparse_A[i&1][threadIdx.x>>3][threadIdx.x&7] = alpha * norm;
    }
    softmax[i&1][0][threadIdx.x>>3] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>3] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (; i<(block_end-2); i++) {
        for (int j=0; j<2; j++) {
            SparseAToX_idx[(i+2)%3][(threadIdx.x>>3)+j*4] = SparseAToX[(i+2)*8+(threadIdx.x>>3)+j*4];
            int cur_addr = __cvta_generic_to_shared(&dense_X[(i+2)%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+2)%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        {
            float alpha = 0.f, norm = 1.f/x_norm[min(SparseAToX_idx[(i+1)%3][threadIdx.x&7], node_num-1)];
            for (int j=0; j<32; j++) {
                alpha += __shfl_sync(FULL_MASK, D[j&3], (j>>2), 8) * dense_X[(i+1)%3][threadIdx.x&7][((threadIdx.x&7)*4+j)&31];
            }
            sparse_A[(i+1)&1][threadIdx.x>>3][threadIdx.x&7] = alpha * norm;
        }      
        {
            int row_start = BitMaskRowOffset[i];
            uint32_t col_mask = BitColMask[i/2]>>(4*(i&1));
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>3))-1))];
            float mask = (col_mask>>(threadIdx.x>>3))&(row_mask>>(threadIdx.x&7))&1;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>3][threadIdx.x&7])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>3]);
            for (int j=1;  j<8; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            }
            softmax[(i+1)&1][0][threadIdx.x>>3] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<8; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>3]*__expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>3] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max) *
                        (softmax[i&1][1][threadIdx.x>>3]+1e-16f)*rcp;
            for (int j=0; j<4; j++) {
                C[j] *= update; 
                for (int k=0; k<8; k++)
                    C[j] += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(j+((threadIdx.x&7)+k)*4)&31];
            }    
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-2)) {
        {
            float alpha = 0.f, norm = 1.f/x_norm[min(SparseAToX_idx[(i+1)%3][threadIdx.x&7], node_num-1)];
            for (int j=0; j<32; j++) {
                alpha += __shfl_sync(FULL_MASK, D[j&3], (j>>2), 8) * dense_X[(i+1)%3][threadIdx.x&7][((threadIdx.x&7)*4+j)&31];
            }
            sparse_A[(i+1)&1][threadIdx.x>>3][threadIdx.x&7] = alpha * norm;
        }
        {
            int row_start = BitMaskRowOffset[i];
            uint32_t col_mask = BitColMask[i/2]>>(4*(i&1));
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>3))-1))];
            float mask = (col_mask>>(threadIdx.x>>3))&(row_mask>>(threadIdx.x&7))&1;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>3][threadIdx.x&7])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>3]);
            for (int j=1;  j<8; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            }
            softmax[(i+1)&1][0][threadIdx.x>>3] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<8; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>3]*__expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>3] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max) *
                        (softmax[i&1][1][threadIdx.x>>3]+1e-16f)*rcp;
            for (int j=0; j<4; j++) {
                C[j] *= update; 
                for (int k=0; k<8; k++)
                    C[j] += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(j+((threadIdx.x&7)+k)*4)&31];
            }    
        }
        i++; 
    }
    __syncthreads();
    if (i == (block_end - 1)) {
        int row_start = BitMaskRowOffset[i];
        uint32_t col_mask = BitColMask[i/2]>>(4*(i&1));
        uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>3))-1))];
        float mask = (col_mask>>(threadIdx.x>>3))&(row_mask>>(threadIdx.x&7))&1;
        float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>3][threadIdx.x&7])*beta[0];
        float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>3]);
        for (int j=1;  j<8; j<<=1) {
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
        }
        softmax[(i+1)&1][0][threadIdx.x>>3] = alpha_max;
        alpha = mask * __expf(alpha-alpha_max);
        float alpha_sum = alpha;
        for (int j=1;  j<8; j<<=1) {
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
        }
        alpha_sum += softmax[i&1][1][threadIdx.x>>3]*__expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max);
        softmax[(i+1)&1][1][threadIdx.x>>3] = alpha_sum;
        float rcp = 1.f/(alpha_sum+1e-16f);
        alpha *= rcp;
        float update = __expf(softmax[i&1][0][threadIdx.x>>3]-alpha_max) *
                    (softmax[i&1][1][threadIdx.x>>3]+1e-16f)*rcp;
        for (int j=0; j<4; j++) {
            C[j] *= update; 
            for (int k=0; k<8; k++)
                C[j] += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(j+((threadIdx.x&7)+k)*4)&31];
        } 
    }
    if (bid*4+(threadIdx.x>>3)<node_num)
        FLOAT4(output[(bid*4+(threadIdx.x>>3))*32+(threadIdx.x&7)*4]) = FLOAT4(C[0]);
}

__global__ void agnn_kernel_2x16_32_3(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint16_t* __restrict__ BitRowMask,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if(block_start == block_end) return;

    __shared__ int SparseAToX_idx[3][16];
    __shared__ float dense_X[3][16][32];
    __shared__ float softmax[2][2][2];
    __shared__ float sparse_A[2][2][16];

    float D[2], C[2] = {0.f, 0.f};
    for (int j=0; j<2; j++) {
        D[j] = x[min(bid*2+(threadIdx.x>>4), node_num-1)*32+(threadIdx.x&15)*2+j]/x_norm[min(bid*2+(threadIdx.x>>4), node_num-1)];
    }
    int i = block_start;
    for (int j=0; j<4; j++) {
        SparseAToX_idx[i%3][(threadIdx.x>>3)+j*4] = SparseAToX[i*16+(threadIdx.x>>3)+j*4];
        int cur_addr = __cvta_generic_to_shared(&dense_X[i%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[i%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if ((i+1)<block_end) {
        for (int j=0; j<4; j++) {
            SparseAToX_idx[(i+1)%3][(threadIdx.x>>3)+j*4] = SparseAToX[(i+1)*16+(threadIdx.x>>3)+j*4];
            int cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+1)%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        float alpha = 0.f, norm = 1.f/x_norm[min(SparseAToX_idx[i%3][threadIdx.x&15], node_num-1)];
        for (int j=0; j<32; j++) {
            alpha += __shfl_sync(FULL_MASK, D[j&1], (j>>1), 16) * dense_X[i%3][threadIdx.x&15][((threadIdx.x&15)*4+j)&31];
        }
        sparse_A[i&1][threadIdx.x>>4][threadIdx.x&15] = alpha;
    }
    softmax[i&1][0][threadIdx.x>>4] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>4] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (; i<(block_end-2); i++) {
        for (int j=0; j<4; j++) {
            SparseAToX_idx[(i+2)%3][(threadIdx.x>>3)+j*4] = SparseAToX[(i+2)*16+(threadIdx.x>>3)+j*4];
            int cur_addr = __cvta_generic_to_shared(&dense_X[(i+2)%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(SparseAToX_idx[(i+2)%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        {
            float alpha = 0, norm = 1.f/x_norm[min(SparseAToX_idx[(i+1)%3][threadIdx.x&15], node_num-1)];
            for (int j=0; j<32; j++) {
                alpha += __shfl_sync(FULL_MASK, D[j&1], (j>>1), 16) * dense_X[(i+1)%3][threadIdx.x&15][((threadIdx.x&15)*4+j)&31];
            }
            sparse_A[(i+1)&1][threadIdx.x>>4][threadIdx.x&15] = alpha;
        }
        {
            int row_start = BitMaskRowOffset[i];
            uint32_t col_mask = BitColMask[i/4]>>(2*(i&3));
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>4))-1))];
            float mask = (col_mask>>(threadIdx.x>>4))&(row_mask>>(threadIdx.x&15))&1;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>4][threadIdx.x&15])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>4]);
            for (int j=1;  j<16; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
            }
            softmax[(i+1)&1][0][threadIdx.x>>4] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<16; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>4]*__expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>4] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max) *
                        (softmax[i&1][1][threadIdx.x>>4]+1e-16f)*rcp;
            for (int j=0; j<2; j++) {
                C[j] *= update; 
                for (int k=0; k<16; k++)
                    C[j] += __shfl_sync(FULL_MASK, alpha, k, 16) * dense_X[i%3][k][(j+(threadIdx.x&15)*2+k*4)&31];
            }    
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if (i == (block_end-2)) {
        {
            float alpha = 0, norm = 1.f/x_norm[min(SparseAToX_idx[(i+1)%3][threadIdx.x&15], node_num-1)];
            for (int j=0; j<32; j++) {
                alpha += __shfl_sync(FULL_MASK, D[j&1], (j>>1), 16) * dense_X[(i+1)%3][threadIdx.x&15][((threadIdx.x&15)*4+j)&31];
            }
            sparse_A[(i+1)&1][threadIdx.x>>4][threadIdx.x&15] = alpha;
        }
        {
            int row_start = BitMaskRowOffset[i];
            uint32_t col_mask = BitColMask[i/4]>>(2*(i&3));
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>4))-1))];
            float mask = (col_mask>>(threadIdx.x>>4))&(row_mask>>(threadIdx.x&15))&1;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>4][threadIdx.x&15])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>4]);
            for (int j=1;  j<16; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
            }
            softmax[(i+1)&1][0][threadIdx.x>>4] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<16; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
            }
            alpha_sum += softmax[i&1][1][threadIdx.x>>4]*__expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max);
            softmax[(i+1)&1][1][threadIdx.x>>4] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max) *
                        (softmax[i&1][1][threadIdx.x>>4]+1e-16f)*rcp;
            for (int j=0; j<2; j++) {
                C[j] *= update; 
                for (int k=0; k<16; k++)
                    C[j] += __shfl_sync(FULL_MASK, alpha, k, 16) * dense_X[i%3][k][(j+(threadIdx.x&15)*2+k*4)&31];
            }    
        }
        i++;
    }
    __syncthreads();
    if (i == (block_end - 1)) {
        int row_start = BitMaskRowOffset[i];
        uint32_t col_mask = BitColMask[i/4]>>(2*(i&3));
        uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<(threadIdx.x>>4))-1))];
        float mask = (col_mask>>(threadIdx.x>>4))&(row_mask>>(threadIdx.x&15))&1;
        float alpha = min(-1.f+2*mask, sparse_A[i&1][threadIdx.x>>4][threadIdx.x&15])*beta[0];
        float alpha_max = max(alpha, softmax[i&1][0][threadIdx.x>>4]);
        for (int j=1;  j<16; j<<=1) {
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
        }
        softmax[(i+1)&1][0][threadIdx.x>>4] = alpha_max;
        alpha = mask * __expf(alpha-alpha_max);
        float alpha_sum = alpha;
        for (int j=1;  j<16; j<<=1) {
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
        }
        alpha_sum += softmax[i&1][1][threadIdx.x>>4]*__expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max);
        softmax[(i+1)&1][1][threadIdx.x>>4] = alpha_sum;
        float rcp = 1.f/(alpha_sum+1e-16f);
        alpha *= rcp;
        float update = __expf(softmax[i&1][0][threadIdx.x>>4]-alpha_max) *
                    (softmax[i&1][1][threadIdx.x>>4]+1e-16f)*rcp;
        for (int j=0; j<2; j++) {
            C[j] *= update; 
            for (int k=0; k<16; k++)
                C[j] += __shfl_sync(FULL_MASK, alpha, k, 16) * dense_X[i%3][k][(j+(threadIdx.x&15)*2+k*4)&31];
        }    
    }
    if (bid*2+(threadIdx.x>>4)<node_num)
        FLOAT2(output[(bid*2+(threadIdx.x>>4))*32+(threadIdx.x&15)*2]) = FLOAT2(C[0]);
}

at::Tensor AGNN_short(
    at::Tensor feature,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta,
    int out_feats,
    int block_high,
    int block_width
) {
    int num_nodes = feature.size(0);
    auto x_norm = feature.norm(2, -1).clamp_min(1e-12);
    auto output = at::empty({num_nodes, out_feats}, feature.options());

    int blocks = (num_nodes + block_high - 1) / block_high;
    int mode = block_high*100+block_width;
    switch (mode) {
        case 1608:
            agnn_kernel_16x8_32<<<blocks, 64>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                (uint16_t*)BitColMask.data_ptr<uint8_t>(),
                BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 1616:
            agnn_kernel_16x16_32<<<blocks, 64>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                (uint16_t*)BitColMask.data_ptr<uint8_t>(),
                (uint16_t*)BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 816:
            agnn_kernel_8x16_32<<<blocks, 32>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                BitColMask.data_ptr<uint8_t>(),
                (uint16_t*)BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 808:
            agnn_kernel_8x8_32_3<<<blocks, 64>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                BitColMask.data_ptr<uint8_t>(),
                BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 408:
            agnn_kernel_4x8_32_3<<<blocks, 32>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                BitColMask.data_ptr<uint8_t>(),
                BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 216:
            agnn_kernel_2x16_32_3<<<blocks, 32>>>(
                RowWindowOffsets.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BitMaskRowOffset.data_ptr<int>(),
                BitColMask.data_ptr<uint8_t>(),
                (uint16_t*)BitRowMask.data_ptr<uint8_t>(),
                beta.data_ptr<float>(),
                feature.data_ptr<float>(),
                x_norm.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        default:
            printf("Unsupported mode: %d\n", mode);
            exit(1);
    }
    return output;
}

__global__ void agnn_csr(
    const int* __restrict__ row_offset,
    const int* __restrict__ index,
    const float* __restrict__ beta,
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int row_start = row_offset[bid];
    int row_end = row_offset[bid+1];

    __shared__ int node_index[3][8];
    __shared__ float dense_X[3][8][32];
    __shared__ float softmax[2][2];
    __shared__ float sparse_A[2][8];

    float D, C = 0.f;
    D = x[bid*32+threadIdx.x]/x_norm[bid];
    for (int j=0; j<2; j++) {
        node_index[0][(threadIdx.x>>3)+j*4] = index[min(row_start+(threadIdx.x>>3)+j*4, row_end-1)];
        int cur_addr = __cvta_generic_to_shared(&dense_X[0][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(node_index[0][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if (8<row_end) {
        for (int j=0; j<2; j++) {
            node_index[1][(threadIdx.x>>3)+j*4] = index[min(row_start+8+(threadIdx.x>>3)+j*4, row_end-1)];
            int cur_addr = __cvta_generic_to_shared(&dense_X[1][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(node_index[1][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        float alpha=0.f, norm = 1.f/x_norm[node_index[0][threadIdx.x>>2]];
        for (int j=0; j<8; j++) {
            alpha += __shfl_sync(FULL_MASK, D, (threadIdx.x&3)*8+j)*dense_X[0][(threadIdx.x>>2)][((threadIdx.x>>2)*4+(threadIdx.x&3)*8+j)&31];
        }
        for (int j=1; j<4; j<<=1) {
            alpha += __shfl_xor_sync(FULL_MASK, alpha, 2*j-1, 4);
        }
        sparse_A[0][threadIdx.x>>2] = alpha*norm;
    }
    softmax[0][0] = -1.0f * beta[0];
    softmax[0][1] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (int i=0; i<((row_end-row_start+7)/8-2); i++) {
        for (int j=0; j<2; j++) {
            node_index[(i+2)%3][(threadIdx.x>>3)+j*4] = index[min(row_start+8*(i+2)+(threadIdx.x>>3)+j*4, row_end-1)];
            int cur_addr = __cvta_generic_to_shared(&dense_X[(i+2)%3][(threadIdx.x>>3)+j*4][(((threadIdx.x&7)+(threadIdx.x>>3)+j*4)*4)&31]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(node_index[(i+2)%3][(threadIdx.x>>3)+j*4], node_num-1)*32+(threadIdx.x&7)*4]));
            asm volatile("cp.async.commit_group;\n"::);
        }
        float alpha=0.f, norm = 1.f/x_norm[node_index[(i+1)%3][threadIdx.x>>2]];
        for (int j=0; j<8; j++) {
            alpha += __shfl_sync(FULL_MASK, D, (threadIdx.x&3)*8+j)*dense_X[(i+1)%3][(threadIdx.x>>2)][((threadIdx.x>>2)*4+(threadIdx.x&3)*8+j)&31];
        }
        for (int j=1; j<4; j<<=1) {
            alpha += __shfl_xor_sync(FULL_MASK, alpha, 2*j-1, 4);
        }
        sparse_A[(i+1)%3][threadIdx.x>>2] = alpha*norm;
        {
            int col_id = threadIdx.x&7;
            float mask = row_start+i*8+col_id<row_end;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][col_id])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0]);
            for (int j=1;  j<8; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
            }
            softmax[(i+1)&1][0] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<8; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
            }
            alpha_sum += softmax[i&1][1]*__expf(softmax[i&1][0]-alpha_max);
            softmax[(i+1)&1][1] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0]-alpha_max) *(softmax[i&1][1]+1e-16f)*rcp;
            C *= update;
            for (int k=0; k<8; k++)
                C += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(threadIdx.x+k*4)&31];
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    if ((row_end-row_start+7)/8>=2) {
        int i = ((row_end-row_start+7)/8-2);
        float alpha=0.f, norm = 1.f/x_norm[node_index[(i+1)%3][threadIdx.x>>2]];
        for (int j=0; j<8; j++) {
            alpha += __shfl_sync(FULL_MASK, D, (threadIdx.x&3)*8+j)*dense_X[(i+1)%3][(threadIdx.x>>2)][((threadIdx.x>>2)*4+(threadIdx.x&3)*8+j)&31];
        }
        for (int j=1; j<4; j<<=1) {
            alpha += __shfl_xor_sync(FULL_MASK, alpha, 2*j-1, 4);
        }
        sparse_A[(i+1)%3][threadIdx.x>>2] = alpha*norm;
        {
            int col_id = threadIdx.x&7;
            float mask = row_start+i*8+col_id<row_end;
            float alpha = min(-1.f+2*mask, sparse_A[i&1][col_id])*beta[0];
            float alpha_max = max(alpha, softmax[i&1][0]);
            for (int j=1;  j<8; j<<=1) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
            }
            softmax[(i+1)&1][0] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1;  j<8; j<<=1) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
            }
            alpha_sum += softmax[i&1][1]*__expf(softmax[i&1][0]-alpha_max);
            softmax[(i+1)&1][1] = alpha_sum;
            float rcp = 1.f/(alpha_sum+1e-16f);
            alpha *= rcp;
            float update = __expf(softmax[i&1][0]-alpha_max) *(softmax[i&1][1]+1e-16f)*rcp;
            C *= update;
            for (int k=0; k<8; k++)
                C += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(threadIdx.x+k*4)&31];
        }
        __syncthreads();
    }
    if ((row_end-row_start+7)/8>=1) {
        int i = ((row_end-row_start+7)/8-1);
        int col_id = threadIdx.x&7;
        float mask = row_start+i*8+col_id<row_end;
        float alpha = min(-1.f+2*mask, sparse_A[i&1][col_id])*beta[0];
        float alpha_max = max(alpha, softmax[i&1][0]);
        for (int j=1;  j<8; j<<=1) {
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 16));
        }
        softmax[(i+1)&1][0] = alpha_max;
        alpha = mask * __expf(alpha-alpha_max);
        float alpha_sum = alpha;
        for (int j=1;  j<8; j<<=1) {
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 16);
        }
        alpha_sum += softmax[i&1][1]*__expf(softmax[i&1][0]-alpha_max);
        softmax[(i+1)&1][1] = alpha_sum;
        float rcp = 1.f/(alpha_sum+1e-16f);
        alpha *= rcp;
        float update = __expf(softmax[i&1][0]-alpha_max) *(softmax[i&1][1]+1e-16f)*rcp;
        C *= update;
        for (int k=0; k<8; k++)
            C += __shfl_sync(FULL_MASK, alpha, k, 8) * dense_X[i%3][k][(threadIdx.x+k*4)&31];
    }
    output[bid*32+threadIdx.x] = C;
}

at::Tensor AGNN_CSR(
    at::Tensor feature,
    at::Tensor beta,
    at::Tensor row_offset,
    at::Tensor index
) {
    int num_nodes = feature.size(0);
    auto x_norm = feature.norm(2, -1).clamp_min(1e-12);
    auto output = at::empty({num_nodes, 32}, feature.options());

    agnn_csr<<<num_nodes, 32>>>(
        row_offset.data_ptr<int>(),
        index.data_ptr<int>(),
        beta.data_ptr<float>(),
        feature.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        output.data_ptr<float>(),
        num_nodes);
    return output;
}
