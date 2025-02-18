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

    int lane_front = lane_id>>2;
    int lane_end = lane_id&3;

    uint32_t D[16], A[4], B[2];
    float C[8] = {0.f};
    for (int i=0; i<16; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+8*(i&1)+lane_front, node_num-1)*32+(i>>2)*8+lane_end+(i&2)*2]/x_norm[min(bid*16+8*(i&1)+lane_front, node_num-1)]));
    }
    int i = block_start;
    SparseAToX_idx[i&1][threadIdx.x>>3] = SparseAToX[i*8+(threadIdx.x>>3)];
    int cur_addr = __cvta_generic_to_shared(&dense_X[i&1][(threadIdx.x>>3)][(lane_id&7)*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[i&1][threadIdx.x>>3]*32+(lane_id&7)*4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (i=block_start; i<(block_end-1); i+=1) {
        // load (global->shared_mem)
        SparseAToX_idx[(i+1)&1][threadIdx.x>>3] = SparseAToX[(i+1)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][(threadIdx.x>>3)][(lane_id&7)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[(i+1)&1][threadIdx.x>>3]*32+(lane_id&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[4] = {0.f};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 2; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_X[i&1][lane_front][j*8+k*4+lane_end]/x_norm[SparseAToX_idx[i&1][lane_front]]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
        }
        // softmax
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
                    alpha[2*j+k] = min(-1.f+2*mask[2*j+k], alpha[2*j+k])*beta[0];
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
        // Matmul
        // {
        //     for (int j = 0; j < 2; j++) {
        //         if (SparseAToX_idx[i&1][lane_end*2+j] < node_num) 
        //             asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i&1][lane_end*2+j][lane_front+warp_id*8]));
        //         else
        //             asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
        //         float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
        //                 *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
        //         for (int k = 0; k < 2; k++) 
        //             C[j*2+k] *= update;
        //     }
        //     asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        //         : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        //         : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
        //         "r"(B[0]), "r"(B[1]), 
        //         "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        // }
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
                        *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j*2+(k&2)*2+(k&1)] *= update;
            }
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < 2; j++) {
                    if (SparseAToX_idx[i&1][lane_end*2+j] < node_num) 
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i&1][lane_end*2+j][lane_front+k*8+warp_id*16]));
                    else
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));            
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                    "r"(B[0]), "r"(B[1]), 
                    "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end-1;
    float alpha[4] = {0.f};
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_X[i&1][lane_front][j*8+k*4+lane_end]/x_norm[SparseAToX_idx[i&1][lane_front]]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
            : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
    }
    // softmax
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
                alpha[2*j+k] = min(-1.f+2*mask[2*j+k], alpha[2*j+k])*beta[0];
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
    // Matmul
    // {
    //     for (int j = 0; j < 2; j++) {
    //         if (SparseAToX_idx[i&1][lane_end*2+j] < node_num) 
    //             asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i&1][lane_end*2+j][lane_front+warp_id*8]));
    //         else
    //             asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
    //         float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
    //                 *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
    //         for (int k = 0; k < 2; k++) 
    //             C[j*2+k] *= update;
    //     }
    //     asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    //         : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
    //         : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
    //         "r"(B[0]), "r"(B[1]), 
    //         "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    // }
    {
        for (int j = 0; j < 2; j++) {
            float update = __expf(softmax[i&1][0][lane_front+j*8] - softmax[(i+1)&1][0][lane_front+j*8])
                    *(softmax[i&1][1][lane_front+j*8]+1e-16f)/(softmax[(i+1)&1][1][lane_front+j*8]+1e-16f);
            for (int k = 0; k < 4; k++) 
                C[j*2+(k&2)*2+(k&1)] *= update;
        }
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                if (SparseAToX_idx[i&1][lane_end*2+j] < node_num) 
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense_X[i&1][lane_end*2+j][lane_front+k*8+warp_id*16]));
                else
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));            
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
        }
    }
    for (int j = 0; j < 4; j++) 
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

    uint32_t D[16], A[8], B[4];
    float C[8] = {0.f};
    for (int i=0; i<16; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+8*(i&1)+(lane_id>>2), node_num-1)*32+(i>>2)*8+(lane_id&3)+(i&2)*2]/x_norm[min(bid*16+8*(i&1)+(lane_id>>2), node_num-1)]));
    }
    int i = block_start;
    SparseAToX_idx[i&1][threadIdx.x>>2] = SparseAToX[i*16+(threadIdx.x>>2)];
    int cur_addr;
    for (int j = 0; j < 2; j++) {
        cur_addr = __cvta_generic_to_shared(&dense_X[i&1][threadIdx.x>>2][(lane_id&3)*8+j*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[i&1][threadIdx.x>>2]*32+(lane_id&3)*8+j*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    softmax[i&1][0][threadIdx.x>>2] = -1.0f * beta[0];
    softmax[i&1][1][threadIdx.x>>2] = 0.f;
    __syncthreads();
    for (i=block_start; i<(block_end-1); i++) {
        SparseAToX_idx[(i+1)&1][threadIdx.x>>2] = SparseAToX[(i+1)*16+(threadIdx.x>>2)];
        for (int j = 0; j < 2; j++) {
            cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][threadIdx.x>>2][(lane_id&3)*8+j*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[(i+1)&1][threadIdx.x>>2]*32+(lane_id&3)*8+j*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[8] = {0.f};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 2; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k*2]) : "f"(dense_X[i&1][(lane_id>>2)+k*8][j*8+(lane_id&3)]/x_norm[SparseAToX_idx[i&1][(lane_id>>2)+k*8]]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k*2+1]) : "f"(dense_X[i&1][(lane_id>>2)+k*8][j*8+4+(lane_id&3)]/x_norm[SparseAToX_idx[i&1][(lane_id>>2)+k*8]]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(alpha[k*4]), "=f"(alpha[k*4+1]), "=f"(alpha[k*4+2]), "=f"(alpha[k*4+3])
                    : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
                    "r"(B[k*2]), "r"(B[k*2+1]), 
                    "f"(alpha[k*4]), "f"(alpha[k*4+1]), "f"(alpha[k*4+2]), "f"(alpha[k*4+3]));
            }
        }
        // Softmax
        {
            int row_start = BitMaskRowOffset[i];
            int row_end = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i];
            float mask[8], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 2; j++) {
                int row_id = (lane_id>>2)+j*8;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                for (int k = 0; k < 4; k++) {
                    int col_id = (lane_id&3)*2+(k&2)*4+(k&1);
                    mask[j*2+(k&2)*2+(k&1)] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                    alpha[j*2+(k&2)*2+(k&1)] = min(-1.f+2*mask[j*2+(k&2)*2+(k&1)], alpha[j*2+(k&2)*2+(k&1)])*beta[0];
                }
                alpha_max[j] = max(max(alpha[j*2], alpha[j*2+1]), max(alpha[j*2+4], alpha[j*2+5]));
                for (int k = 1; k < 4; k<<=1) {
                    alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
                }
                alpha_max[j] = max(alpha_max[j], softmax[i&1][0][(lane_id>>2)+j*8]);
                softmax[(i+1)&1][0][(lane_id>>2)+j*8] = alpha_max[j];
                alpha_sum[j] = 0;
                for (int k = 0; k < 4; k++) {
                    alpha[j*2+(k&2)*2+(k&1)] = mask[j*2+(k&2)*2+(k&1)] * __expf(alpha[j*2+(k&2)*2+(k&1)]-alpha_max[j]);
                    alpha_sum[j] += alpha[j*2+(k&2)*2+(k&1)];
                }
                for (int k = 1; k < 4; k<<=1) {
                    alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
                }
                alpha_sum[j] += softmax[i&1][1][(lane_id>>2)+j*8] * __expf(softmax[i&1][0][(lane_id>>2)+j*8] - alpha_max[j]);
                softmax[(i+1)&1][1][(lane_id>>2)+j*8] = alpha_sum[j];
                float rcp = 1.f/(alpha_sum[j]+1e-16f);
                for (int k = 0; k < 4; k++) {
                    alpha[j*2+(k&2)*2+(k&1)] *= rcp;
                }
            }
            for (int j = 0; j < 8; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
        }
        // Matmul
        {
            for (int j = 0; j < 2; j++) {
                float update = __expf(softmax[i&1][0][(lane_id>>2)+j*8] - softmax[(i+1)&1][0][(lane_id>>2)+j*8])
                        *(softmax[i&1][1][(lane_id>>2)+j*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+j*8]+1e-16f);
                for (int k = 0; k < 4; k++) 
                    C[j*2+(k&2)*2+(k&1)] *= update;
            }
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {
                        if (SparseAToX_idx[i&1][j*8+(lane_id&3)*2+l] < node_num)
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j*2+l]) : "f"(dense_X[i&1][j*8+(lane_id&3)*2+l][(lane_id>>2)+k*8+warp_id*16]));
                        else
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                        : "r"(A[j*4]), "r"(A[j*4+1]), "r"(A[j*4+2]), "r"(A[j*4+3]), 
                        "r"(B[j*2]), "r"(B[j*2+1]), 
                        "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end-1;
    // SDDMM
    float alpha[8] = {0.f};
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k*2]) : "f"(dense_X[i&1][(lane_id>>2)+k*8][j*8+(lane_id&3)]/x_norm[SparseAToX_idx[i&1][(lane_id>>2)+k*8]]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k*2+1]) : "f"(dense_X[i&1][(lane_id>>2)+k*8][j*8+4+(lane_id&3)]/x_norm[SparseAToX_idx[i&1][(lane_id>>2)+k*8]]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(alpha[k*4]), "=f"(alpha[k*4+1]), "=f"(alpha[k*4+2]), "=f"(alpha[k*4+3])
                : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
                "r"(B[k*2]), "r"(B[k*2+1]), 
                "f"(alpha[k*4]), "f"(alpha[k*4+1]), "f"(alpha[k*4+2]), "f"(alpha[k*4+3]));
        }
    }
    // Softmax
    {
        int row_start = BitMaskRowOffset[i];
        int row_end = BitMaskRowOffset[i+1];
        uint32_t col_mask = BitColMask[i];
        float mask[8], alpha_max[2], alpha_sum[2];
        for (int j = 0; j < 2; j++) {
            int row_id = (lane_id>>2)+j*8;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 4; k++) {
                int col_id = (lane_id&3)*2+(k&2)*4+(k&1);
                mask[j*2+(k&2)*2+(k&1)] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                alpha[j*2+(k&2)*2+(k&1)] = min(-1.f+2*mask[j*2+(k&2)*2+(k&1)], alpha[j*2+(k&2)*2+(k&1)])*beta[0];
            }
            alpha_max[j] = max(max(alpha[j*2], alpha[j*2+1]), max(alpha[j*2+4], alpha[j*2+5]));
            for (int k = 1; k < 4; k<<=1) {
                alpha_max[j] = max(alpha_max[j], __shfl_xor_sync(FULL_MASK, alpha_max[j], 2*k-1, 4));
            }
            alpha_max[j] = max(alpha_max[j], softmax[i&1][0][(lane_id>>2)+j*8]);
            softmax[(i+1)&1][0][(lane_id>>2)+j*8] = alpha_max[j];
            alpha_sum[j] = 0;
            for (int k = 0; k < 4; k++) {
                alpha[j*2+(k&2)*2+(k&1)] = mask[j*2+(k&2)*2+(k&1)] * __expf(alpha[j*2+(k&2)*2+(k&1)]-alpha_max[j]);
                alpha_sum[j] += alpha[j*2+(k&2)*2+(k&1)];
            }
            for (int k = 1; k < 4; k<<=1) {
                alpha_sum[j] += __shfl_xor_sync(FULL_MASK, alpha_sum[j], 2*k-1, 4);
            }
            alpha_sum[j] += softmax[i&1][1][(lane_id>>2)+j*8] * __expf(softmax[i&1][0][(lane_id>>2)+j*8] - alpha_max[j]);
            softmax[(i+1)&1][1][(lane_id>>2)+j*8] = alpha_sum[j];
            float rcp = 1.f/(alpha_sum[j]+1e-16f);
            for (int k = 0; k < 4; k++) {
                alpha[j*2+(k&2)*2+(k&1)] *= rcp;
            }
        }
        for (int j = 0; j < 8; j++)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(alpha[j]));
    }
    // Matmul
    {
        for (int j = 0; j < 2; j++) {
            float update = __expf(softmax[i&1][0][(lane_id>>2)+j*8] - softmax[(i+1)&1][0][(lane_id>>2)+j*8])
                    *(softmax[i&1][1][(lane_id>>2)+j*8]+1e-16f)/(softmax[(i+1)&1][1][(lane_id>>2)+j*8]+1e-16f);
            for (int k = 0; k < 4; k++) 
                C[j*2+(k&2)*2+(k&1)] *= update;
        }
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    if (SparseAToX_idx[i&1][j*8+(lane_id&3)*2+l] < node_num)
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j*2+l]) : "f"(dense_X[i&1][j*8+(lane_id&3)*2+l][(lane_id>>2)+k*8+warp_id*16]));
                    else
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(C[k*4]), "=f"(C[k*4+1]), "=f"(C[k*4+2]), "=f"(C[k*4+3])
                    : "r"(A[j*4]), "r"(A[j*4+1]), "r"(A[j*4+2]), "r"(A[j*4+3]), 
                    "r"(B[j*2]), "r"(B[j*2+1]), 
                    "f"(C[k*4]), "f"(C[k*4+1]), "f"(C[k*4+2]), "f"(C[k*4+3]));
            }
        }
    }
    for (int j = 0; j < 4; j++)
        *(float2*)(&output[(bid*16+(lane_id>>2)+(j&1)*8)*32+warp_id*16+(j&2)*4+(lane_id&3)*2]) = *(float2*)(&C[j*2]);
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

    int threads = 64;
    int blocks = (num_nodes + block_high - 1) / block_high;
    int mode = block_high*100+block_width;
    switch (mode) {
        case 1608:
            agnn_kernel_16x8_32<<<blocks, threads>>>(
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
            agnn_kernel_16x16_32<<<blocks, threads>>>(
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
        default:
            printf("Unsupported mode: %d\n", mode);
            exit(1);
    }
    return output;
}