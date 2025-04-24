#include <cuda_runtime.h>
#include <cstdint>
#include "gat.cuh"

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float leaky_relu(float x) {
    return x - 0.99f * min(0.f, x);
}

__global__ void gat_kernel_8x8_64(
    const int* __restrict__ RowWindowId,
    const int* __restrict__ RowWindowRowOffset,
    const int* __restrict__ RowWindowBlockOffset,
    const int* __restrict__ RowWindowSparseAToXOffset,
    const int* __restrict__ RowWindowColMaskOffset,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BlockRowMaskOffset,
    const uint8_t* __restrict__ ColMask,
    const uint8_t* __restrict__ RowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = RowWindowId[blockIdx.x];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; //1 bit 
    int block_start = RowWindowBlockOffset[bid];
    int block_num = RowWindowBlockOffset[bid+1] - RowWindowBlockOffset[bid];
    int row_off = RowWindowRowOffset[bid];
    int sparseAToX_off = RowWindowSparseAToXOffset[bid];
    int colmask_off = RowWindowColMaskOffset[bid];

    if(block_num == 0) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float sparse_A[2][8][8];
    __shared__ float dense[2][8][64];
    __shared__ float softmax[3][2][8];

    softmax[0][0][lane_id&7] = std::numeric_limits<float>::lowest();
    softmax[0][1][lane_id&7] = 0.f;
    // load
    if (lane_id < 8)
        SparseAToX_idx[0][lane_id] = SparseAToX[sparseAToX_off+lane_id];
    for (int j=0; j<2; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense[0][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[0][j*4+(threadIdx.x>>4)], node_num-1)*64+4*(lane_id&15)]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    // softmax
    {
        int row_start_byte = BlockRowMaskOffset[block_start];
        uint32_t col_mask = ColMask[block_start];
        int row_id = warp_id*4+(lane_id>>3);
        int col_id = lane_id&7;
        uint32_t row_mask = RowMask[row_start_byte+__popc(col_mask&((1<<row_id)-1))];
        float mask = (col_mask>>row_id)&(row_mask>>col_id)&1;
        float alpha = (mask)?leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[0][col_id], node_num-1)]):
                                std::numeric_limits<float>::lowest();
        float alpha_max = max(alpha, softmax[0][0][row_id]);
        for (int j=1; j<8; j<<=1)
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
        softmax[1][0][row_id] = alpha_max;
        alpha = mask * __expf(alpha-alpha_max);
        float alpha_sum = alpha;
        for (int j=1; j<8; j<<=1)
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
        alpha_sum += softmax[0][1][row_id]*__expf(softmax[0][0][row_id]-alpha_max);
        softmax[1][1][row_id] = alpha_sum;
        sparse_A[0][row_id][col_id] = alpha / (alpha_sum+1e-16f);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    for (int i=0; i<block_num-1; i++) {
        // load
        if (lane_id < 8)
            SparseAToX_idx[(i+1)&1][lane_id] = SparseAToX[sparseAToX_off+(1+i)*8+lane_id];
        for (int j=0; j<2; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>4)], node_num-1)*64+4*(lane_id&15)]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // softmax
        {
            int row_start_byte = BlockRowMaskOffset[block_start+i+1];
            uint32_t col_mask = ColMask[colmask_off+i+1];
            int row_id = warp_id*4+(lane_id>>3);
            int col_id = lane_id&7;
            uint32_t row_mask = RowMask[row_start_byte+__popc(col_mask&((1<<row_id)-1))];
            float mask = (col_mask>>row_id)&(row_mask>>col_id)&1;
            float alpha = (mask)?leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[(i+1)&1][col_id], node_num-1)]):
                                    std::numeric_limits<float>::lowest();
            float alpha_max = max(alpha, softmax[(i+1)%3][0][row_id]);
            for (int j=1; j<8; j<<=1)
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            softmax[(i+2)%3][0][row_id] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1; j<8; j<<=1)
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            alpha_sum += softmax[(i+1)%3][1][row_id]*__expf(softmax[(i+1)%3][0][row_id]-alpha_max);
            softmax[(i+2)%3][1][row_id] = alpha_sum;
            sparse_A[(i+1)&1][row_id][col_id] = alpha / (alpha_sum+1e-16f);
        }
        // matmul
        {
            uint32_t B[8], A[2];
            for (int k=0; k<8; k++) {
                if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][(warp_id*32+(k&3)*8+((k&4)+(lane_id&3))*4+(lane_id>>2))&63]));
                else
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f)); 
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+j*4])); 
            for (int k=0; k<2; k++) {
                float update = __expf(softmax[i%3][0][(lane_id&3)*2+k]-softmax[(i+1)%3][0][(lane_id&3)*2+k])
                            *(softmax[i%3][1][(lane_id&3)*2+k]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+k]+1e-16f);
                for (int j=0; j<4; j++)
                    D[k+j*2] *= update;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3]) 
                    : "r"(B[j*4]), "r"(B[j*4+1]), "r"(B[j*4+2]), "r"(B[j*4+3]), 
                    "r"(A[0]), "r"(A[1]),
                    "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }   
    int i = block_num - 1;
    // matmul
    {
        uint32_t B[8], A[2];
        for (int k=0; k<8; k++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][(warp_id*32+(k&3)*8+((k&4)+(lane_id&3))*4+(lane_id>>2))&63]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f)); 
        }
        for (int j = 0; j < 2; j++)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+j*4])); 
        for (int k=0; k<2; k++) {
            float update = __expf(softmax[i%3][0][(lane_id&3)*2+k]-softmax[(i+1)%3][0][(lane_id&3)*2+k])
                        *(softmax[i%3][1][(lane_id&3)*2+k]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+k]+1e-16f);
            for (int j=0; j<4; j++)
                D[k+j*2] *= update;
        }
        for (int j = 0; j < 2; j++)
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3]) 
                : "r"(B[j*4]), "r"(B[j*4+1]), "r"(B[j*4+2]), "r"(B[j*4+3]), 
                "r"(A[0]), "r"(A[1]),
                "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
    }
    for (int k=0; k<8; k++)
        if ((row_off+(lane_id&3)*2+(k&1))<node_num)
            out[(row_off+(lane_id&3)*2+(k&1))*64+warp_id*32+(k&6)*4+(lane_id>>2)] = D[k];
}

__global__ void gat_kernel_8x16_64(
    const int* __restrict__ RowWindowId,
    const int* __restrict__ RowWindowRowOffset,
    const int* __restrict__ RowWindowBlockOffset,
    const int* __restrict__ RowWindowSparseAToXOffset,
    const int* __restrict__ RowWindowColMaskOffset,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BlockRowMaskOffset,
    const uint8_t* __restrict__ ColMask,
    const uint8_t* __restrict__ RowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = RowWindowId[blockIdx.x];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; // 1 bit
    int block_start = RowWindowRowOffset[bid];
    int block_num = RowWindowBlockOffset[bid+1] - RowWindowBlockOffset[bid];
    int row_off = RowWindowRowOffset[bid];
    int sparseAToX_off = RowWindowSparseAToXOffset[bid];
    int colmask_off = RowWindowColMaskOffset[bid];

    if(block_num == 0) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float sparse_A[2][8][16];
    __shared__ float dense[2][16][64];
    __shared__ float softmax[3][2][8];

    softmax[0][0][warp_id*4+(lane_id>>3)] = std::numeric_limits<float>::lowest();
    softmax[0][1][warp_id*4+(lane_id>>3)] = 0.f;
    // load
    if (lane_id < 16)
        SparseAToX_idx[0][lane_id] = SparseAToX[sparseAToX_off+lane_id];
    for (int j = 0; j < 4; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense[0][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[0][j*4+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    // softmax
    {
        int row_start_byte = BlockRowMaskOffset[block_start];
        uint32_t col_mask = ColMask[colmask_off];
        int row_id = warp_id*4+(lane_id>>3);
        uint8_t row_mask[2] = {RowMask[row_start_byte+__popc(col_mask&((1<<row_id)-1))*2],
                            RowMask[row_start_byte+__popc(col_mask&((1<<row_id)-1))*2+1]};
        float mask[2], alpha[2];
        int col_id[2];
        for (int j=0; j<2; j++) {
            col_id[j] = (lane_id&7)*2+j;
            mask[j] = (col_mask>>row_id)&(*((uint16_t*)row_mask)>>col_id[j])&1;
            alpha[j] = (mask[j])?leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[0][col_id[j]], node_num-1)]):
                                    std::numeric_limits<float>::lowest();
        }
        float alpha_max = max(max(alpha[0],alpha[1]), softmax[0][0][row_id]);
        for (int j=1; j<8; j<<=1)
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
        softmax[1][0][row_id] = alpha_max;
        alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
        alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
        float alpha_sum = alpha[0] + alpha[1];
        for (int j=1; j<8; j<<=1)
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
        alpha_sum += softmax[0][1][row_id]*__expf(softmax[0][0][row_id]-alpha_max);
        softmax[1][1][row_id] = alpha_sum;
        for (int j=0; j<2; j++)
            sparse_A[0][row_id][col_id[j]] = alpha[j] / (alpha_sum+1e-16f);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    for (int i=0; i<(block_num-1); i++) {
        // load
        if (lane_id < 16)
            SparseAToX_idx[(i+1)&1][lane_id] = SparseAToX[sparseAToX_off+(i+1)*16+lane_id];
        // __syncwarp();
        for (int j = 0; j < 4; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // softmax
        {
            int row_start = BlockRowMaskOffset[block_start+i+1];
            uint32_t col_mask = ColMask[colmask_off+i+1];
            int row_id = warp_id*4+(lane_id>>3);
            uint8_t row_mask[2] = {RowMask[row_start+__popc(col_mask&((1<<row_id)-1))*2],
                            RowMask[row_start+__popc(col_mask&((1<<row_id)-1))*2+1]};
            float mask[2], alpha[2];
            int col_id[2];
            for (int j=0; j<2; j++) {
                col_id[j] = (lane_id&7)*2+j;
                mask[j] = (col_mask>>row_id)&(*((uint16_t*)row_mask)>>col_id[j])&1;
                alpha[j] = (mask[j])?leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[(i+1)&1][col_id[j]], node_num-1)]):
                                        std::numeric_limits<float>::lowest();
            }
            float alpha_max = max(max(alpha[0],alpha[1]), softmax[(i+1)%3][0][row_id]);
            for (int j=1; j<8; j<<=1)
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            softmax[(i+2)%3][0][row_id] = alpha_max;
            alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
            alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
            float alpha_sum = alpha[0] + alpha[1];
            for (int j=1; j<8; j<<=1)
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            alpha_sum += softmax[(i+1)%3][1][row_id]*__expf(softmax[(i+1)%3][0][row_id]-alpha_max);
            softmax[(i+2)%3][1][row_id] = alpha_sum;
            for (int j=0; j<2; j++)
                sparse_A[(i+1)&1][row_id][col_id[j]] = alpha[j] / (alpha_sum+1e-16f);
        }
        //matmul
        {
            for (int j=0; j<2; j++) {
                float update = __expf(softmax[i%3][0][(lane_id&3)*2+j]-softmax[(i+1)%3][0][(lane_id&3)*2+j]) *
                            (softmax[i%3][1][(lane_id&3)*2+j]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+j]+1e-16f);
                for (int k=0; k<4; k++)
                    D[j+2*k] *= update;
            }
            uint32_t B[4], A[2];
            for (int j=0; j<2; j++) {
                for (int k=0; k<2; k++) {
                    for (int l=0; l<2; l++) {
                        if (SparseAToX_idx[i&1][(lane_id&3)+(k*2+l)*4] < node_num) {
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+(lane_id&3)*4+(lane_id>>2))&63]));
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+8+(lane_id&3)*4+(lane_id>>2))&63]));
                        } else {
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(0.f));
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(0.f));
                        }
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[l]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+(k*2+l)*4]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
                        : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                        "r"(A[0]), "r"(A[1]),
                        "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    int i = block_num - 1;
    //matmul
    {
        for (int j=0; j<2; j++) {
            float update = __expf(softmax[i%3][0][(lane_id&3)*2+j]-softmax[(i+1)%3][0][(lane_id&3)*2+j]) *
                        (softmax[i%3][1][(lane_id&3)*2+j]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+j]+1e-16f);
            for (int k=0; k<4; k++)
                D[j+2*k] *= update;
        }
        uint32_t B[4], A[2];
        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                for (int l=0; l<2; l++) {
                    if (SparseAToX_idx[i&1][(lane_id&3)+(k*2+l)*4] < node_num) {
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+(lane_id&3)*4+(lane_id>>2))&63]));
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+8+(lane_id&3)*4+(lane_id>>2))&63]));
                    } else {
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(0.f));
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(0.f));
                    }
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[l]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+(k*2+l)*4]));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                    "r"(A[0]), "r"(A[1]),
                    "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
            }
        }
    }
    for (int k=0; k<8; k++)
        if ((row_off+(lane_id&3)*2+(k&1))<node_num)
            out[(row_off+(lane_id&3)*2+(k&1))*64+warp_id*32+(k&6)*4+(lane_id>>2)] = D[k];
}

__global__ void gat_kernel_16x8_64(
    const int* __restrict__ RowWindowId,
    const int* __restrict__ RowWindowRowOffset,
    const int* __restrict__ RowWindowBlockOffset,
    const int* __restrict__ RowWindowSparseAToXOffset,
    const int* __restrict__ RowWindowColMaskOffset,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BlockRowMaskOffset,
    const uint8_t* __restrict__ ColMask,
    const uint8_t* __restrict__ RowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = RowWindowId[blockIdx.x];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowBlockOffset[bid];
    int block_num = RowWindowBlockOffset[bid+1]-RowWindowBlockOffset[bid];
    int sparseAToX_off = RowWindowSparseAToXOffset[bid];
    int row_off = RowWindowRowOffset[bid];
    int colmask_off = RowWindowColMaskOffset[bid];

    if(block_num == 0) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense[2][8][64];
    __shared__ float softmax[2][2][16];

    SparseAToX_idx[0][threadIdx.x>>4] = SparseAToX[sparseAToX_off + (threadIdx.x>>4)];

    softmax[0][0][lane_id&15] = std::numeric_limits<float>::lowest();
    softmax[0][1][lane_id&15] = 0.f;
    int cur_addr = __cvta_generic_to_shared(&dense[0][threadIdx.x>>4][(lane_id&15)*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[0][threadIdx.x>>4], node_num-1)*64+4*(lane_id&15)]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    uint32_t B[4];
    uint32_t A[4];

    for (int i=0; i<(block_num-1); i++) {
        // load
        SparseAToX_idx[(i+1)&1][threadIdx.x>>4] = SparseAToX[sparseAToX_off+(i+1)*8+(threadIdx.x>>4)];
        int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][threadIdx.x>>4][(lane_id&15)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][(threadIdx.x>>4)], node_num-1)*64+4*(lane_id&15)]));
        asm volatile("cp.async.commit_group;\n"::);
        for (int k = 0; k < 4; k++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][warp_id*16+(k&1)*8+(lane_id>>2)]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
        }
        // softmax
        {
            int row_start = BlockRowMaskOffset[block_start+i];
            float alpha[4];
            uint8_t col_mask[2] = {ColMask[colmask_off+i*2], ColMask[colmask_off+i*2+1]};
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int k = 0; k < 4; k++) {
                int row_id = (lane_id>>2)+(k&1)*8;
                int col_id = (lane_id&3)+(k&2)*2;
                uint32_t row_mask = RowMask[row_start+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))];
                mask[k] = (*((uint16_t*)col_mask)>>row_id)&(row_mask>>col_id)&1;
                if (mask[k])
                    alpha[k] = leaky_relu(alphai[row_off + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
                else
                    alpha[k] = std::numeric_limits<float>::lowest();
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
    int i = block_num - 1;
    for (int k = 0; k < 4; k++) {
        if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][warp_id*16+(k&1)*8+(lane_id>>2)]));
        else
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
    }
    // softmax
    {
        int row_start = BlockRowMaskOffset[block_start+i];
        float alpha[4];
        uint8_t col_mask[2] = {ColMask[colmask_off+i*2], ColMask[colmask_off+i*2+1]};
        float mask[4], alpha_max[2], alpha_sum[2];
        for (int k = 0; k < 4; k++) {
            int row_id = (lane_id>>2)+(k&1)*8;
            int col_id = (lane_id&3)+(k&2)*2;
            uint32_t row_mask = RowMask[row_start+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))];
            mask[k] = (*((uint16_t*)col_mask)>>row_id)&(row_mask>>col_id)&1;
            if (mask[k])
                alpha[k] = leaky_relu(alphai[row_off + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
            else
                alpha[k] = std::numeric_limits<float>::lowest();
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
        if ((row_off+(lane_id>>2)+(k&1)*8) < node_num)
            *(float2*)(&out[(row_off+(lane_id>>2)+(k&1)*8)*64+warp_id*16+(k&2)*4+(lane_id&3)*2]) = *(float2*)(&D[k*2]);
}

__global__ void gat_kernel_16x16_64(
    const int* __restrict__ RowWindowId,
    const int* __restrict__ RowWindowRowOffset,
    const int* __restrict__ RowWindowBlockOffset,
    const int* __restrict__ RowWindowSparseAToXOffset,
    const int* __restrict__ RowWindowColMaskOffset,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BlockRowMaskOffset,
    const uint8_t* __restrict__ ColMask,
    const uint8_t* __restrict__ RowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = RowWindowId[blockIdx.x];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowBlockOffset[bid];
    int block_num = RowWindowBlockOffset[bid+1] - RowWindowBlockOffset[bid];
    int row_off = RowWindowRowOffset[bid];
    int sparseAToX_off = RowWindowSparseAToXOffset[bid];
    int colmask_off = RowWindowColMaskOffset[bid];

    if (block_num == 0) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float dense[2][16][64];
    __shared__ float softmax[2][2][16];

    softmax[0][0][lane_id&15] = std::numeric_limits<float>::lowest();
    softmax[0][1][lane_id&15] = 0.f;

    for (int j = 0; j < 2; j++) {
        SparseAToX_idx[0][(threadIdx.x>>4)+j*8] = SparseAToX[sparseAToX_off+j*8+(threadIdx.x>>4)];
        int cur_addr = __cvta_generic_to_shared(&dense[0][j*8+(threadIdx.x>>4)][(lane_id&15)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[0][j*8+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    uint32_t A[8], B[8];
    for (int i=0; i<(block_num-1); i++) {
        // load
        for (int j = 0; j < 2; j++) {
            SparseAToX_idx[(i+1)&1][(threadIdx.x>>4)+j*8] = SparseAToX[sparseAToX_off+(i+1)*16+8*j+(threadIdx.x>>4)];
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*8+(threadIdx.x>>4)][(lane_id&15)*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*8+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        for (int j = 0; j < 8; j++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(j&6)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense[i&1][(lane_id&3)+(j&6)*2][warp_id*16+(j&1)*8+(lane_id>>2)]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
        }
        // softmax
        {
            int row_start_byte = BlockRowMaskOffset[i+block_start];
            float alpha[8];
            uint8_t col_mask[2] = {ColMask[colmask_off+i*2], ColMask[colmask_off+i*2+1]};
            float mask[8], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 8; j++) {
                int row_id = (lane_id>>2)+(j&1)*8;
                int col_id = (lane_id&3)+(j&6)*2;
                uint8_t row_mask[2] = {RowMask[row_start_byte+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))*2],
                                RowMask[row_start_byte+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))*2+1]};
                mask[j] = (*((uint16_t*)col_mask)>>row_id)&(*((uint16_t*)row_mask)>>col_id)&1;
                if (mask[j])
                    alpha[j] = leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id], node_num-1)]);
                else
                    alpha[j] = std::numeric_limits<float>::lowest();
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
                    D[j*2+(k&1)+(k&2)*2] *= update;
                }
            }
            for (int j = 0; j < 4; j++) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                    : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                    : "r"(A[(j&2)*2]), "r"(A[(j&2)*2+1]), "r"(A[(j&2)*2+2]), "r"(A[(j&2)*2+3]), 
                    "r"(B[(j&2)*2+(j&1)]), "r"(B[2+(j&2)*2+(j&1)]),
                    "f"(D[(j&1)*4]), "f"(D[(j&1)*4+1]), "f"(D[(j&1)*4+2]), "f"(D[(j&1)*4+3]));
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    int i = block_num-1;
    for (int j = 0; j < 8; j++) {
        if (SparseAToX_idx[i&1][(lane_id&3)+(j&6)*2] < node_num)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense[i&1][(lane_id&3)+(j&6)*2][warp_id*16+(j&1)*8+(lane_id>>2)]));
        else
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
    }
    // softmax
    {
        int row_start = BlockRowMaskOffset[i+block_start];
        float alpha[8];
        uint8_t col_mask[2] = {ColMask[colmask_off+i*2], ColMask[colmask_off+i*2+1]};
        float mask[8], alpha_max[2], alpha_sum[2];
        for (int j = 0; j < 8; j++) {
            int row_id = (lane_id>>2)+(j&1)*8;
            int col_id = (lane_id&3)+(j&6)*2;
            uint8_t row_mask[2] = {RowMask[row_start+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))*2],
                                RowMask[row_start+__popc(*((uint16_t*)col_mask)&((1<<row_id)-1))*2+1]};
            mask[j] = (*((uint16_t*)col_mask)>>row_id)&(*((uint16_t*)row_mask)>>col_id)&1;
            if (mask[j])
                alpha[j] = leaky_relu(alphai[min(row_off+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id], node_num-1)]);
            else
                alpha[j] = std::numeric_limits<float>::lowest();
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
                D[j*2+(k&1)+(k&2)*2] *= update;
            }
        }
        for (int j = 0; j < 4; j++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                : "r"(A[(j&2)*2]), "r"(A[(j&2)*2+1]), "r"(A[(j&2)*2+2]), "r"(A[(j&2)*2+3]), 
                "r"(B[(j&2)*2+(j&1)]), "r"(B[2+(j&2)*2+(j&1)]),
                "f"(D[(j&1)*4]), "f"(D[(j&1)*4+1]), "f"(D[(j&1)*4+2]), "f"(D[(j&1)*4+3]));
        }
    }
    for (int j = 0; j < 4; j++) 
        if ((row_off+(lane_id>>2)+(j&1)*8)<node_num)
            *(float2*)(&out[(row_off+(lane_id>>2)+(j&1)*8)*64+warp_id*16+(j&2)*4+(lane_id&3)*2]) = *(float2*)(&D[j*2]);
}

__global__ void gat_kernel_8x8_64_2_(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; //1 bit 
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    if(block_end == block_start) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float sparse_A[2][8][8];
    __shared__ float dense[2][8][64];
    __shared__ float softmax[3][2][8];

    int i = block_start;
    softmax[i%3][0][lane_id&7] = std::numeric_limits<float>::lowest();
    softmax[i%3][1][lane_id&7] = 0.f;
    // load
    if (lane_id < 8)
        SparseAToX_idx[i&1][lane_id] = SparseAToX[i*8+lane_id];
    for (int j=0; j<2; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense[i&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[i&1][j*4+(threadIdx.x>>4)], node_num-1)*64+4*(lane_id&15)]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    // softmax
    {
        int row_start = BitMaskRowOffset[i];
        uint32_t col_mask = BitColMask[i];
        int row_id = warp_id*4+(lane_id>>3);
        int col_id = lane_id&7;
        uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
        float mask = (col_mask>>row_id)&(row_mask>>col_id)&1;
        float alpha = (mask)?leaky_relu(alphai[min(bid*8+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id], node_num-1)]):
                                std::numeric_limits<float>::lowest();
        float alpha_max = max(alpha, softmax[i%3][0][row_id]);
        for (int j=1; j<8; j<<=1)
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
        softmax[(i+1)%3][0][row_id] = alpha_max;
        alpha = mask * __expf(alpha-alpha_max);
        float alpha_sum = alpha;
        for (int j=1; j<8; j<<=1)
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
        alpha_sum += softmax[i%3][1][row_id]*__expf(softmax[i%3][0][row_id]-alpha_max);
        softmax[(i+1)%3][1][row_id] = alpha_sum;
        sparse_A[i&1][row_id][col_id] = alpha / (alpha_sum+1e-16f);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    for (; i<(block_end-1); i++) {
        // load
        if (lane_id < 8)
            SparseAToX_idx[(i+1)&1][lane_id] = SparseAToX[(i+1)*8+lane_id];
        for (int j=0; j<2; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>4)], node_num-1)*64+4*(lane_id&15)]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // softmax
        {
            int row_start = BitMaskRowOffset[i+1];
            uint32_t col_mask = BitColMask[i+1];
            int row_id = warp_id*4+(lane_id>>3);
            int col_id = lane_id&7;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            float mask = (col_mask>>row_id)&(row_mask>>col_id)&1;
            float alpha = (mask)?leaky_relu(alphai[min(bid*8+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[(i+1)&1][col_id], node_num-1)]):
                                    std::numeric_limits<float>::lowest();
            float alpha_max = max(alpha, softmax[(i+1)%3][0][row_id]);
            for (int j=1; j<8; j<<=1)
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            softmax[(i+2)%3][0][row_id] = alpha_max;
            alpha = mask * __expf(alpha-alpha_max);
            float alpha_sum = alpha;
            for (int j=1; j<8; j<<=1)
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            alpha_sum += softmax[(i+1)%3][1][row_id]*__expf(softmax[(i+1)%3][0][row_id]-alpha_max);
            softmax[(i+2)%3][1][row_id] = alpha_sum;
            sparse_A[(i+1)&1][row_id][col_id] = alpha / (alpha_sum+1e-16f);
        }
        // matmul
        {
            uint32_t B[8], A[2];
            for (int k=0; k<8; k++) {
                if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][(warp_id*32+(k&3)*8+((k&4)+(lane_id&3))*4+(lane_id>>2))&63]));
                else
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f)); 
            }
            for (int j = 0; j < 2; j++)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+j*4])); 
            for (int k=0; k<2; k++) {
                float update = __expf(softmax[i%3][0][(lane_id&3)*2+k]-softmax[(i+1)%3][0][(lane_id&3)*2+k])
                            *(softmax[i%3][1][(lane_id&3)*2+k]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+k]+1e-16f);
                for (int j=0; j<4; j++)
                    D[k+j*2] *= update;
            }
            for (int j = 0; j < 2; j++)
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3]) 
                    : "r"(B[j*4]), "r"(B[j*4+1]), "r"(B[j*4+2]), "r"(B[j*4+3]), 
                    "r"(A[0]), "r"(A[1]),
                    "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end - 1;
    // matmul
    {
        uint32_t B[8], A[2];
        for (int k=0; k<8; k++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(k&2)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense[i&1][(lane_id&3)+(k&2)*2][(warp_id*32+(k&3)*8+((k&4)+(lane_id&3))*4+(lane_id>>2))&63]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f)); 
        }
        for (int j = 0; j < 2; j++)
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+j*4])); 
        for (int k=0; k<2; k++) {
            float update = __expf(softmax[i%3][0][(lane_id&3)*2+k]-softmax[(i+1)%3][0][(lane_id&3)*2+k])
                        *(softmax[i%3][1][(lane_id&3)*2+k]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+k]+1e-16f);
            for (int j=0; j<4; j++)
                D[k+j*2] *= update;
        }
        for (int j = 0; j < 2; j++)
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3]) 
                : "r"(B[j*4]), "r"(B[j*4+1]), "r"(B[j*4+2]), "r"(B[j*4+3]), 
                "r"(A[0]), "r"(A[1]),
                "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
    }
    for (int k=0; k<8; k++)
        if ((bid*8+(lane_id&3)*2+(k&1))<node_num)
            out[(bid*8+(lane_id&3)*2+(k&1))*64+warp_id*32+(k&6)*4+(lane_id>>2)] = D[k];
}

__global__ void gat_kernel_8x16_64_2_(
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint8_t* __restrict__ BitColMask,
    const uint16_t* __restrict__ BitRowMask,
    const float* __restrict__ feat,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    float* __restrict__ out,
    const int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; // 1 bit
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];

    if(block_end == block_start) return;

    __shared__ int SparseAToX_idx[2][16];
    __shared__ float sparse_A[2][8][16];
    __shared__ float dense[2][16][64];
    __shared__ float softmax[3][2][8];

    int i = block_start;
    softmax[i%3][0][warp_id*4+(lane_id>>3)] = std::numeric_limits<float>::lowest();
    softmax[i%3][1][warp_id*4+(lane_id>>3)] = 0.f;
    // load
    if (lane_id < 16)
        SparseAToX_idx[i&1][lane_id] = SparseAToX[i*16+lane_id];
    for (int j = 0; j < 4; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense[i&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[i&1][j*4+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    // softmax
    {
        int row_start = BitMaskRowOffset[i]/2;
        uint32_t col_mask = BitColMask[i];
        int row_id = warp_id*4+(lane_id>>3);
        uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
        float mask[2], alpha[2];
        int col_id[2];
        for (int j=0; j<2; j++) {
            col_id[j] = (lane_id&7)*2+j;
            mask[j] = (col_mask>>row_id)&(row_mask>>col_id[j])&1;
            alpha[j] = (mask[j])?leaky_relu(alphai[min(bid*8+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id[j]], node_num-1)]):
                                    std::numeric_limits<float>::lowest();
        }
        float alpha_max = max(max(alpha[0],alpha[1]), softmax[i%3][0][row_id]);
        for (int j=1; j<8; j<<=1)
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
        softmax[(i+1)%3][0][row_id] = alpha_max;
        alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
        alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
        float alpha_sum = alpha[0] + alpha[1];
        for (int j=1; j<8; j<<=1)
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
        alpha_sum += softmax[i%3][1][row_id]*__expf(softmax[i%3][0][row_id]-alpha_max);
        softmax[(i+1)%3][1][row_id] = alpha_sum;
        for (int j=0; j<2; j++)
            sparse_A[i&1][row_id][col_id[j]] = alpha[j] / (alpha_sum+1e-16f);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    for (; i < (block_end - 1); i++) {
        // load
        if (lane_id < 16)
            SparseAToX_idx[(i+1)&1][lane_id] = SparseAToX[(i+1)*16+lane_id];
        // __syncwarp();
        for (int j = 0; j < 4; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*4+(threadIdx.x>>4)][(((lane_id&15)+j*4+(threadIdx.x>>4))*4)&63]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // softmax
        {
            int row_start = BitMaskRowOffset[i+1]/2;
            uint32_t col_mask = BitColMask[i+1];
            int row_id = warp_id*4+(lane_id>>3);
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            float mask[2], alpha[2];
            int col_id[2];
            for (int j=0; j<2; j++) {
                col_id[j] = (lane_id&7)*2+j;
                mask[j] = (col_mask>>row_id)&(row_mask>>col_id[j])&1;
                alpha[j] = (mask[j])?leaky_relu(alphai[min(bid*8+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[(i+1)&1][col_id[j]], node_num-1)]):
                                        std::numeric_limits<float>::lowest();
            }
            float alpha_max = max(max(alpha[0],alpha[1]), softmax[(i+1)%3][0][row_id]);
            for (int j=1; j<8; j<<=1)
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2*j-1, 8));
            softmax[(i+2)%3][0][row_id] = alpha_max;
            alpha[0] = mask[0] * __expf(alpha[0]-alpha_max);
            alpha[1] = mask[1] * __expf(alpha[1]-alpha_max);
            float alpha_sum = alpha[0] + alpha[1];
            for (int j=1; j<8; j<<=1)
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2*j-1, 8);
            alpha_sum += softmax[(i+1)%3][1][row_id]*__expf(softmax[(i+1)%3][0][row_id]-alpha_max);
            softmax[(i+2)%3][1][row_id] = alpha_sum;
            for (int j=0; j<2; j++)
                sparse_A[(i+1)&1][row_id][col_id[j]] = alpha[j] / (alpha_sum+1e-16f);
        }
        //matmul
        {
            for (int j=0; j<2; j++) {
                float update = __expf(softmax[i%3][0][(lane_id&3)*2+j]-softmax[(i+1)%3][0][(lane_id&3)*2+j]) *
                            (softmax[i%3][1][(lane_id&3)*2+j]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+j]+1e-16f);
                for (int k=0; k<4; k++)
                    D[j+2*k] *= update;
            }
            uint32_t B[4], A[2];
            for (int j=0; j<2; j++) {
                for (int k=0; k<2; k++) {
                    for (int l=0; l<2; l++) {
                        if (SparseAToX_idx[i&1][(lane_id&3)+(k*2+l)*4] < node_num) {
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+(lane_id&3)*4+(lane_id>>2))&63]));
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+8+(lane_id&3)*4+(lane_id>>2))&63]));
                        } else {
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(0.f));
                            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(0.f));
                        }
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[l]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+(k*2+l)*4]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
                        : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                        "r"(A[0]), "r"(A[1]),
                        "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end -1;
    //matmul
    {
        for (int j=0; j<2; j++) {
            float update = __expf(softmax[i%3][0][(lane_id&3)*2+j]-softmax[(i+1)%3][0][(lane_id&3)*2+j]) *
                        (softmax[i%3][1][(lane_id&3)*2+j]+1e-16f)/(softmax[(i+1)%3][1][(lane_id&3)*2+j]+1e-16f);
            for (int k=0; k<4; k++)
                D[j+2*k] *= update;
        }
        uint32_t B[4], A[2];
        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                for (int l=0; l<2; l++) {
                    if (SparseAToX_idx[i&1][(lane_id&3)+(k*2+l)*4] < node_num) {
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+(lane_id&3)*4+(lane_id>>2))&63]));
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(dense[i&1][(lane_id&3)+(k*2+l)*4][(warp_id*32+(k*2+l+j)*16+8+(lane_id&3)*4+(lane_id>>2))&63]));
                    } else {
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2]) : "f"(0.f));
                        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[l*2+1]) : "f"(0.f));
                    }
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[l]) : "f"(sparse_A[i&1][lane_id>>2][(lane_id&3)+(k*2+l)*4]));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
                    : "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
                    "r"(A[0]), "r"(A[1]),
                    "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
            }
        }
    }
    for (int k=0; k<8; k++)
        if ((bid*8+(lane_id&3)*2+(k&1))<node_num)
            out[(bid*8+(lane_id&3)*2+(k&1))*64+warp_id*32+(k&6)*4+(lane_id>>2)] = D[k];
}

__global__ void gat_kernel_16x8_64_2_(
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
    // if (warp_id == 0) {
    softmax[i&1][0][lane_0_4] = std::numeric_limits<float>::lowest();
    softmax[i&1][1][lane_0_4] = 0.f;
    // }
    int cur_addr = __cvta_generic_to_shared(&dense[i&1][thread_4_7][lane_0_4*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[i&1][thread_4_7], node_num-1)*64+4*lane_0_4]));
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
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][thread_4_7], node_num-1)*64+4*lane_0_4]));
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
            float alpha[4];
            uint32_t col_mask = BitColMask[i];
            float mask[4], alpha_max[2], alpha_sum[2];
            for (int k = 0; k < 4; k++) {
                int row_id = (lane_id>>2)+(k&1)*8;
                int col_id = (lane_id&3)+(k&2)*2;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                if (mask[k])
                    alpha[k] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
                else
                    alpha[k] = std::numeric_limits<float>::lowest();
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
        float alpha[4]; 
        uint32_t col_mask = BitColMask[i];
        float mask[4], alpha_max[2], alpha_sum[2];
        for (int k = 0; k < 4; k++) {
            int row_id = (lane_id>>2)+(k&1)*8;
            int col_id = (lane_id&3)+(k&2)*2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            mask[k] = (col_mask>>row_id)&(row_mask>>col_id)&1;
            if (mask[k])
                alpha[k] = leaky_relu(alphai[bid * 16 + row_id] + alphaj[SparseAToX_idx[i&1][col_id]]);
            else 
                alpha[k] = std::numeric_limits<float>::lowest();
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
        *(float2*)(&out[(bid*16+(lane_id>>2)+(k&1)*8)*64+warp_id*16+(k&2)*4+(lane_id&3)*2]) = *(float2*)(&D[k*2]);
}

__global__ void gat_kernel_16x16_64_2_(
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
    // if (warp_id == 0) {
    softmax[i&1][0][lane_id&15] = std::numeric_limits<float>::lowest();
    softmax[i&1][1][lane_id&15] = 0.f;
    // }
    for (int j = 0; j < 2; j++) {
        SparseAToX_idx[i&1][(threadIdx.x>>4)+j*8] = SparseAToX[i*16+j*8+(threadIdx.x>>4)];
        int cur_addr = __cvta_generic_to_shared(&dense[i&1][j*8+(threadIdx.x>>4)][(lane_id&15)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[i&1][j*8+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float D[8] = {0.f};
    uint32_t A[8], B[8];
    for (int i = block_start; i < (block_end - 1); i++) {
        // load and init
        for (int j = 0; j < 2; j++) {
            SparseAToX_idx[(i+1)&1][(threadIdx.x>>4)+j*8] = SparseAToX[(i+1)*16+8*j+(threadIdx.x>>4)];
            int cur_addr = __cvta_generic_to_shared(&dense[(i+1)&1][j*8+(threadIdx.x>>4)][(lane_id&15)*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(cur_addr), "l"(&feat[min(SparseAToX_idx[(i+1)&1][j*8+(threadIdx.x>>4)], node_num-1)*64+(lane_id&15)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        for (int j = 0; j < 8; j++) {
            if (SparseAToX_idx[i&1][(lane_id&3)+(j&6)*2] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(dense[i&1][(lane_id&3)+(j&6)*2][warp_id*16+(j&1)*8+(lane_id>>2)]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[j]) : "f"(0.f));
        }
        // softmax
        {
            int row_start = BitMaskRowOffset[i]/2;
            // int row_end = BitMaskRowOffset[i+1];
            float alpha[8];
            uint32_t col_mask = BitColMask[i];
            float mask[8], alpha_max[2], alpha_sum[2];
            for (int j = 0; j < 8; j++) {
                int row_id = (lane_id>>2)+(j&1)*8;
                int col_id = (lane_id&3)+(j&6)*2;
                uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
                mask[j] = (col_mask>>row_id)&(row_mask>>col_id)&1;
                if (mask[j])
                    alpha[j] = leaky_relu(alphai[min(bid*16+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id], node_num-1)]);
                else
                    alpha[j] = std::numeric_limits<float>::lowest();
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
                    D[j*2+(k&1)+(k&2)*2] *= update;
                }
            }
            for (int j = 0; j < 4; j++) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                    : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                    : "r"(A[(j&2)*2]), "r"(A[(j&2)*2+1]), "r"(A[(j&2)*2+2]), "r"(A[(j&2)*2+3]), 
                    "r"(B[(j&2)*2+(j&1)]), "r"(B[2+(j&2)*2+(j&1)]),
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
        int row_start = BitMaskRowOffset[i]/2;
        // int row_end = BitMaskRowOffset[i+1];
        float alpha[8];
        uint32_t col_mask = BitColMask[i];
        float mask[8], alpha_max[2], alpha_sum[2];
        for (int j = 0; j < 8; j++) {
            int row_id = (lane_id>>2)+(j&1)*8;
            int col_id = (lane_id&3)+(j&6)*2;
            uint32_t row_mask = BitRowMask[row_start+__popc(col_mask&((1<<row_id)-1))];
            mask[j] = (col_mask>>row_id)&(row_mask>>col_id)&1;
            if (mask[j])
                alpha[j] = leaky_relu(alphai[min(bid*16+row_id, node_num-1)]+alphaj[min(SparseAToX_idx[i&1][col_id], node_num-1)]);
            else
                alpha[j] = std::numeric_limits<float>::lowest();
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
                D[j*2+(k&1)+(k&2)*2] *= update;
            }
        }
        for (int j = 0; j < 4; j++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" 
                : "=f"(D[(j&1)*4]), "=f"(D[(j&1)*4+1]), "=f"(D[(j&1)*4+2]), "=f"(D[(j&1)*4+3]) 
                : "r"(A[(j&2)*2]), "r"(A[(j&2)*2+1]), "r"(A[(j&2)*2+2]), "r"(A[(j&2)*2+3]), 
                "r"(B[(j&2)*2+(j&1)]), "r"(B[2+(j&2)*2+(j&1)]),
                "f"(D[(j&1)*4]), "f"(D[(j&1)*4+1]), "f"(D[(j&1)*4+2]), "f"(D[(j&1)*4+3]));
        }
    }
    for (int j = 0; j < 4; j++) 
        if ((bid*16+(lane_id>>2)+(j&1)*8)<node_num)
            *(float2*)(&out[(bid*16+(lane_id>>2)+(j&1)*8)*64+warp_id*16+(j&2)*4+(lane_id&3)*2]) = *(float2*)(&D[j*2]);
}

at::Tensor GAT_adaptive(
    at::Tensor feature,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    at::Tensor RowWindowId_8x8,
    at::Tensor RowWindowId_8x16,
    at::Tensor RowWindowId_16x8,
    at::Tensor RowWindowId_16x16,
    at::Tensor RowWindowRowOffset,
    at::Tensor RowWindowBlockOffset,
    at::Tensor RowWindowSparseAToXOffset,
    at::Tensor RowWindowColMaskOffset,
    at::Tensor BlockRowMaskOffset,
    at::Tensor ColMask,
    at::Tensor RowMask,
    at::Tensor SparseAToX
) {
    int num_nodes = feature.size(0);
    auto feats = torch::mm(feature, lin_weight.t());
    auto alpha_i = torch::matmul(feats, att_i.squeeze());
    auto alpha_j = torch::matmul(feats, att_j.squeeze());
    int out_feats = feats.size(1);
    auto output = at::empty({num_nodes, out_feats}, feature.options()).fill_(0);
    int block_8x8 = RowWindowId_8x8.size(0);
    int block_8x16 = RowWindowId_8x16.size(0);
    int block_16x8 = RowWindowId_16x8.size(0);
    int block_16x16 = RowWindowId_16x16.size(0);
    uint8_t select = (block_8x8>0)+(block_8x16>0)*(1<<1)+(block_16x8>0)*(1<<2)+(block_16x16>0)*(1<<3);
    cudaStream_t stream[4];
    cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream[2], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream[3], cudaStreamNonBlocking);
    switch(select) {
        case 1:
            gat_kernel_8x8_64_2_<<<block_8x8, 64>>>(
                RowWindowBlockOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 2:
            gat_kernel_8x16_64_2_<<<block_8x16, 64>>>(
                RowWindowBlockOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                (uint16_t*)RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 4:
            gat_kernel_16x8_64_2_<<<block_16x8, 128>>>(
                RowWindowBlockOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                (uint16_t*)ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 8:
            gat_kernel_16x16_64_2_<<<block_16x16, 128>>>(
                RowWindowBlockOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                (uint16_t*)ColMask.data_ptr<uint8_t>(),
                (uint16_t*)RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 3:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_8x8_64<<<block_8x16, 64, 0, stream[1]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 5:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[1]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 9:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[1]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 6:
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[0]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[1]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 10:
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[0]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[1]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 12:
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[0]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[1]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 7:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[1]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[2]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 11:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[1]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[2]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 13:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[1]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[2]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 14:
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[0]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[1]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[2]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        case 15:
            gat_kernel_8x8_64<<<block_8x8, 64, 0, stream[0]>>>(
                RowWindowId_8x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_8x16_64<<<block_8x16, 64, 0, stream[1]>>>(
                RowWindowId_8x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x8_64<<<block_16x8, 128, 0, stream[2]>>>(
                RowWindowId_16x8.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            gat_kernel_16x16_64<<<block_16x16, 128, 0, stream[3]>>>(
                RowWindowId_16x16.data_ptr<int>(),
                RowWindowRowOffset.data_ptr<int>(),
                RowWindowBlockOffset.data_ptr<int>(),
                RowWindowSparseAToXOffset.data_ptr<int>(),
                RowWindowColMaskOffset.data_ptr<int>(),
                SparseAToX.data_ptr<int>(),
                BlockRowMaskOffset.data_ptr<int>(),
                ColMask.data_ptr<uint8_t>(),
                RowMask.data_ptr<uint8_t>(),
                feats.data_ptr<float>(),
                alpha_i.data_ptr<float>(),
                alpha_j.data_ptr<float>(),
                output.data_ptr<float>(),
                num_nodes);
            break;
        default:
            printf("Unsupported select: %d\n", select);
            exit(1);
    }
    return output;
}