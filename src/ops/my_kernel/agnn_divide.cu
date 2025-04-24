#include <cuda_runtime.h>
#include <cstdint>
#include "agnn.cuh"

#define FULL_MASK 0xffffffff

__global__ void SDDMM_kernel_16x8_32(
    const float* __restrict__ x_norm,
    const float* __restrict__ x,
    const float* __restrict__ beta,
    const int* row_pointers,
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    float* __restrict__ attention,
    int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if (block_start == block_end) return;
    int write_start[2] = {row_pointers[bid*16], row_pointers[min(bid*16+8, node_num)]};

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense_x[2][8][32];

    uint32_t D[16], B[2];
    for (int i = 0; i < 16; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+8*(i&1)+(lane_id>>2), node_num-1)*32+(i>>2)*8+(lane_id&3)+(i&2)*2]/x_norm[min(bid*16+8*(i&1)+(lane_id>>2), node_num-1)]
        ));
    }
    int i = block_start;
    for (int j = 0; j < 2; j++) {
        SparseAToX_idx[i&1][j*4+(threadIdx.x>>3)] = SparseAToX[i*8+j*4+(threadIdx.x>>3)];
        int cur_addr = __cvta_generic_to_shared(&dense_x[i&1][j*4+(threadIdx.x>>3)][(lane_id&7)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[i&1][j*4+(threadIdx.x>>3)]*32+(lane_id&7)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (i=block_start; i<(block_end-1); i+=1) {
        // load (global->shared_mem)
        for (int j = 0; j < 2; j++) {
            SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>3)] = SparseAToX[i*8+j*4+(threadIdx.x>>3)];
            int cur_addr = __cvta_generic_to_shared(&dense_x[(i+1)&1][j*4+(threadIdx.x>>3)][(lane_id&7)*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>3)]*32+(lane_id&7)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[4] = {0.f};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 2; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][lane_id>>2][j*8+k*4+((lane_id&3))]/x_norm[SparseAToX_idx[i&1][lane_id>>2]]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
        }
        // write back
        {
            // int row_start = BitMaskRowOffset[i];
            // int row_end = BitMaskRowOffset[i+1];         
            // uint32_t col_mask = BitColMask[i];
            // for (int j = 0; j < 2; j++) {
            //     int row_id = (lane_id>>2)+j*8;
            //     int row_mask_offset = __popc(col_mask&((1<<row_id)-1));
            //     uint32_t row_mask = ((col_mask>>row_id)&1) * BitRowMask[row_start+row_mask_offset];
            //     for (int k = 0; k < 2; k++) {
            //         int col_id = (lane_id&3)*2+k;
            //         if ((row_mask>>col_id)&1)
            //             attention[write_start[j]+__popc(row_mask&((1<<col_id)-1))] = alpha[j*2+k] * beta[0];
            //     }
            //     write_start[j] += __popc(row_mask);
            // }
            attention[0] = 1.f;
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end - 1;
    // SDDMM
    float alpha[4] = {0.f};
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][lane_id>>2][j*8+k*4+((lane_id&3))]/x_norm[SparseAToX_idx[i&1][lane_id>>2]]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
            : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
    }
    // write back
    {
        int row_start = BitMaskRowOffset[i];
        int row_end = BitMaskRowOffset[i+1];         
        uint32_t col_mask = BitColMask[i];
        for (int j = 0; j < 2; j++) {
            int row_id = (lane_id>>2)+j*8;
            int row_mask_offset = __popc(col_mask&((1<<row_id)-1));
            uint32_t row_mask = ((col_mask>>row_id)&1) * BitRowMask[row_start+row_mask_offset];
            for (int k = 0; k < 2; k++) {
                int col_id = (lane_id&3)*2+k;
                if ((row_mask>>col_id)&1)
                    attention[write_start[j]+__popc(row_mask&((1<<col_id)-1))] = alpha[j*2+k] * beta[0];
            }
            write_start[j] += __popc(row_mask);
        }
    }
}

__global__ void softmax(const float *__restrict__ features, const int *__restrict__ pointer,
                         float *__restrict__ next_layer)
{
    int neighbor_offset = pointer[blockIdx.x];
    int degree = pointer[blockIdx.x + 1] - neighbor_offset;

    float max_local = 0.0f;
    for (int i = 0; i < degree / 32; i++) {
        max_local = max(features[neighbor_offset + i * 32 + threadIdx.x],
                        max_local);
    }
    if (threadIdx.x < degree % 32) {
        max_local = max(features[neighbor_offset + degree - (degree % 32) +
                                        threadIdx.x],
                        max_local);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        max_local = max(__shfl_down_sync(FULL_MASK, max_local, offset), max_local);
    }
    max_local = __shfl_sync(FULL_MASK, max_local, 0);

    float exp_local = 0.0f;
    for (int i = 0; i < degree / 32; i++) {
        exp_local += expf(
            features[neighbor_offset + i * 32 + threadIdx.x] - max_local);
    }
    if (threadIdx.x < degree % 32) {
        exp_local += expf(features[neighbor_offset + degree -
                                        (degree % 32) + threadIdx.x] -
                        max_local);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        exp_local += __shfl_down_sync(FULL_MASK, exp_local, offset);
    }
    float sum_exp_local = 1 / __shfl_sync(FULL_MASK, exp_local, 0);

    for (int i = 0; i < degree / 32; i++) {
        int neighbor = neighbor_offset + i * 32 + threadIdx.x;
        next_layer[neighbor] = expf(features[neighbor] - max_local) * sum_exp_local;
    }
    if (threadIdx.x < degree % 32) {
        int neighbor = neighbor_offset + degree - (degree % 32) + threadIdx.x;
        next_layer[neighbor] = expf(features[neighbor] - max_local) * sum_exp_local;
    }
    return;
}

__global__ void SpMM_kernel_16x8_32(
    const float* __restrict__ attention,
    const float* __restrict__ features,
    const int* __restrict__ row_pointers,
    const int* __restrict__ RowWindowOffsets,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ BitMaskRowOffset,
    const uint16_t* __restrict__ BitColMask,
    const uint8_t* __restrict__ BitRowMask,
    float* __restrict__ output,
    int node_num
) {
    int bid = blockIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int block_start = RowWindowOffsets[bid];
    int block_end = RowWindowOffsets[bid+1];
    if (block_start == block_end) return;

    __shared__ int SparseAToX_idx[2][8];
    __shared__ float dense_x[2][8][32];

    uint32_t A[4], B[2];
    float D[8] = {0.f};
    int read_start[2] = {row_pointers[bid*16], row_pointers[min(bid*16+8, node_num)]};

    int i = block_start;
    SparseAToX_idx[i&1][threadIdx.x>>3] = SparseAToX[i*8+(threadIdx.x>>3)];
    int cur_addr = __cvta_generic_to_shared(&dense_x[i&1][threadIdx.x>>3][(lane_id&7)*4]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&features[SparseAToX_idx[i&1][threadIdx.x>>3]*32+(lane_id&7)*4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    for (i=block_start; i<(block_end-1); i+=1) {
        // load (global->shared_mem)
        SparseAToX_idx[(i+1)&1][threadIdx.x>>3] = SparseAToX[(i+1)*8+(threadIdx.x>>3)];
        cur_addr = __cvta_generic_to_shared(&dense_x[(i+1)&1][threadIdx.x>>3][(lane_id&7)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&features[SparseAToX_idx[(i+1)&1][threadIdx.x>>3]*32+(lane_id&7)*4]));
        asm volatile("cp.async.commit_group;\n"::);
        // load A 
        uint16_t col_mask = BitColMask[i];
        int rowmask_start = BitMaskRowOffset[i];
        for (int j = 0; j < 2; j++) {
            int row_id = (lane_id>>2)+j*8;
            uint8_t row_mask = 0;
            if ((col_mask>>row_id)&1) 
                row_mask = BitRowMask[rowmask_start+__popc(col_mask&((1<<row_id)-1))];
            for (int k = 0; k < 2; k++) {
                int col_id = (lane_id&3)+k*4;
                int read_offset = __popc(row_mask&((1<<col_id)-1));
                if ((row_mask>>col_id)&1)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j+k*2]) : "f"(attention[read_start[j]+read_offset]));
                else
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j+k*2]) : "f"(0.f));
            }
            read_start[j] += __popc(row_mask);
        }
        // spmm
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int col_idx = (lane_id&3)+k*4;
                if (SparseAToX_idx[i&1][col_idx] < node_num)
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][col_idx][(lane_id>>2)+j*8+warp_id*16]));
                else 
                    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
        }
    }
    i = block_end - 1;
    // load A 
    uint16_t col_mask = BitColMask[i];
    int rowmask_start = BitMaskRowOffset[i];
    for (int j = 0; j < 2; j++) {
        int row_id = (lane_id>>2)+j*8;
        uint8_t row_mask = 0;
        if ((col_mask>>row_id)&1) 
            row_mask = BitRowMask[rowmask_start+__popc(col_mask&((1<<row_id)-1))];
        for (int k = 0; k < 2; k++) {
            int col_id = (lane_id&3)+k*4;
            int read_offset = __popc(row_mask&((1<<col_id)-1));
            if ((row_mask>>col_id)&1)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j+k*2]) : "f"(attention[read_start[j]+read_offset]));
            else
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(A[j+k*2]) : "f"(0.f));
        }
        read_start[j] += __popc(row_mask);
    }
    // spmm
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
            int col_idx = (lane_id&3)+k*4;
            if (SparseAToX_idx[i&1][col_idx] < node_num)
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][col_idx][(lane_id>>2)+j*8+warp_id*16]));
            else 
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(0.f));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(D[j*4]), "=f"(D[j*4+1]), "=f"(D[j*4+2]), "=f"(D[j*4+3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(D[j*4]), "f"(D[j*4+1]), "f"(D[j*4+2]), "f"(D[j*4+3]));
    }
    for (int j = 0; j < 4; j++) {
        *(float2*)&output[(bid*16+(j&1)*8)*32+(lane_id&3)*2+(j&2)*4+warp_id*16] = *(float2*)&D[j*4];
    }
}

__global__ void SDDMM_TCGNN_kernel(
    const float* __restrict__ x_norm,
    const float* __restrict__ x,
    const float* __restrict__ beta,
    const int* __restrict__ row_pointers,
    const int* __restrict__ column_index,
    const int* __restrict__ blockPartition,
    const int* __restrict__ edgeToColumn,
    const int* __restrict__ edgeToRow,
    const int numNodes,
    const int numEdges,
    float* __restrict__ attention
) {
    int bid = blockIdx.x;
    int block_num = blockPartition[bid];
    if (block_num == 0) return;
    int element_start = row_pointers[bid*16];
    int element_end = row_pointers[min(bid*16+16, numNodes)];
    int write_start[2] = {row_pointers[bid*16], row_pointers[min(bid*16+8, numNodes)]};

    __shared__ int SparseAToX_idx[2][8];
    __shared__ uint32_t mask[2][4];
    __shared__ float dense_x[2][8][32];

    uint32_t D[16], B[2];
    for (int i = 0; i < 16; i++) {
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(D[i]) : "f"(
            x[min(bid*16+8*(i&1)+(threadIdx.x>>2), numNodes-1)*32+(i>>2)*8+(threadIdx.x&3)+(i&2)*2]/x_norm[min(bid*16+8*(i&1)+(threadIdx.x>>2), numNodes-1)]
        ));
    }
    int i = 0;
    if (threadIdx.x < 4)
        mask[0][threadIdx.x] = 0;
    if (threadIdx.x < 8)
        SparseAToX_idx[0][threadIdx.x] = 0;
    __syncthreads();
    for (int element_idx = threadIdx.x + element_start; element_idx < element_end; element_idx += blockDim.x) {
        int col = edgeToColumn[element_idx];
        int row = edgeToRow[element_idx];
        if (col < 8) {
            int local_col = ((col&7)+(row&15)*8)&31;
            int local_row = ((col&7)+(row&15)*8)>>5;
            atomicOr(&mask[0][local_row], 1<<local_col);
            SparseAToX_idx[0][col] = column_index[element_idx];
        }
    }
    __syncthreads();
    for (int j = 0; j < 2; j++) {
        int cur_addr = __cvta_generic_to_shared(&dense_x[i&1][j*4+(threadIdx.x>>3)][(threadIdx.x&7)*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[i&1][j*4+(threadIdx.x>>3)]*32+(threadIdx.x&7)*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (int i = 0; i < block_num; i++) {
        // load
        if (threadIdx.x < 4)
            mask[(i+1)&1][threadIdx.x] = 0;
        if (threadIdx.x < 8)
            SparseAToX_idx[(i+1)&1][threadIdx.x] = 0;
        __syncthreads();
        for (int element_idx = threadIdx.x + element_start; element_idx < element_end; element_idx += blockDim.x) {
            int col = edgeToColumn[element_idx];
            int row = edgeToRow[element_idx];
            if (col >= 8 * (i+1) && col < 8 * (i+2)) {
                int local_col = ((col&7)+(row&15)*8)&31;
                int local_row = ((col&7)+(row&15)*8)>>5;
                atomicOr(&mask[(i+1)&1][local_row], 1<<local_col);
                SparseAToX_idx[(i+1)&1][col-8*(i+1)] = column_index[element_idx];
            }
        }
        __syncthreads();
        for (int j = 0; j < 2; j++) {
            int cur_addr = __cvta_generic_to_shared(&dense_x[(i+1)&1][j*4+(threadIdx.x>>3)][(threadIdx.x&7)*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[SparseAToX_idx[(i+1)&1][j*4+(threadIdx.x>>3)]*32+(threadIdx.x&7)*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        // SDDMM
        float alpha[4] = {0.f};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 2; k++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][threadIdx.x>>2][j*8+k*4+((threadIdx.x&3))]/x_norm[SparseAToX_idx[i&1][threadIdx.x>>2]]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
                : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
        }
        // write back
        for (int j = 0; j < 2; j++) {
            int co_mask = mask[i&1][j*2+(threadIdx.x>>4)];
            for (int k = 0; k < 2; k++) {
                int col_offset = (threadIdx.x&15)*2+k;
                if (co_mask & (1<<col_offset)) {
                    int col_id = (threadIdx.x&3)*2+k;
                    uint32_t row_mask = (co_mask>>8*((threadIdx.x>>2)&3));
                    attention[write_start[j]+__popc(row_mask&((1<<col_id)-1))] = alpha[j*2+k] * beta[0];
                }
                write_start[j] += __popc(co_mask&7);
            }
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_num - 1;
    // SDDMM
    float alpha[4] = {0.f};
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(B[k]) : "f"(dense_x[i&1][threadIdx.x>>2][j*8+k*4+((threadIdx.x&3))]/x_norm[SparseAToX_idx[i&1][threadIdx.x>>2]]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(alpha[0]), "=f"(alpha[1]), "=f"(alpha[2]), "=f"(alpha[3])
            : "r"(D[j*4]), "r"(D[j*4+1]), "r"(D[j*4+2]), "r"(D[j*4+3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(alpha[0]), "f"(alpha[1]), "f"(alpha[2]), "f"(alpha[3]));
    }
    // write back
    for (int j = 0; j < 2; j++) {
        int co_mask = mask[i&1][j*2+(threadIdx.x>>4)];
        for (int k = 0; k < 2; k++) {
            int col_offset = (threadIdx.x&15)*2+k;
            if (co_mask & (1<<col_offset)) {
                int col_id = (threadIdx.x&3)*2+k;
                uint32_t row_mask = (co_mask>>8*((threadIdx.x>>2)&3));
                attention[write_start[j]+__popc(row_mask&((1<<col_id)-1))] = alpha[j*2+k] * beta[0];
            }
            write_start[j] += __popc(co_mask&7);
        }
    }
}

at::Tensor SDDMM(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta
) {
    int num_nodes = feature.size(0);
    auto x_norm = feature.norm(2, -1).clamp_min(1e-12);
    int blocks = (num_nodes + 15) / 16;
    int edge_num;
    cudaMemcpy(&edge_num, row_pointers.data_ptr<int>()+num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    auto attention = torch::empty({edge_num}, feature.options());

    SDDMM_kernel_16x8_32<<<blocks, 32>>>(
        x_norm.data_ptr<float>(), feature.data_ptr<float>(),
        beta.data_ptr<float>(), row_pointers.data_ptr<int>(),
        RowWindowOffsets.data_ptr<int>(),
        SparseAToX.data_ptr<int>(),
        BitMaskRowOffset.data_ptr<int>(),
        (uint16_t*)BitColMask.data_ptr<uint8_t>(),
        BitRowMask.data_ptr<uint8_t>(),
        attention.data_ptr<float>(),
        num_nodes
    );
    return attention;
}

at::Tensor SDDMM_TCGNN(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor blockPartition,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow,
    at::Tensor beta
) {
    int num_nodes = feature.size(0);
    auto x_norm = feature.norm(2, -1).clamp_min(1e-12);
    int blocks = (num_nodes + 15) / 16;
    int edge_num;
    cudaMemcpy(&edge_num, row_pointers.data_ptr<int>()+num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    auto attention = torch::empty({edge_num}, feature.options());
    SDDMM_TCGNN_kernel<<<blocks, 32>>>(
        x_norm.data_ptr<float>(), feature.data_ptr<float>(),
        beta.data_ptr<float>(), row_pointers.data_ptr<int>(),
        column_index.data_ptr<int>(),
        blockPartition.data_ptr<int>(),
        edgeToColumn.data_ptr<int>(),
        edgeToRow.data_ptr<int>(),
        num_nodes, edge_num,
        attention.data_ptr<float>()
    );
    return attention;
}

at::Tensor AGNN_divide(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta
) {
    int num_nodes = feature.size(0);
    auto x_norm = feature.norm(2, -1).clamp_min(1e-12);
    int blocks = (num_nodes + 15) / 16;
    int edge_num;
    cudaMemcpy(&edge_num, row_pointers.data_ptr<int>()+num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    auto attention = torch::empty({edge_num}, feature.options());
    auto next_layer = torch::empty({edge_num}, feature.options());
    SDDMM_kernel_16x8_32<<<blocks, 32>>>(
        x_norm.data_ptr<float>(), feature.data_ptr<float>(),
        beta.data_ptr<float>(), row_pointers.data_ptr<int>(),
        RowWindowOffsets.data_ptr<int>(),
        SparseAToX.data_ptr<int>(),
        BitMaskRowOffset.data_ptr<int>(),
        (uint16_t*)BitColMask.data_ptr<uint8_t>(),
        BitRowMask.data_ptr<uint8_t>(),
        attention.data_ptr<float>(),
        num_nodes
    );
    softmax<<<num_nodes, 32>>>(
        attention.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        next_layer.data_ptr<float>());
    auto output = torch::empty({num_nodes, 32}, feature.options());
    SpMM_kernel_16x8_32<<<blocks, 64>>>(
        next_layer.data_ptr<float>(), feature.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        RowWindowOffsets.data_ptr<int>(),
        SparseAToX.data_ptr<int>(),
        BitMaskRowOffset.data_ptr<int>(),
        (uint16_t*)BitColMask.data_ptr<uint8_t>(),
        BitRowMask.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        num_nodes
    );
    return output;
}