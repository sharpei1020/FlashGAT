#include <curand_kernel.h>
#include "gcn.cuh"
#include "assert.h"

#define TILESIZE_X 128
#define TILESIZE_Y 64
#define TILESIZE 32
#define BLOCKROWS 8
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__global__ void matmul(const float* __restrict__ Y, const float* __restrict__ X, float*  __restrict__ Z, int M, int K, int N) {
    const int BK = 16;
    const int K_rem = K % BK;
    const int more = K_rem > 0;
    const int BK_rem = K_rem % 4;
    bool flag = false;

    int tid = (threadIdx.y << 5) | (threadIdx.x & 31);

    if (blockIdx.x * TILESIZE_X >= M || blockIdx.y * TILESIZE_Y >= N) flag = true;
    __shared__ float tmp_a[TILESIZE_X][BK*2+1], tmp_b[TILESIZE_Y][BK*2+1];
    int tmp_a_addr = __cvta_generic_to_shared(tmp_a);
    int tmp_b_addr = __cvta_generic_to_shared(tmp_b);
    
    float tmp_c[32] = {0};

    #pragma unroll
    for (int i = 0; i < BK; i += 4) {
        if (flag) {
            if ((tid & 127) < (M - blockIdx.x * TILESIZE_X)) {
                int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + i + (tid >> 7)) * sizeof(float);
                int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + i + (tid >> 7);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr + 2 * (int)sizeof(float)), "l"(&X[in_a_cur_addr + 2]));    
            }
            int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + (i + (tid >> 6))) * sizeof(float);
            int in_b_cur_addr = (i + (tid >> 6)) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
        } else {
            int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + i + (tid >> 7)) * sizeof(float);
            int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + i + (tid >> 7);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr + 2 * (int)sizeof(float)), "l"(&X[in_a_cur_addr + 2]));
            int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + (i + (tid >> 6))) * sizeof(float);
            int in_b_cur_addr = (i + (tid >> 6)) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
        }
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);

    __syncthreads();

    for (int k = 1; k < K / BK + more; k++) {
        int j = 0;
        #pragma unroll
        for ( ; j < BK && (k * BK + j) < K - K_rem; j += 4) {
           if (flag) {
                if ((tid & 127) < (M - blockIdx.x * TILESIZE_X)) {
                    int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + j + (tid >> 7) + (k % 2) * BK) * sizeof(float);
                    int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + k * BK + j + (tid >> 7);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr + 2 * (int)sizeof(float)), "l"(&X[in_a_cur_addr + 2]));
                }
                int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + (j + (tid >> 6) + (k % 2) * BK)) * sizeof(float);
                int in_b_cur_addr = (k * BK + j + (tid >> 6)) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
            } else {
                int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + j + (tid >> 7) + (k % 2) * BK) * sizeof(float);
                int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + k * BK + j + (tid >> 7);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr + 2 * (int)sizeof(float)), "l"(&X[in_a_cur_addr + 2]));
                int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + (j + (tid >> 6) + (k % 2) * BK)) * sizeof(float);
                int in_b_cur_addr = (k * BK + j + (tid >> 6)) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
            }
        } 

        if (k == K / BK) {
            #pragma unroll
            for ( ; (k * BK + j) < K; j++) {
                if (flag) {
                    if ((tid & 127) < (M - blockIdx.x * TILESIZE_X)) {
                        int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + j + (k % 2) * BK) * sizeof(float);
                        int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + k * BK + j;
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
                    }
                    int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + j + (k % 2) * BK) * sizeof(float);
                    int in_b_cur_addr = (k * BK + j) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
                } else {
                    int tmp_a_cur_addr = tmp_a_addr + ((tid & 127) * (2 * BK + 1) + j + (k % 2) * BK) * sizeof(float);
                    int in_a_cur_addr = (blockIdx.x * TILESIZE_X + (tid & 127)) * K + k * BK + j;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_a_cur_addr), "l"(&X[in_a_cur_addr]));
                    int tmp_b_cur_addr = tmp_b_addr + ((tid & 63) * (2 * BK + 1) + j + (k % 2) * BK) * sizeof(float);
                    int in_b_cur_addr = (k * BK + j) * N + blockIdx.y * TILESIZE_Y + (tid & 63);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_b_cur_addr), "l"(&Y[in_b_cur_addr]));
                }
            }
            #pragma unroll
            for (int j = K_rem; j < BK; j++) {
                tmp_a[(tid & 127)][j + (k % 2) * BK] = 0.0f;
                tmp_b[(tid & 63)][j + (k % 2) * BK] = 0.0f;
            }        
        }
    
        for (j = ((k - 1) % 2) * BK; j < ((k - 1) % 2 + 1) * BK; j++){
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int l = 0; l < 8; l++)
                    tmp_c[i * 8 + l] += tmp_b[threadIdx.y + l * BLOCKROWS][j] * tmp_a[threadIdx.x + i * TILESIZE][j];
        } 

        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group 0;\n"::);

        __syncthreads();
    }

    int k = K / BK + more;
    #pragma unroll
    for (int j = ((k - 1) % 2) * BK; j < ((k - 1) % 2 + 1) * BK; j++){
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int l = 0; l < 8; l++)
                tmp_c[i * 8 + l] += tmp_b[threadIdx.y + l * BLOCKROWS][j] * tmp_a[threadIdx.x + i * TILESIZE][j];
    }
   if (flag) {
       #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (i * TILESIZE + threadIdx.x < (M - blockIdx.x * TILESIZE_X))
                     Z[(blockIdx.x * TILESIZE_X + threadIdx.x + i * TILESIZE) * N + blockIdx.y * TILESIZE_Y + threadIdx.y + j * BLOCKROWS] = tmp_c[i * 8 + j];
            }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Z[(blockIdx.x * TILESIZE_X + threadIdx.x + i * TILESIZE) * N + blockIdx.y * TILESIZE_Y + threadIdx.y + j * BLOCKROWS] = tmp_c[i * 8 + j];
    }
}

__global__ void spmm(
    float* __restrict__ in_f, 
    float* __restrict__ out_f, 
    const int* __restrict__ n_out_edge_index,
    const int* __restrict__ n_out_offsets,
    const int* __restrict__ n_out_counts, 
    const int* __restrict__ coo_dst,
    int rows, int weight_cols, int* thread_map) {

    if(blockIdx.x * TILESIZE * BLOCKROWS + threadIdx.x >= rows) return;
    int idx = thread_map[blockIdx.x * TILESIZE * BLOCKROWS + threadIdx.x];
    int start = n_out_offsets[idx - 1];
    int end = n_out_offsets[idx];
    int count = n_out_counts[idx];

    float feat[weight_cols];
    
    for (int i = start; i < end; i++) {
        int dst_node = coo_dst[n_out_edge_index[i]];
        float weight = n_out_counts[idx] * n_out_counts[dst_node] == 0 ? 0. : powf(n_out_counts[idx] * n_out_counts[dst_node], -0.5);
        for (int j = 0; j < weight_cols; j++)
            feat[j] += in_f[dst_node * weight_cols + j] * weight;
    }

    for (int i = 0; i < weight_cols ; i += 4)
        FLOAT4(out_f[idx * weight_cols + i]) = FLOAT4(feat[i]);
}


void linear(const float* lin_weight, const float* feature, float* out_feature,
        int rows, int cols, int weight_rows, int weight_cols) {

        int x_grid_num = (rows + TILESIZE_X - 1) / TILESIZE_X;
        int y_grid_num = (weight_cols + TILESIZE_Y - 1) / TILESIZE_Y;
        dim3 BlockDim(TILESIZE, BLOCKROWS, 1);
        dim3 GridDim(x_grid_num, y_grid_num, 1);

        assert(rows == cols);
        matmul<<<GridDim, BlockDim>>>(lin_weight, feature, out_feature, rows, weight_rows, weight_cols);
}

void optimizer(
    int* thread_map, 
    const int* n_out_edge_index, 
    const int* n_out_offsets, 
    const int* n_out_counts, 
    const int* coo_src,
    const int* coo_dst,
    int rows
) {
    
}

void gather(
    float* in_f, 
    float* out_f, 
    const int* n_out_edge_index,
    const int* n_out_offsets, 
    const int* n_out_counts,
    const int* coo_dst,
    int rows,
    int weight_cols,
    int* thread_map
) {
    int grid_num = (rows + TILESIZE * BLOCKROWS - 1) / (TILESIZE * BLOCKROWS);
    dim3 BlockDim(TILESIZE * BLOCKROWS, 1, 1);
    dim3 GridDim(grid_num, 1, 1);
    spmm<<<GridDim, BlockDim>>>(in_f, out_f, n_out_edge_index, n_out_offsets, n_out_counts, coo_dst, rows, weight_cols, thread_map);
}


