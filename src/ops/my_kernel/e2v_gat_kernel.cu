#include <cuda_runtime.h>
#include <cstdint>
#include "e2v_gat_kernel.cuh"

#define FULL_MASK 0xffffffff

template <int head_dim, int log2_head_dim, int block_tile, int log2_block_tile>
__global__ void e2v_gat(
    int* edge_id,
    int* row_offset,
    float* x,
    float* k_feature,
    float* v_feature,
    float* output,
    int node_num,
    int edge_num
) {
    int block_start = row_offset[blockIdx.x];
    int block_end = row_offset[blockIdx.x + 1];
    if (block_start == block_end) return;
    int lane_id = threadIdx.x & (block_tile-1);
    int warp_id = threadIdx.x >> log2_block_tile;
    const float scale = 0.3535533906f; // 1/sqrt(8)
    int read_rows = 4*block_tile/head_dim; 
    int read_times = head_dim/4;

    __shared__ float KFeat[2][block_tile][8*head_dim];
    __shared__ float VFeat[2][block_tile][8*head_dim];
    __shared__ int edge_idx[2][block_tile];
    __shared__ float softmax[2][2][block_tile][8];

    float D[head_dim], C[head_dim] = {0.f};
    int i = block_start;
    int cur_addr;
    for (int j = 0; j < read_times; j++) {
        cur_addr = __cvta_generic_to_shared(&KFeat[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows][(threadIdx.x&(2*head_dim-1))*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[min(blockIdx.x*block_tile+j*read_rows+(threadIdx.x>>(1+log2_head_dim)), node_num-1)*8*head_dim+(threadIdx.x&(2*head_dim-1))*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (int j = 0; j < read_times; j++) {
        FLOAT4(D[j*4]) = FLOAT4(KFeat[(i+1)&1][lane_id][warp_id*head_dim+j*4]);
        edge_idx[i&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows] = edge_id[block_start*block_tile+(threadIdx.x>>(1+log2_head_dim))+j*read_rows];
        cur_addr = __cvta_generic_to_shared(&KFeat[i&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows][(threadIdx.x&(2*head_dim-1))*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&k_feature[min(edge_idx[i&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows], edge_num-1)*8*head_dim+(threadIdx.x&(2*head_dim-1))*4]));
        cur_addr = __cvta_generic_to_shared(&VFeat[i&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows][(threadIdx.x&(2*head_dim-1))*4]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&v_feature[min(edge_idx[i&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows], edge_num-1)*8*head_dim+(threadIdx.x&(2*head_dim-1))*4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    softmax[i&1][0][threadIdx.x>>3][threadIdx.x&7] = std::numeric_limits<float>::lowest();
    softmax[i&1][1][threadIdx.x>>3][threadIdx.x&7] = 0.f;
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (; i < block_end - 1; i++) {
        for (int j = 0; j < read_times; j++) {
            edge_idx[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows] = edge_id[(i+1)*block_tile+(threadIdx.x>>(1+log2_head_dim))+j*read_rows];
            cur_addr = __cvta_generic_to_shared(&KFeat[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows][(threadIdx.x&(2*head_dim-1))*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&k_feature[min(edge_idx[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows], edge_num-1)*8*head_dim+(threadIdx.x&(2*head_dim-1))*4]));
            cur_addr = __cvta_generic_to_shared(&VFeat[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows][(threadIdx.x&(2*head_dim-1))*4]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&v_feature[min(edge_idx[(i+1)&1][(threadIdx.x>>(1+log2_head_dim))+j*read_rows], edge_num-1)*8*head_dim+(threadIdx.x&(2*head_dim-1))*4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        bool mask = edge_idx[i&1][lane_id] < edge_num;
        float alpha = 0.f;
        for (int j = 0; j < head_dim; j++)
            alpha += D[j] * KFeat[i&1][lane_id][warp_id*head_dim+j];
        alpha *= scale;
        alpha = mask ? alpha : std::numeric_limits<float>::lowest();
        float alpha_max = max(softmax[i&1][0][lane_id][warp_id], alpha);
        softmax[(i+1)&1][0][lane_id][warp_id] = alpha_max;
        alpha = float(mask) * __expf(alpha - alpha_max);
        softmax[(i+1)&1][1][lane_id][warp_id] = softmax[i&1][1][lane_id][warp_id] * 
                                    __expf(softmax[i&1][0][lane_id][warp_id] - softmax[(i+1)&1][0][lane_id][warp_id]);
        softmax[(i+1)&1][1][lane_id][warp_id] += alpha;
        float update = 1.f/(softmax[(i+1)&1][1][lane_id][warp_id]+1e-16f);
        alpha *= update;
        update *= __expf(softmax[i&1][0][lane_id][warp_id]-softmax[(i+1)&1][0][lane_id][warp_id]) 
                *(softmax[i&1][1][lane_id][warp_id]+1e-16f); 
        for (int j = 0; j < head_dim; j++) {
            C[j] *= update;
            C[j] += alpha*VFeat[i&1][lane_id][warp_id*head_dim+j];
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    i = block_end - 1;
    bool mask = edge_idx[i&1][lane_id] < edge_num;
    float alpha = 0.f;
    for (int j = 0; j < head_dim; j++)
        alpha += D[j] * KFeat[i&1][lane_id][warp_id*head_dim+j];
    alpha *= scale;
    alpha = mask ? alpha : std::numeric_limits<float>::lowest();
    float alpha_max = max(softmax[i&1][0][lane_id][warp_id], alpha);
    softmax[(i+1)&1][0][lane_id][warp_id] = alpha_max;
    alpha = float(mask) * __expf(alpha - alpha_max);
    softmax[(i+1)&1][1][lane_id][warp_id] = softmax[i&1][1][lane_id][warp_id] * 
                                __expf(softmax[i&1][0][lane_id][warp_id] - softmax[(i+1)&1][0][lane_id][warp_id]);
    softmax[(i+1)&1][1][lane_id][warp_id] += alpha;
    float update = 1.f/(softmax[(i+1)&1][1][lane_id][warp_id]+1e-16f);
    alpha *= update;
    update *= __expf(softmax[i&1][0][lane_id][warp_id]-softmax[(i+1)&1][0][lane_id][warp_id]) 
            *(softmax[i&1][1][lane_id][warp_id]+1e-16f); 
    for (int j = 0; j < head_dim; j++) {
        C[j] *= update;
        C[j] += alpha*VFeat[i&1][lane_id][warp_id*head_dim+j];
    }
    if (blockIdx.x*block_tile+lane_id < node_num)
        for (int j = 0; j < read_times; j++) 
            FLOAT4(output[(blockIdx.x*block_tile+lane_id)*8*head_dim+warp_id*head_dim+j*4]) = FLOAT4(C[j*4]);
}

at::Tensor E2V_GAT(
    at::Tensor x,
    at::Tensor k_feature,
    at::Tensor v_feature,
    at::Tensor edge_id,
    at::Tensor row_offset,
    int num_head,
    int block_size
) {
    int num_node = x.size(0);
    int num_edge = k_feature.size(0);
    int feature_dim = x.size(1);
    auto output = at::empty({num_node, feature_dim}, x.options());

    int blocks = (num_node + block_size - 1) / block_size;
    int mode = (feature_dim / 8) * 100 + block_size;
    switch(mode) {
        case 832:
            e2v_gat<8, 3, 32, 5><<<blocks, 256>>>(
                edge_id.data<int>(),
                row_offset.data<int>(),
                x.data<float>(),
                k_feature.data<float>(),
                v_feature.data<float>(),
                output.data<float>(),
                num_node, num_edge);
            break;
        case 816:
            e2v_gat<8, 3, 16, 4><<<blocks, 128>>>(
                edge_id.data<int>(),
                row_offset.data<int>(),
                x.data<float>(),
                k_feature.data<float>(),
                v_feature.data<float>(),
                output.data<float>(),
                num_node, num_edge);
            break;
        default:
        printf("Unsupported mode: %d\n", mode);
        exit(1);
    }   
    return output;

}