#include "gcn.cuh"
#include "gat.cuh"
#include "assert.h"

#define TILESIZE_X 128
#define TILESIZE 16
#define BLOCKROWS 8
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


// __global__ void linear(const float* __restrict__ weight, const float* __restrict__ bias, const float* __restrict__ X, float*  __restrict__ Z, int M, int K, bool bias_flag) {
//     const int BK = 4;
//     const int K_rem = K % BK;
//     const int more = K_rem > 0;
//     // bool flag = false;

//     // if (blockIdx.x * TILESIZE_X >= M) flag = true;
//     __shared__ float tmp_x[TILESIZE_X*(2* BK+1)], tmp_y[TILESIZE_Y*(2*BK+1)];
//     int tmp_x_addr = __cvta_generic_to_shared(tmp_x);
//     int tmp_y_addr = __cvta_generic_to_shared(tmp_y);
    
//     float tmp_c[32];
//     for (int i = 0; i < 8; i++)
//         for (int j = 0; j < 4; j++)
//             tmp_c[i * 4 + j] = bias_flag ? bias[j*16+(threadIdx.x & 15)] : 0.f;

//     int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK + 1)) * sizeof(float);
//     int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K;
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));   
//     int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK + 1)) * sizeof(float);
//     int in_y_cur_addr = (threadIdx.x >> 2) * K;
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
    
//     asm volatile("cp.async.commit_group;\n"::);
//     asm volatile("cp.async.wait_group 0;\n"::);

//     __syncthreads();

//     for (int k = 1; k < K / BK + more; k++) {
//         if(k < (K / BK)) {
//             int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK + 1) + (k % 2) * BK) * sizeof(float);
//             int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K + k * BK;
//             asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));   
//             int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK + 1) + (k % 2) * BK) * sizeof(float);
//             int in_y_cur_addr = (threadIdx.x >> 2) * K + k * BK;
//             asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
//         } else {
//             for (int i = 0; i < 4; i++) {
//                 if (k * BK + i < K) {
//                     int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK + 1) + (k % 2) * BK + i) * sizeof(float);
//                     int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K + k * BK + i;
//                     asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));
//                     int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK + 1) + (k % 2) * BK + i) * sizeof(float);
//                     int in_y_cur_addr = (threadIdx.x >> 2) * K + k * BK + i;
//                     asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
//                 } else {
//                     tmp_x[(threadIdx.x >> 1) * (2 * BK + 1) + (k % 2) * BK + i] = 0.f;
//                     tmp_y[(threadIdx.x >> 2) * (2 * BK + 1) + (k % 2) * BK + i] = 0.f;
//                 }
//             }
//         }
//         #pragma unroll
//         for (int i = 0; i < 8; i++)
//             #pragma unroll
//             for (int j = 0; j < 4; j++) {
//                 float4 x_ = FLOAT4(tmp_x[(i*16+(threadIdx.x >> 4))*(2*BK+1) + ((k - 1) % 2)*BK]);
//                 float4 w_ = FLOAT4(tmp_y[(j*16+(threadIdx.x & 15))*(2*BK+1) + ((k - 1) % 2)*BK]);
//                 tmp_c[i * 4 + j] += (x_.x * w_.x + x_.y * w_.y + x_.z * w_.z + x_.w * w_.w);
//             }

//         asm volatile("cp.async.commit_group;\n"::);
//         asm volatile("cp.async.wait_group 0;\n"::);

//         __syncthreads();
//     }

//     int k = K / BK + more;
//     #pragma unroll
//     for (int i = 0; i < 8; i++)
//         #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             float4 x_ = FLOAT4(tmp_x[(i*16+(threadIdx.x >> 4))*(2*BK+1) + ((k - 1) % 2)*BK]);
//             float4 w_ = FLOAT4(tmp_y[(j*16+(threadIdx.x & 15))*(2*BK+1) + ((k - 1) % 2)*BK]);
//             tmp_c[i * 4 + j] += (x_.x * w_.x + x_.y * w_.y + x_.z * w_.z + x_.w * w_.w);
//         }

//     #pragma unroll
//     for (int i = 0; i < 8; i++)
//         #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             Z[min((blockIdx.x * TILESIZE_X + (threadIdx.x >> 4) + i * 16), M-1) * TILESIZE_Y + j * 16 + (threadIdx.x & 15)] = tmp_c[i * 4 + j];
//         }
// }

__global__ void node_weight(
    int *edge_index_0,
    int *edge_index_1,
    int *count,
    int *out,

    float* x_weight,
    float* result,
    int total_size
) {
    int idx = blockIdx.x * TILESIZE + (threadIdx.x >> 4);
    if(idx >= total_size) return;
    int start = count[idx];
    int end = count[idx + 1];
    int dst_idx = edge_index_1[out[start]];
    bool flag = false;
    for (int i = start; i < end; i++) {
        if (edge_index_0[out[i]] == dst_idx) 
            flag = true;
    }
    float weight = flag ? __powf(end - start, -0.5) : __powf(end - start + 1, -0.5);
    x_weight[dst_idx] = weight;
    float4 x = FLOAT4(result[dst_idx * 64 + (threadIdx.x & 15) * 4]);
    x.x = weight * x.x * weight;
    x.y = weight * x.y * weight;
    x.z = weight * x.z * weight;
    x.w = weight * x.w * weight;
    FLOAT4(result[dst_idx * 64 + (threadIdx.x & 15) * 4]) = x;
}

__global__ void gcn(
    int *edge_index_0,
    int *edge_index_1,
    int *count,
    int *out,
    float* x_weight,
    float*  x, 
    float* result, 
    int total_size) {

    int idx = blockIdx.x * TILESIZE + (threadIdx.x >> 4);
    if(idx >= total_size) return;
    int start = count[idx];
    int end = count[idx + 1];
    int dst_idx = edge_index_1[out[start]];
    float weight = x_weight[dst_idx];
    
    float feat[4];
    FLOAT4(feat) = FLOAT4(result[dst_idx * 64 + (threadIdx.x & 15) * 4]);
    for (int i = start; i < end; i++) {
        int src_idx = edge_index_0[out[i]];
        if (src_idx == dst_idx) continue;
        float4 x_ = FLOAT4(x[src_idx * 64 + (threadIdx.x & 15) * 4]);
        float weight_ = x_weight[src_idx];
        feat[0] += (weight * weight_ * x_.x);
        feat[1] += (weight * weight_ * x_.y);
        feat[2] += (weight * weight_ * x_.z);
        feat[3] += (weight * weight_ * x_.w);
    }
    FLOAT4(result[dst_idx * 64 + (threadIdx.x & 15) * 4]) = FLOAT4(feat);
    
}


at::Tensor GCN(
    torch::Tensor lin_weight, 
    torch::Tensor lin_bias,
    torch::Tensor edge_index_0,
    torch::Tensor edge_index_1,
    torch::Tensor count,
    torch::Tensor out,
    torch::Tensor x,
    int out_feats) {
    
    auto x_size = x.sizes().vec();
    int total_size = count.sizes().vec()[0] - 1;
    auto feats = x.new_zeros({x_size[0], out_feats}, torch::kFloat32);
    auto x_weight = x.new_ones({x_size[0]}, torch::kFloat32);
    auto result =  x.new_zeros({x_size[0], out_feats}, torch::kFloat32);
    
    int x_grid_num = (x_size[0] + TILESIZE_X - 1) / TILESIZE_X;
    dim3 BlockDim(256, 1, 1);
    dim3 GridDim(x_grid_num, 1, 1);

    linear<<<GridDim, BlockDim>>>(
        lin_weight.data_ptr<float>(), 
        lin_bias.data_ptr<float>(), 
        x.data_ptr<float>(), 
        feats.data_ptr<float>(), 
        x_size[0], 
        x_size[1], 
        false);

    result.copy_(feats);

    // BlockDim = dim3(1024, 1, 1);
    GridDim = dim3((total_size + 15) / 16, 1, 1);
    node_weight<<<GridDim, BlockDim>>>(
        edge_index_0.data_ptr<int>(),
        edge_index_1.data_ptr<int>(),
        count.data_ptr<int>(),
        out.data_ptr<int>(),
        x_weight.data_ptr<float>(),
        result.data_ptr<float>(),
        total_size);

    gcn<<<GridDim, BlockDim>>>(
        edge_index_0.data_ptr<int>(),
        edge_index_1.data_ptr<int>(),
        count.data_ptr<int>(),
        out.data_ptr<int>(),
        x_weight.data_ptr<float>(),
        feats.data_ptr<float>(),
        result.data_ptr<float>(),
        total_size);

    return result;
}



