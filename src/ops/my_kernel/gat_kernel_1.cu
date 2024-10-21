#include "gat.cuh"
#include "assert.h"
#include "time.cuh"

#define TILESIZE_X 128
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__global__ void linear(
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ X,
    const float* __restrict__ att_i,
    const float* __restrict__ att_j, 
    float*  __restrict__ Z, 
    float*  __restrict__ alpha_i,
    float*  __restrict__ alpha_j,
    int M, 
    int K, 
    bool bias_flag) {
    const int BK = 4;
    const int K_rem = K % BK;
    const int more = (K_rem > 0) ? 1 : 0;
    // bool flag = false;

    // if (blockIdx.x * TILESIZE_X >= M) flag = true;
    __shared__ float tmp_x[TILESIZE_X*(2*BK)], tmp_y[TILESIZE_Y*(2*BK)], param_att_i[64], param_att_j[64];
    int tmp_x_addr = __cvta_generic_to_shared(tmp_x);
    int tmp_y_addr = __cvta_generic_to_shared(tmp_y);
    
    float tmp_c[32], alpha1[8], alpha2[8];
    for (int i = 0; i < 8; i++) {
        alpha1[i] = 0.f;
        alpha2[i] = 0.f;
        for (int j = 0; j < 4; j++)
            tmp_c[i * 4 + j] = bias_flag ? bias[j*16+(threadIdx.x & 15)] : 0.f;
    }

    int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1)*(2 * BK)) * sizeof(float);
    int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));   
    int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2)*(2 * BK)) * sizeof(float);
    int in_y_cur_addr = (threadIdx.x >> 2) * K;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
    int cur_addr = __cvta_generic_to_shared(param_att_i) + (threadIdx.x >> 2) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(att_i));
    cur_addr = __cvta_generic_to_shared(param_att_j) + (threadIdx.x >> 2) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(att_j));
    
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);

    __syncthreads();

    for (int k = 1; k < ((K / BK) + more); k++) {
        if(k < (K / BK)) {
            int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK) + (k % 2) * BK) * sizeof(float);
            int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K + k * BK;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));   
            int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK) + (k % 2) * BK) * sizeof(float);
            int in_y_cur_addr = (threadIdx.x >> 2) * K + k * BK;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
        } else {
            for (int i = 0; i < 4; i++) {
                if ((k * BK + i) < K) {
                    int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK) + (k % 2) * BK + i) * sizeof(float);
                    int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K + k * BK + i;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));
                    int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK) + (k % 2) * BK + i) * sizeof(float);
                    int in_y_cur_addr = (threadIdx.x >> 2) * K + k * BK + i;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
                } else {
                    tmp_x[(threadIdx.x >> 1) * (2 * BK) + (k % 2) * BK + i] = 0.f;
                    tmp_y[(threadIdx.x >> 2) * (2 * BK) + (k % 2) * BK + i] = 0.f;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float4 x_ = FLOAT4(tmp_x[(i*16+(threadIdx.x >> 4))*(2*BK) + ((k - 1) % 2)*BK]);
                float4 w_ = FLOAT4(tmp_y[(j*16+(threadIdx.x & 15))*(2*BK) + ((k - 1) % 2)*BK]);
                tmp_c[i * 4 + j] += (x_.x * w_.x + x_.y * w_.y + x_.z * w_.z + x_.w * w_.w);
            }

        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group 0;\n"::);

        __syncthreads();
    }

    int k = K / BK + more;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float4 x_ = FLOAT4(tmp_x[(i*16+(threadIdx.x >> 4))*(2*BK) + ((k - 1) % 2)*BK]);
            float4 w_ = FLOAT4(tmp_y[(j*16+(threadIdx.x & 15))*(2*BK) + ((k - 1) % 2)*BK]);
            tmp_c[i * 4 + j] += (x_.x * w_.x + x_.y * w_.y + x_.z * w_.z + x_.w * w_.w);
        }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Z[min((blockIdx.x * TILESIZE_X + (threadIdx.x >> 4) + i * 16), M-1) * TILESIZE_Y + j * 16 + (threadIdx.x & 15)] = tmp_c[i * 4 + j];
            alpha1[i] += tmp_c[i * 4 + j] * param_att_i[j * 16 + (threadIdx.x & 15)];
            alpha2[i] += tmp_c[i * 4 + j] * param_att_j[j * 16 + (threadIdx.x & 15)];
        }
        int iter = 16;
        while (iter > 1) {
            alpha1[i] += __shfl_xor_sync(FULL_MASK, alpha1[i], iter-1);
            alpha2[i] += __shfl_xor_sync(FULL_MASK, alpha2[i], iter-1);
            iter /= 2;
        }
        alpha_i[min((blockIdx.x * TILESIZE_X + (threadIdx.x >> 4) + i * 16), M-1)] = alpha1[i];
        alpha_j[min((blockIdx.x * TILESIZE_X + (threadIdx.x >> 4) + i * 16), M-1)] = alpha2[i];
    }
}

// __global__ void gat(
//     int* edge_index_0,
//     int* edge_index_1,
//     int* count,
//     int* out,
//     float* x,
//     float* output,
//     float* alpha_i,
//     float* alpha_j,

//     int num_heads,
//     int out_feats,
//     int total_size,
//     int edge_len
// ) {
//     extern __shared__ float param[];
//     int n = out_feats / 8;
//     float* max_alpha = param + (blockDim.x / n);
//     float* last_max_alpha = param + (blockDim.x / n);
//     float* alpha_sum = last_max_alpha + (blockDim.x / n);
//     int* edge_off = (int*) (alpha_sum + (blockDim.x / n));
//     float* feat = alpha_sum + 2 * (blockDim.x / n);

//     float temp[8], alpha_init = 0.f;

//     int id = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (num_heads * n));
//     int edge_begin = count[min(id, total_size-1)];
//     max_alpha[(threadIdx.x / n)] = std::numeric_limits<float>::lowest();
//     last_max_alpha[(threadIdx.x / n)] = 0.f;
//     alpha_sum[(threadIdx.x / n)] = 0.f;

//     alpha_init += alpha_i[edge_index_1[out[min(id, total_size-1)]]];

//     edge_off[threadIdx.x / n] = count[min(id+1, total_size)];
//     int start = __shfl_sync(FULL_MASK, edge_begin, 0);
//     int start_ = start + ((threadIdx.x & 31) / (num_heads * n));
//     int end = edge_off[(threadIdx.x & (blockDim.x - 32)) / n + (32 / n) - 1];
//     int warp_node_num = 256 / (num_heads * out_feats);
//     int end_ = ((end - start) + (warp_node_num - 1)) / warp_node_num * warp_node_num + start;
//     u_int filter = (1 << (threadIdx.x & (n * num_heads - 1)));
//     for (int j = 1; j < (32 / (n * num_heads)); j++) {
//         filter |= (filter << (n * num_heads));
//     }
//     for (int i = 0; i < 8; i++)
//         feat[threadIdx.x * 9 + i] = 0.f;
//     for (int i = start_; i < end_; i += warp_node_num) {
//         int idx = min(i, end-1);
//         int map = out[idx];
//         int block_delta = (threadIdx.x & (blockDim.x - 32)) / n, delta = ((threadIdx.x/n) & (num_heads - 1));
//         for (int j = delta; j < (32/n); j += num_heads) {
//             delta += ((idx >= edge_off[block_delta + j]) ? 1 : 0) * num_heads;
//         }
//         block_delta += delta;
//         float alpha = alpha_init + alpha_j[edge_index_0[map]];
//         for (int j = 0; j < 2; j++) {
//             FLOAT4(temp[j * 4]) = FLOAT4(x[edge_index_0[map] * num_heads * out_feats + j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
//         }
//         alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
//         int iter = n;
//         while (iter > 1) {
//             alpha += __shfl_xor_sync(FULL_MASK, alpha, iter-1);
//             iter /= 2;
//         }
//         u_int agg_mask = (__match_any_sync(FULL_MASK, delta) & filter);
//         int num = __popc(agg_mask), offset = (__ffs(agg_mask) - 1) / (num_heads * n);
//         iter = num;
//         int iter_offset = offset;
//         float max_ = max(max_alpha[block_delta], alpha);
//         while (iter > 1) {
//             max_ = max(__shfl_sync(agg_mask, max_, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1))), max_);
//             iter = (iter + 1) >> 1;
//             iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//         }
//         float sum_ = alpha_sum[block_delta] * __expf(last_max_alpha[block_delta] - max_);
//         float agg_sum = (i > end) ? 0.f : __expf(alpha - max_);
//         iter = num;
//         iter_offset = offset;
//         while(iter > 1) {
//             agg_sum += ((2 * iter_offset + iter - 1) == (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) << 1) ? 0.f : 1.f) * __shfl_sync(agg_mask, agg_sum, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1)));
//             iter = (iter + 1) >> 1;
//             iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//         }
//         sum_ += agg_sum;
//         float s = ((sum_ > 0.f) ? (__expf(last_max_alpha[block_delta] - max_) * (alpha_sum[block_delta] + 1e-16f) / (sum_ + 1e-16f)) : 0.f);
//         float w = (i >= end) ? 0.f : __expf(alpha - max_) / (sum_ + 1e-16f);
//         for (int j = 0; j < 8; j++) {
//             float tmp = s * feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 9 + j];
//             float agg_tmp = w * temp[j];
//             iter = num;
//             iter_offset = offset;
//             while(iter > 1) {
//                 agg_tmp += ((2 * iter_offset + iter - 1) == (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) << 1) ? 0.f : 1.f) * __shfl_sync(agg_mask, agg_tmp, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1)));
//                 iter = (iter + 1) >> 1;
//                 iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//             }
//             feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 9 + j] = agg_tmp + tmp;
//         }
//         max_alpha[block_delta] = max_;
//         last_max_alpha[block_delta] = max_;
//         alpha_sum[block_delta] = sum_;
//         __syncwarp();
//     }
//     if (id < total_size) {
//         debug(feat + threadIdx.x * 9, output + edge_index_1[out[edge_begin]] * num_heads * out_feats + ((threadIdx.x / n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8, 8);
//     }
// }

__global__ void gat(
    int* edge_index_0,
    int* edge_index_1,
    int* count,
    int* out,
    float* x,
    float* output,
    float* alpha_i,
    float* alpha_j,

    int num_heads,
    int out_feats,
    int total_size,
    int edge_len
) {
    extern __shared__ float param[];
    int n = out_feats / 8;
    float* max_alpha = param;
    float* last_max_alpha = param + (blockDim.x / n);
    float* alpha_sum = last_max_alpha + (blockDim.x / n);
    int* edge_off = (int*) (alpha_sum + (blockDim.x / n));
    float* feat = alpha_sum + 2 * (blockDim.x / n);

    float temp[8], alpha_init = 0.f;

    int id = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (num_heads * n));
    int edge_begin = count[min(id, total_size-1)];
    int map1 = edge_index_1[out[edge_begin]];
    max_alpha[(threadIdx.x / n)] = std::numeric_limits<float>::lowest();
    last_max_alpha[(threadIdx.x / n)] = 0.f;
    alpha_sum[(threadIdx.x / n)] = 0.f;

    alpha_init += alpha_i[map1];

    edge_off[threadIdx.x / n] = count[min(id+1, total_size)];
    int start = __shfl_sync(FULL_MASK, edge_begin, 0);
    int start_ = start + ((threadIdx.x & 31) / (num_heads * n));
    int end = edge_off[(threadIdx.x & (blockDim.x - 32)) / n + (32 / n) - 1];
    int warp_node_num = 256 / (num_heads * out_feats);
    int end_ = ((end - start) + (warp_node_num - 1)) / warp_node_num * warp_node_num + start;
    u_int filter = (1 << (threadIdx.x & (n * num_heads - 1)));
    for (int j = 1; j < (32 / (n * num_heads)); j++) {
        filter |= (filter << (n * num_heads));
    }
    for (int i = 0; i < 8; i++)
        feat[threadIdx.x * 9 + i] = 0.f;
    for (int i = start_; i < end_; i += warp_node_num) {
        int idx = min(i, end-1);
        int map0 = edge_index_0[out[idx]];
        int block_delta = (threadIdx.x & (blockDim.x - 32)) / n, delta = ((threadIdx.x/n) & (num_heads - 1));
        for (int j = delta; j < (32/n); j += num_heads) {
            delta += ((idx >= edge_off[block_delta + j]) ? 1 : 0) * num_heads;
        }
        block_delta += delta;
        float alpha = alpha_init + alpha_j[map0];
        for (int j = 0; j < 2; j++) {
            FLOAT4(temp[j * 4]) = FLOAT4(x[map0 * num_heads * out_feats + j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
        }
        alpha = alpha - 0.99f * min(0.0f, alpha);
        int iter = n;
        while (iter > 1) {
            alpha += __shfl_xor_sync(FULL_MASK, alpha, iter-1);
            iter /= 2;
        }
        u_int agg_mask = (__match_any_sync(FULL_MASK, delta) & filter);
        int num = __popc(agg_mask), offset = (__ffs(agg_mask) - 1) / (num_heads * n);
        iter = num;
        int iter_offset = offset;
        float max_ = max(max_alpha[block_delta], alpha);
        while (iter > 1) {
            max_ = max(__shfl_sync(agg_mask, max_, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1))), max_);
            iter = (iter + 1) >> 1;
            iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
        }
        float sum_ = alpha_sum[block_delta] * __expf(last_max_alpha[block_delta] - max_);
        float agg_sum = (i > end) ? 0.f : __expf(alpha - max_);
        iter = num;
        iter_offset = offset;
        while(iter > 1) {
            agg_sum += ((2 * iter_offset + iter - 1) == (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) << 1) ? 0.f : 1.f) * __shfl_sync(agg_mask, agg_sum, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1)));
            iter = (iter + 1) >> 1;
            iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
        }
        sum_ += agg_sum;
        float s = ((sum_ > 0.f) ? (__expf(last_max_alpha[block_delta] - max_) * (alpha_sum[block_delta] + 1e-16f) / (sum_ + 1e-16f)) : 0.f);
        float w = (i >= end) ? 0.f : __expf(alpha - max_) / (sum_ + 1e-16f);
        for (int j = 0; j < 8; j++) {
            float tmp = s * feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 9 + j];
            float agg_tmp = w * temp[j];
            iter = num;
            iter_offset = offset;
            while(iter > 1) {
                agg_tmp += ((2 * iter_offset + iter - 1) == (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) << 1) ? 0.f : 1.f) * __shfl_sync(agg_mask, agg_tmp, (2 * iter_offset + iter - 1 - ((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1))) * num_heads * n + (threadIdx.x & (num_heads * n - 1)));
                iter = (iter + 1) >> 1;
                iter_offset = (((threadIdx.x / (n * num_heads)) & (32 / (n * num_heads) - 1)) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
            }
            feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 9 + j] = agg_tmp + tmp;
        }
        max_alpha[block_delta] = max_;
        last_max_alpha[block_delta] = max_;
        alpha_sum[block_delta] = sum_;
        __syncwarp();
    }
    if (id < total_size) {
        debug(feat + threadIdx.x * 9, output + map1 * num_heads * out_feats + ((threadIdx.x / n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8, 8);
    }
}

at::Tensor GAT(
    torch::Tensor x, 
    torch::Tensor edge_index_0,
    torch::Tensor edge_index_1,
    torch::Tensor counts,
    torch::Tensor out,
    torch::Tensor lin_weight,
    torch::Tensor lin_bias,
    torch::Tensor att_i,
    torch::Tensor att_j,
    int num_heads,
    int out_feats
) {
    auto size = x.sizes().vec();
    int edge_len = edge_index_0.sizes().vec()[0];
    float* feats, * alpha_i, * alpha_j;
    cudaMalloc(&feats, size[0] * out_feats * sizeof(float));
    cudaMalloc(&alpha_i, size[0] * sizeof(float));
    cudaMalloc(&alpha_j, size[0] * sizeof(float));
    // auto feats = torch::empty({size[0], out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    // auto alpha_i = torch::empty({size[0]}, device(torch::kCUDA)).to(torch::kFloat32);
    // auto alpha_j = torch::empty({size[0]}, device(torch::kCUDA)).to(torch::kFloat32);
    auto output = torch::empty({size[0], num_heads * out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    int grid_num = (size[0] + TILESIZE_X - 1) / TILESIZE_X;
    dim3 BlockDim(256, 1, 1);
    dim3 GridDim(grid_num, 1, 1);

    linear<<<GridDim, BlockDim>>>(
        lin_weight.data_ptr<float>(), 
        lin_bias.data_ptr<float>(), 
        x.data_ptr<float>(), 
        att_i.data_ptr<float>(), 
        att_j.data_ptr<float>(), 
        feats, 
        alpha_i, 
        alpha_j, 
        size[0], 
        size[1], 
        false);

    auto total_size = counts.sizes().vec();
    int block_node_num = 128 * 8 / (num_heads * out_feats);
    grid_num = (total_size[0] + block_node_num - 2) / block_node_num;
    GridDim = dim3(grid_num, 1, 1);
    BlockDim = dim3(128, 1, 1);

    cudaFuncSetAttribute(gat, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    gat<<<GridDim, BlockDim, (128 * 9 + 4 * block_node_num * num_heads) * sizeof(float)>>>(
        edge_index_0.data_ptr<int>(), 
        edge_index_1.data_ptr<int>(),
        counts.data_ptr<int>(),
        out.data_ptr<int>(),
        feats,
        output.data_ptr<float>(),
        
        alpha_i,
        alpha_j,
        num_heads,
        out_feats,
        total_size[0]-1,
        edge_len);

    return output;
}