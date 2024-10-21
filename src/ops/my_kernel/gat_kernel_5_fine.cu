#include "gat.cuh"
#include "assert.h"
#include "time.cuh"

#define TILESIZE_X 64
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__global__ void linear_reduce(
    const float* X,
    const float* weight,
    const float* att_i_,
    const int* node_map,
    float* feat,
    float* alpha,
    int node_len,
    int K
) {
    const int BK = 16;
    __shared__ float a_tmp[TILESIZE_X * (2 * BK + 1)], b_tmp[TILESIZE_Y * (2 * BK + 1)], att_i[64];
    float c_tmp[8], alphai = 0.f;
    for (int i = 0; i < 8; i++) {
        c_tmp[i] = 0.f;
    }
    int block_offset = blockIdx.x * TILESIZE_X;
    for (int i = 0; i < (BK / 8); i++) {
        int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1))) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[node_map[min(node_len - 1, block_offset + (threadIdx.x >> 5) + i * 16)] * K + (threadIdx.x & 31)]));
        cur_addr = __cvta_generic_to_shared(b_tmp) + ((threadIdx.x & 63) * (2 * BK + 1) + (threadIdx.x >> 6) + i * 8) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[((threadIdx.x >> 6) + i * 8) * 64 + (threadIdx.x & 63)]));
    }

    int cur_addr = __cvta_generic_to_shared(att_i) + (threadIdx.x & 63) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&att_i_[threadIdx.x & 63]));

    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    for (int k = 1; k < (((K - 1) >> 5) + 1); k++) {
        for (int i = 0; i < (BK / 8); i++) {
            int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1)) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[node_map[min(node_len - 1, block_offset + (threadIdx.x >> 5) + i * 16)] * K + min(k * BK + (threadIdx.x & (BK - 1)), K - 1)]));
            cur_addr = __cvta_generic_to_shared(b_tmp) + ((threadIdx.x & 63) * (2 * BK + 1) + (threadIdx.x >> 6) + i * 8 + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[min((threadIdx.x >> 6) + i * 8 + k * BK, K - 1) * 64 + (threadIdx.x & 63)]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        
        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < 8; j++)
                c_tmp[j] += a_tmp[(threadIdx.x >> 3) * (2 * BK + 1) + ((k - 1) & 1) * BK + i] * b_tmp[((threadIdx.x & 7) + j * 8) * (2 * BK + 1) + ((k - 1) & 1) * BK + i];
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();

    }
    int k = ((K - 1) >> 5) + 1;
    for (int i = 0; i < BK; i++) {
        for (int j = 0; j < 8; j++)
            c_tmp[j] += (((k - 1) * BK + i) < K ? 1 : 0) * a_tmp[(threadIdx.x >> 3) * (2 * BK + 1) + ((k - 1) & 1) * BK + i] * b_tmp[((threadIdx.x & 7) + j * 8) * (2 * BK + 1) + ((k - 1) & 1) * BK + i];
    }
    for (int i = 0; i < 8; i++) {
        feat[min(node_len - 1, block_offset + (threadIdx.x >> 3)) * 64 + (threadIdx.x & 7) +  i * 8] = c_tmp[i];
        alphai += c_tmp[i] * att_i[(threadIdx.x & 7) + i * 8];
    }
    int i = 8;
    while (i > 1) {
        alphai += __shfl_xor_sync(FULL_MASK, alphai,  i - 1);
        i >>= 1;
    }
    alpha[min(node_len - 1, block_offset + (threadIdx.x >> 3))] = alphai;
 
}

__global__ void gat_kernel(
    const int* edge_index_0,
    const int* edge_index_1,
    const int* node_map,
    const int* count,
    const int* out,
    const float* feat,
    const float* alphai,
    const float* att_j_,
    float* output,
    int num_heads,
    int out_feats,
    int total_size,
    int edge_len
){
    const int num = (blockDim.x >> 5);
    int num_id = (threadIdx.x >> 5);
    const int edge_num = 6;
    extern __shared__ float params[];
    float* att_j = params;
    float* max_alpha = att_j + out_feats;
    float* alpha_sum = max_alpha + num_heads * 4;
    float* feats = alpha_sum + num_heads * 4;

    int cur_addr = __cvta_generic_to_shared(att_j) + ((threadIdx.x & 15) * 4) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j_[(threadIdx.x & 15) * 4]));

    int id = min(blockIdx.x * num + num_id, total_size - 1);
    int edge_begin = (id > 0 ? count[id - 1] : 0), edge_end = count[id];
    if (edge_end == edge_begin) return;
    float alpha_init = alphai[edge_index_1[out[edge_begin]]];
    float feat_tmp[2];
    max_alpha[num_id * num_heads + ((threadIdx.x & 31) % num_heads)] = 0.f;
    alpha_sum[num_id * num_heads + ((threadIdx.x & 31) % num_heads)] = 0.f;

    for (int i = 0; i < (edge_num >> 1); i++) {
        cur_addr = __cvta_generic_to_shared(feats) + ((num_id * edge_num + i * 2) * out_feats + (threadIdx.x & 31) * 4)* sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[min(edge_begin + i * 2 + ((threadIdx.x & 31) >> 4), edge_len - 1)] * out_feats + (threadIdx.x & 15) * 4]));
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncwarp(FULL_MASK);
    for (int k = 1; k < ((edge_end - edge_begin + edge_num - 1) / edge_num); k++) {
        for (int i = 0; i < (edge_num >> 1); i++) {
            cur_addr = __cvta_generic_to_shared(feats) + ((((k & 1) * num + num_id) * edge_num + i * 2) * out_feats + (threadIdx.x & 31) * 4)* sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[min(edge_begin + i * 2 + ((threadIdx.x & 31) >> 4) + k * edge_num, edge_len - 1)] * out_feats + (threadIdx.x & 15) * 4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        float alphas[edge_num];
        int agg_num = min(edge_end - edge_begin - k * edge_num, edge_num);
        if(num_heads > 1) {            
            int thread_part = out_feats / num_heads;
            int iter_num = out_feats / 32;
            for (int i = 0; i < iter_num; i++) {
                float alpha_max = std::numeric_limits<float>::lowest();
                for (int j = 0; j < agg_num; j++) {
                    alphas[j] = feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32] * att_j[(threadIdx.x & 31) + i * 32];
                    for (int l = 1; l < thread_part;) {
                        alphas[j] += __shfl_xor_sync(FULL_MASK, alphas[j], 2 * l - 1);
                        l *= 2;
                    }
                    alphas[j] += alpha_init;
                    alphas[j] = alphas[j] - 0.99f * min(0.0f, alphas[j]);
                    alpha_max = max(alpha_max, alphas[j]);
                }
                alpha_max = (k == 1) ? max(alpha_max, std::numeric_limits<float>::lowest()) : max(alpha_max, max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part]);
                float sum = 0;
                for (int j = 0; j < agg_num; j++) {
                    sum += __expf(alphas[j] - alpha_max);
                }
                sum += alpha_sum[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] * 
                            __expf(max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] - alpha_max);
                float s = (k > 1) ? __expf(max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] - alpha_max) * 
                            (alpha_sum[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] + 1e-16f) / (sum + 1e-16f) : 0.f;
                for (int j = 0; j < agg_num; j++) {
                    feat_tmp[i] = s * feat_tmp[i] + __expf(alphas[j] - alpha_max) / (sum + 1e-16f) * feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32];
                }
                max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] = alpha_max;
                alpha_sum[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] = sum;
            }
        } else {
            float alpha_max = std::numeric_limits<float>::lowest();
            for (int j = 0; j < agg_num; j++) {
                alphas[j] = 0.f;
                int iter_num = out_feats / 32;
                for (int i = 0; i < iter_num; i++) {
                    alphas[j] += feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32] * att_j[(threadIdx.x & 31) + i * 32];
                }
                for (int l = 1; l < 32;) {
                    alphas[j] += __shfl_xor_sync(FULL_MASK, alphas[j], 2 * l - 1);
                    l *= 2;
                }
                alphas[j] += alpha_init;
                alphas[j] = alphas[j] - 0.99f * min(0.0f, alphas[j]);
                alpha_max = max(alpha_max, alphas[j]);
            }
            alpha_max = (k == 1) ? max(alpha_max, std::numeric_limits<float>::lowest()) : max(alpha_max, max_alpha[num_id]);
            float sum = 0;
            float s = (k > 1) ? __expf(max_alpha[num_id] - alpha_max) * (alpha_sum[num_id] + 1e-16f) / (sum + 1e-16f) : 0.f;
            for (int j = 0; j < agg_num; j++) {
                int iter_num = out_feats / 32;
                for (int i = 0; i < iter_num; i++) {
                    feat_tmp[i] = s * feat_tmp[i] + __expf(alphas[j] - alpha_max) / (sum + 1e-16f) * feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32];
                }
            }
            max_alpha[num_id] = alpha_max;
            alpha_sum[num_id] = sum;
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncwarp(FULL_MASK);
    }
    int k = (edge_end - edge_begin + edge_num - 1) / edge_num;
    float alphas[4];
    int agg_num = min(edge_end - edge_begin - k * edge_num, edge_num);
    if(num_heads > 1) {            
        int thread_part = out_feats / num_heads;
        int iter_num = out_feats / 32;
        for (int i = 0; i < iter_num; i++) {
            float alpha_max = std::numeric_limits<float>::lowest();
            for (int j = 0; j < agg_num; j++) {
                alphas[j] = feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32] * att_j[(threadIdx.x & 31) + i * 32];
                for (int l = 1; l < thread_part;) {
                    alphas[j] += __shfl_xor_sync(FULL_MASK, alphas[j], 2 * l - 1);
                    l *= 2;
                }
                alphas[j] += alpha_init;
                alphas[j] = alphas[j] - 0.99f * min(0.0f, alphas[j]);
                alpha_max = max(alpha_max, alphas[j]);
            }
            alpha_max = (k == 1) ? max(alpha_max, std::numeric_limits<float>::lowest()) : max(alpha_max, max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part]);
            float sum = 0;
            for (int j = 0; j < agg_num; j++) {
                sum += __expf(alphas[j] - alpha_max);
            }
            sum += alpha_sum[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] * 
                        __expf(max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] - alpha_max);
            float s = (k > 1) ? __expf(max_alpha[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] - alpha_max) * 
                        (alpha_sum[num_id * num_heads + i * num_heads / iter_num + (threadIdx.x & 31) / thread_part] + 1e-16f) / (sum + 1e-16f) : 0.f;
            for (int j = 0; j < agg_num; j++) {
                feat_tmp[i] = s * feat_tmp[i] + __expf(alphas[j] - alpha_max) / (sum + 1e-16f) * feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32];
            }
        }
    } else {
        float alpha_max = std::numeric_limits<float>::lowest();
        for (int j = 0; j < agg_num; j++) {
            alphas[j] = 0.f;
            int iter_num = out_feats / 32;
            for (int i = 0; i < iter_num; i++) {
                alphas[j] += feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32] * att_j[(threadIdx.x & 31) + i * 32];
            }
            for (int l = 1; l < 32;) {
                alphas[j] += __shfl_xor_sync(FULL_MASK, alphas[j], 2 * l - 1);
                l *= 2;
            }
            alphas[j] += alpha_init;
            alphas[j] = alphas[j] - 0.99f * min(0.0f, alphas[j]);
            alpha_max = max(alpha_max, alphas[j]);
        }
        alpha_max = (k == 1) ? max(alpha_max, std::numeric_limits<float>::lowest()) : max(alpha_max, max_alpha[num_id]);
        float sum = 0;
        float s = (k > 1) ? __expf(max_alpha[num_id] - alpha_max) * (alpha_sum[num_id] + 1e-16f) / (sum + 1e-16f) : 0.f;
        for (int j = 0; j < agg_num; j++) {
            int iter_num = out_feats / 32;
            for (int i = 0; i < iter_num; i++) {
                feat_tmp[i] = s * feat_tmp[i] + __expf(alphas[j] - alpha_max) / (sum + 1e-16f) * feats[((((k - 1) & 1) * num + num_id) * edge_num + j) * out_feats + (threadIdx.x & 31) + i * 32];
            }
        }
    }
    for (int i = 0; i < 2; i++)
        output[node_map[edge_index_1[out[edge_begin]]] * out_feats + (threadIdx.x & 31) + i * 32] = feat_tmp[i];
}

at::Tensor GAT(
    torch::Tensor x,
    torch::Tensor edge_index_0,
    torch::Tensor edge_index_1,
    torch::Tensor node_map,
    torch::Tensor counts,
    torch::Tensor out,
    torch::Tensor lin_weight,
    torch::Tensor att_i,
    torch::Tensor att_j,
    int num_heads,
    int out_feats
) {
    auto size = x.sizes().vec();
    int edge_len = edge_index_0.sizes().vec()[0];
    auto output = torch::zeros({size[0], out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    float *node_feats, *node_alpha;
    cudaMalloc(&node_feats, size[0] * num_heads * out_feats * sizeof(float));
    cudaMalloc(&node_alpha, size[0] * sizeof(float));
    int grid = (size[0] + 63) / 64;

    // auto time = Timer();
    // time.tik();

    linear_reduce<<<grid, 512>>>(
        x.data_ptr<float>(),
        lin_weight.data_ptr<float>(),
        att_i.data_ptr<float>(),
        node_map.data_ptr<int>(),
        node_feats,
        node_alpha,
        size[0],
        size[1]
    );
    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "linear_reduce time: " << time.get_time() << std::endl;

    auto total_size = counts.sizes().vec()[0];
    grid = (total_size + 2) / 3;
    cudaFuncSetAttribute(gat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (64 + 4 * 2 + 64 * 6 * 3 * 2) * sizeof(float));
    // time.tik();
    gat_kernel<<<grid, 96, (64 + 4 * 2 + 64 * 6 * 3 * 2) * sizeof(float)>>>(
        edge_index_0.data_ptr<int>(),
        edge_index_1.data_ptr<int>(),
        node_map.data_ptr<int>(),
        counts.data_ptr<int>(),
        out.data_ptr<int>(),
        node_feats,
        node_alpha,
        att_j.data_ptr<float>(),
        output.data_ptr<float>(),
        num_heads,
        out_feats,
        total_size,
        edge_len
    );
    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "gat_kernel time: " << time.get_time() << std::endl;
    cudaFree(node_feats);
    cudaFree(node_alpha);
    return output;
}
