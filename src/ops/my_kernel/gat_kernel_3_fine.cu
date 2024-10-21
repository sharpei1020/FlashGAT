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

// __global__ void gat_kernel(
//     const int* edge_index_0,
//     const int* edge_index_1,
//     const int* map,
//     const int* count,
//     const int* out,
//     const float* feat,
//     const float* alphai,
//     const float* att_j_,
//     float* output,
//     int num_heads,
//     int out_feats,
//     int total_size,
//     int edge_len
// ){
//     extern __shared__ float params[];
//     float* att_j = params;
//     float* feats = att_j + 64;
//     int id = map[min(blockIdx.x * 2 + (threadIdx.x >> 5), total_size - 1)];
//     int n = out_feats / 8;
//     int edge_begin = count[id], edge_end = count[id + 1];
//     float alpha_init = alphai[edge_index_1[out[edge_begin]]];
//     float max_alpha = std::numeric_limits<float>::lowest(), last_max_alpha = 0.f, alpha_sum = 0.f, feat_part[8] = {0.f};

//     int cur_addr = __cvta_generic_to_shared(att_j) + (threadIdx.x & 31) * 2 * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&att_j_[(threadIdx.x & 31) * 2]));
//     for (int i = 0; i < 8; i++) {
//         cur_addr = __cvta_generic_to_shared(feats) + ((threadIdx.x >> 5) * ((out_feats * num_heads) >> 2) * 33 + i * 33 + (threadIdx.x & 31)) * sizeof(float);
//         asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + (i >> 1), edge_end - 1)]] * out_feats * num_heads + (threadIdx.x & 31) + (i & 1) * 32]));
//     }
//     asm volatile("cp.async.commit_group;\n"::);
//     asm volatile("cp.async.wait_group 0;\n"::);
//     __syncwarp();
//     for (int k = 1; k < ((edge_end - edge_begin + 3) / 4); k++) {
//         for (int i = 0; i < 8; i++) {
//             cur_addr = __cvta_generic_to_shared(feats) + ((threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 33 + (i + ((out_feats * num_heads) >> 3) * (k & 1))* 33 + (threadIdx.x & 31)) * sizeof(float);
//             asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + (i >> 1) + k * 4, edge_end - 1)]] * out_feats * num_heads + (threadIdx.x & 31) + (i & 1) * 32]));
//         }
//         asm volatile("cp.async.commit_group;\n"::);
//         float alpha = 0.f;
//         for (int i = 0; i < 8; i++) {
//             alpha += feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 33 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)] * att_j[(threadIdx.x & 7) + i * 8 + (i >> 2)];
//         }
//         int iter = n;
//         while(iter > 1) {
//             alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
//             iter >>= 1;
//         } 
//         alpha += alpha_init;
//         alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
//         iter = 32;
//         while(iter > n * num_heads) {
//             alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - 1));
//             iter >>= 1;
//         }
//         max_alpha = max(max_alpha, alpha);
//         float sum = __expf(alpha - max_alpha);
//         iter = 32;
//         while(iter > n * num_heads) {
//             sum += __shfl_xor_sync(FULL_MASK, sum,  iter - 1);
//             iter >>= 1;
//         }
//         sum += alpha_sum * __expf(last_max_alpha - max_alpha);
//         float s = ((sum > 0.f) ? (__expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f)) : 0.f);
//         float w = __expf(alpha - max_alpha) / (sum + 1e-16f);
//         for (int i = 0; i < 8; i++) {
//             float feat_tmp = w * feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 33 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)];
//             iter = 32;
//             while(iter > n * num_heads) {
//                 feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - 1);
//                 iter >>= 1;
//             }
//             feat_part[i] = s * feat_part[i] + feat_tmp;
//         }
//         last_max_alpha = max_alpha;
//         alpha_sum = sum;
//         __syncwarp();
//     }
//     float alpha = 0.f;
//     int k = (edge_end - edge_begin + 3) / 4 - 1;
//     for (int i = 0; i < 8; i++) {
//         alpha += feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 33 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)] * att_j[(threadIdx.x & 7) + i * 8 + (i >> 2)];
//     }
//     int iter = n;
//     while(iter > 1) {
//         alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
//         iter >>= 1;
//     } 
//     alpha += alpha_init;
//     alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
//     iter = 32;
//     while(iter > n * num_heads) {
//         alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - n * num_heads));
//         iter >>= 1;
//     }
//     max_alpha = max(max_alpha, alpha);
//     float sum = (k * 4 + edge_begin + (threadIdx.x >> 3) & 3) < edge_end ? __expf(alpha - max_alpha) : 0.f;
//     iter = 32;
//     while(iter > n * num_heads) {
//         sum += __shfl_xor_sync(FULL_MASK, sum,  iter - n * num_heads);
//         iter >>= 1;
//     }
//     sum += alpha_sum * __expf(last_max_alpha - max_alpha);
//     float s = ((sum > 0.f) ? (__expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f)) : 0.f);
//     float w = (k * 4 + edge_begin + (threadIdx.x >> 3) & 3) < edge_end ? __expf(alpha - max_alpha) / (sum + 1e-16f) : 0.f;
//     for (int i = 0; i < 8; i++) {
//         float feat_tmp = w * feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 33 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)];
//         iter = 32;
//         while(iter > n * num_heads) {
//             feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - n * num_heads);
//             iter >>= 1;
//         }
//         feat_part[i] = s * feat_part[i] + feat_tmp;
//     }
//     for (int i = 0; i < 8; i++)
//         output[edge_index_1[out[edge_begin]] * out_feats * num_heads + (threadIdx.x & 7) + i * 8] = feat_part[i];
// }

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
    extern __shared__ float params[];
    float* att_j = params;
    float* feats = att_j + 68;
    int group = 1;
    
    int cur_addr = __cvta_generic_to_shared(att_j) + ((threadIdx.x & 16) * 4 + ((threadIdx.x & 8) >> 1)) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j_[(threadIdx.x & 16) * 4]));
    for (int j = 0; j < group; j++) {
        int id = min((blockIdx.x * group + j) * 3 + (threadIdx.x >> 5), total_size - 1);
        int n = out_feats / 8;
        int edge_begin = (id > 0 ? count[id - 1] : 0), edge_end = count[id];
        if (edge_begin >= edge_end) continue;
        float alpha_init = alphai[edge_index_1[out[edge_begin]]];
        float max_alpha = std::numeric_limits<float>::lowest(), last_max_alpha = 0.f, alpha_sum = 0.f, feat_part[8];
 
        for (int i = 0; i < 2; i++) {
            cur_addr = __cvta_generic_to_shared(feats) + ((threadIdx.x >> 5) * ((out_feats * num_heads) >> 2) * 36 + i * 144 + (threadIdx.x & 31) * 4 + ((threadIdx.x & 24) >> 1)) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + i * 2 + ((threadIdx.x & 16) >> 4), edge_end - 1)]] * out_feats * num_heads + (threadIdx.x & 15)* 4]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncwarp();
        for (int k = 1; k < ((edge_end - edge_begin + 3) / 4); k++) {
            for (int i = 0; i < 2; i++) {
                cur_addr = __cvta_generic_to_shared(feats) + ((threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 36 + (i * 4 + ((out_feats * num_heads) >> 3) * (k & 1))* 36 + (threadIdx.x & 31) * 4 + ((threadIdx.x & 24) >> 1)) * sizeof(float);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + i * 2 + k * 4 + ((threadIdx.x & 16) >> 4), edge_end - 1)]] * out_feats * num_heads + (threadIdx.x & 15)* 4]));
            }
            asm volatile("cp.async.commit_group;\n"::);
            float alpha = 0.f;
            for (int i = 0; i < 8; i++) {
                alpha += feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 36 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 36 + (threadIdx.x & 7) + i * 8 + (i & 4)] * att_j[(threadIdx.x & 7) + i * 8 + (i & 4)];
            }
            int iter = n;
            while(iter > 1) {
                alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
                iter >>= 1;
            } 
            alpha += alpha_init;
            alpha = alpha - 0.99f * min(0.0f, alpha);
            iter = 32;
            while(iter > n * num_heads) {
                alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - n * num_heads));
                iter >>= 1;
            }
            max_alpha = max(max_alpha, alpha);
            float sum = __expf(alpha - max_alpha);
            iter = 32;
            while(iter > n * num_heads) {
                sum += __shfl_xor_sync(FULL_MASK, sum,  iter - n * num_heads);
                iter >>= 1;
            }
            sum += alpha_sum * __expf(last_max_alpha - max_alpha);
            float s = ((alpha_sum > 0.f) ? (__expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f)) : 0.f);
            float w = __expf(alpha - max_alpha) / (sum + 1e-16f);
            for (int i = 0; i < 8; i++) {
                float feat_tmp = w * feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 36 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 36 + (threadIdx.x & 7) + i * 8 + (i & 4)];
                iter = 32;
                while(iter > n * num_heads) {
                    feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - n * num_heads);
                    iter >>= 1;
                }
                feat_part[i] = s * feat_part[i] + feat_tmp;
            }
            last_max_alpha = max_alpha;
            alpha_sum = sum;
            asm volatile("cp.async.wait_group 0;\n"::);
            __syncwarp();
        }
        float alpha = 0.f;
        int k = (edge_end - edge_begin + 3) / 4;
        for (int i = 0; i < 8; i++) {
            alpha += feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 36 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 36 + (threadIdx.x & 7) + i * 8 + (i & 4)] * att_j[(threadIdx.x & 7) + i * 8 + (i & 4)];
        }
        int iter = n;
        while(iter > 1) {
            alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
            iter >>= 1;
        } 
        alpha += alpha_init;
        alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
        iter = 32;
        while(iter > n * num_heads) {
            alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - n * num_heads));
            iter >>= 1;
        }
        max_alpha = max(max_alpha, alpha);
        float sum = ((k * 4 + edge_begin + (threadIdx.x & 31) / (n * num_heads)) < edge_end) ? __expf(alpha - max_alpha) : 0.f;
        iter = 32;
        while(iter > n * num_heads) {
            sum += __shfl_xor_sync(FULL_MASK, sum,  iter - n * num_heads);
            iter >>= 1;
        }
        sum += alpha_sum * __expf(last_max_alpha - max_alpha);
        float s = __expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f);
        float w = (k * 4 + edge_begin + (threadIdx.x >> 3) & 3) < edge_end ? __expf(alpha - max_alpha) / (sum + 1e-16f) : 0.f;
        for (int i = 0; i < 8; i++) {
            float feat_tmp = w * feats[(threadIdx.x >> 5) * (((out_feats * num_heads) >> 2)) * 36 + (((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * ((threadIdx.x >> 3) & 3)) * 36 + (threadIdx.x & 7) + i * 8 + (i & 4)];
            iter = 32;
            while(iter > n * num_heads) {
                feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - n * num_heads);
                iter >>= 1;
            }
            feat_part[i] = s * feat_part[i] + feat_tmp;
        }
        for (int i = 0; i < 8; i++)
            output[node_map[edge_index_1[out[edge_begin]]] * out_feats * num_heads + (threadIdx.x & 7) + i * 8] = feat_part[i];
    }
}

// __global__ void gat_kernel(
//     const int* edge_index_0,
//     const int* edge_index_1,
//     const int* count,
//     const int* out,
//     const float* feat,
//     const float* alphai,
//     const float* att_j_,
//     float* output,
//     int num_heads,
//     int out_feats,
//     int total_size,
//     int edge_len
// ){
//     extern __shared__ float params[];
//     float* att_j = params;
//     float* feats = att_j + 64;
//     int n = out_feats / 8;
//     int edge_begin = count[min(blockIdx.x, total_size - 1)], edge_end = count[min(blockIdx.x + 1, total_size)];
//     float alpha_init = alphai[edge_index_1[out[edge_begin]]];
//     float max_alpha = std::numeric_limits<float>::lowest(), last_max_alpha = 0.f, alpha_sum = 0.f, feat_part[8] = {0.f};

//     int cur_addr = __cvta_generic_to_shared(att_j) + threadIdx.x * 2 * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&att_j_[threadIdx.x * 2]));
//     for (int i = 0; i < 8; i++) {
//         cur_addr = __cvta_generic_to_shared(feats) + (i * 33 + threadIdx.x) * sizeof(float);
//         asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + (i >> 1), edge_end - 1)]] * out_feats * num_heads + threadIdx.x + (i & 1) * 32]));
//     }
//     asm volatile("cp.async.commit_group;\n"::);
//     asm volatile("cp.async.wait_group 0;\n"::);
//     __syncwarp();
//     for (int k = 1; k < ((edge_end - edge_begin + 3) / 4); k++) {
//         for (int i = 0; i < 8; i++) {
//             cur_addr = __cvta_generic_to_shared(feats) + ((i + ((out_feats * num_heads) >> 3) * (k & 1))* 33 + threadIdx.x) * sizeof(float);
//             asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feat[edge_index_0[out[min(edge_begin + (i >> 1) + k * 4, edge_end - 1)]] * out_feats * num_heads + threadIdx.x + (i & 1) * 32]));
//         }
//         asm volatile("cp.async.commit_group;\n"::);
//         float alpha = 0.f;
//         for (int i = 0; i < 8; i++) {
//             alpha += feats[(((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * (threadIdx.x >> 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)] * att_j[(threadIdx.x & 7) + i * 8 + (i >> 2)];
//         }
//         int iter = n;
//         while(iter > 1) {
//             alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
//             iter >>= 1;
//         } 
//         alpha += alpha_init;
//         alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
//         iter = 32;
//         while(iter > n * num_heads) {
//             alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - 1));
//             iter >>= 1;
//         }
//         max_alpha = max(max_alpha, alpha);
//         float sum = __expf(alpha - max_alpha);
//         iter = 32;
//         while(iter > n * num_heads) {
//             sum += __shfl_xor_sync(FULL_MASK, sum,  iter - 1);
//             iter >>= 1;
//         }
//         sum += alpha_sum * __expf(last_max_alpha - max_alpha);
//         float s = ((sum > 0.f) ? (__expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f)) : 0.f);
//         float w = __expf(alpha - max_alpha) / (sum + 1e-16f);
//         for (int i = 0; i < 8; i++) {
//             float feat_tmp = w * feats[(((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * (threadIdx.x >> 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)];
//             iter = 32;
//             while(iter > n * num_heads) {
//                 feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - 1);
//                 iter >>= 1;
//             }
//             feat_part[i] = s * feat_part[i] + feat_tmp;
//         }
//         last_max_alpha = max_alpha;
//         alpha_sum = sum;
//         __syncwarp();
//     }
//     float alpha = 0.f;
//     int k = (edge_end - edge_begin + 3) / 4 - 1;
//     for (int i = 0; i < 8; i++) {
//         alpha += feats[(((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * (threadIdx.x >> 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)] * att_j[(threadIdx.x & 7) + i * 8 + (i >> 2)];
//     }
//     int iter = n;
//     while(iter > 1) {
//         alpha += __shfl_xor_sync(FULL_MASK, alpha,  iter - 1);
//         iter >>= 1;
//     } 
//     alpha += alpha_init;
//     alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
//     iter = 32;
//     while(iter > n * num_heads) {
//         alpha = max(alpha, __shfl_xor_sync(FULL_MASK, alpha,  iter - 1));
//         iter >>= 1;
//     }
//     max_alpha = max(max_alpha, alpha);
//     float sum = (k * 4 + edge_begin + (threadIdx.x >> 3) & 3) < edge_end ? __expf(alpha - max_alpha) : 0.f;
//     iter = 32;
//     while(iter > n * num_heads) {
//         sum += __shfl_xor_sync(FULL_MASK, sum,  iter - 1);
//         iter >>= 1;
//     }
//     sum += alpha_sum * __expf(last_max_alpha - max_alpha);
//     float s = ((sum > 0.f) ? (__expf(last_max_alpha - max_alpha) * (alpha_sum + 1e-16f) / (sum + 1e-16f)) : 0.f);
//     float w = (k * 4 + edge_begin + (threadIdx.x >> 3) & 3) < edge_end ? __expf(alpha - max_alpha) / (sum + 1e-16f) : 0.f;
//     for (int i = 0; i < 8; i++) {
//         float feat_tmp = w * feats[(((out_feats * num_heads) >> 3) * ((k - 1) & 1) + ((out_feats * num_heads) >> 5) * (threadIdx.x >> 3)) * 33 + (threadIdx.x & 7) + i * 8 + (i >> 2)];
//         iter = 32;
//         while(iter > n * num_heads) {
//             feat_tmp += __shfl_xor_sync(FULL_MASK, feat_tmp,  iter - 1);
//             iter >>= 1;
//         }
//         feat_part[i] = s * feat_part[i] + feat_tmp;
//     }
//     for (int i = 0; i < 8; i++)
//         output[edge_index_1[out[edge_begin]] * out_feats * num_heads + (threadIdx.x & 7) + i * 8] = feat_part[i];
// }

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
    cudaFuncSetAttribute(gat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (68 + 72 * 24) * sizeof(float));
    // time.tik();
    gat_kernel<<<grid, 96, (68 + 72 * 24) * sizeof(float)>>>(
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
