#include "gat.cuh"
#include "assert.h"
#include "time.cuh"

#define TILESIZE_X 128
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__device__ static inline float atomicMax(float *address, const float val) {
    unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
    unsigned int old = *address_as_ui;                       // NOLINT
    unsigned int assumed;                                    // NOLINT
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);  // NOLINT
    return __uint_as_float(old);
}


__global__ void linear(const float* __restrict__ weight, const float* __restrict__ bias, const float* __restrict__ X, float*  __restrict__ Z, int M, int K, bool bias_flag) {
    const int BK = 4;
    const int K_rem = K % BK;
    const int more = (K_rem > 0) ? 1 : 0;
    // bool flag = false;

    // if (blockIdx.x * TILESIZE_X >= M) flag = true;
    __shared__ float tmp_x[TILESIZE_X*(2* BK)], tmp_y[TILESIZE_Y*(2*BK)];
    int tmp_x_addr = __cvta_generic_to_shared(tmp_x);
    int tmp_y_addr = __cvta_generic_to_shared(tmp_y);
    
    float tmp_c[32];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            tmp_c[i * 4 + j] = bias_flag ? bias[j*16+(threadIdx.x & 15)] : 0.f;

    int tmp_x_cur_addr = tmp_x_addr + ((threadIdx.x >> 1) * (2 * BK)) * sizeof(float);
    int in_x_cur_addr = (blockIdx.x * TILESIZE_X + min((threadIdx.x >> 1), M - blockIdx.x * TILESIZE_X - 1)) * K;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_x_cur_addr), "l"(&X[in_x_cur_addr]));   
    int tmp_y_cur_addr = tmp_y_addr + ((threadIdx.x >> 2) * (2 * BK)) * sizeof(float);
    int in_y_cur_addr = (threadIdx.x >> 2) * K;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(tmp_y_cur_addr), "l"(&weight[in_y_cur_addr]));
    
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
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Z[min((blockIdx.x * TILESIZE_X + (threadIdx.x >> 4) + i * 16), M-1) * TILESIZE_Y + j * 16 + (threadIdx.x & 15)] = tmp_c[i * 4 + j];
        }
}

// __global__ void gat(
//     int *edge_index_0,
//     int *edge_index_1, 
//     int *count, 
//     int *out, 
//     float *x,
//     float *output,
    
//     float *att_i,
//     float *att_j,
    
//     int num_heads,
//     int out_feats,
//     int total_size,
//     int edge_len) {

//     extern __shared__ float parameters[];
//     float *att_i_param = parameters;
//     float *att_j_param = parameters + num_heads * out_feats;

//     {
//         int cur_addr = __cvta_generic_to_shared(att_i_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//         asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_i[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
//         cur_addr = __cvta_generic_to_shared(att_j_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//         asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));

//         asm volatile("cp.async.commit_group;\n"::);
//         asm volatile("cp.async.wait_group 0;\n"::);

//         __syncthreads();
//     }

//     int idx = blockIdx.x * 32 + (threadIdx.x >> 3);

//     if (idx >= total_size) return;

//     int edge_begin = count[idx];
//     int edge_end = count[idx + 1];

//     for (int k = 0; k < num_heads; k++) {
//         float alpha_sum = 0.0f, last_alpha_sum = 1.0f;
//         float alpha_max = std::numeric_limits<float>::lowest();
//         float last_alpha_max = 1.0f;
//         float temp_sum[8] = {0.0f}, temp[8];
//         for (int i = edge_begin; i < edge_end; i++) {
//             float alpha = 0.0f;
//             for (int j = 0; j < 2; j++) {
//                 float4 tmp = FLOAT4(x[edge_index_0[out[i]] * num_heads * out_feats + k * out_feats + j * 4 + (threadIdx.x & 7) * 8]);
//                 float4 att = FLOAT4(att_j_param[k * out_feats + j * 4 + (threadIdx.x & 7) * 8]);
//                 alpha += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//                 temp[j * 4] = tmp.x;
//                 temp[j * 4 + 1] = tmp.y;
//                 temp[j * 4 + 2] = tmp.z;
//                 temp[j * 4 + 3] = tmp.w;
//                 tmp = FLOAT4(x[edge_index_1[out[i]] * num_heads * out_feats + k * out_feats + j * 4 + (threadIdx.x & 7) * 8]);
//                 att = FLOAT4(att_i_param[k * out_feats + j * 4 + (threadIdx.x & 7) * 8]);
//                 alpha += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//             }
//             int loop = 1;
//             while (loop < (out_feats / 8)) {
//                 loop *= 2;
//                 alpha += __shfl_xor_sync(FULL_MASK, alpha, loop - 1);
//             }
//             alpha = __expf(max(0.0f, alpha) + 0.01f * min(0.0f, alpha));
//             alpha_max = max(alpha_max, alpha);
//             alpha_sum *= (last_alpha_max / alpha_max);
//             alpha_sum += (alpha / alpha_max);
//             scale(temp_sum, (last_alpha_max * last_alpha_sum) / (alpha_max * alpha_sum), 8);
//             scale_sum(temp_sum, temp, (alpha / (alpha_max * alpha_sum)), 8);
//             last_alpha_max = alpha_max;
//             last_alpha_sum = alpha_sum;
//         }
//         debug(temp_sum, output + edge_index_1[out[edge_begin]] * num_heads * out_feats + k * out_feats + (threadIdx.x & 7) * 8, 8);    
//     }       
// }

// template<int out_feats>
// __global__ void gat(
//     int *edge_index_0,
//     int *edge_index_1,
//     int *count,
//     int *out,
//     float *x,
//     float *output,
    
//     float* att_i,
//     float* att_j,
//     int num_heads,

//     int total_size,
//     int edge_len
// ) {
//     extern __shared__ float parameters[];
//     float* att_j_param = parameters;
//     float* att_i_param = parameters + num_heads * out_feats;
//     float* alpha_init = parameters + 2 * num_heads * out_feats;
//     float* max_alpha = alpha_init + blockDim.x;
//     float* alpha_sum = max_alpha + blockDim.x;
//     float* last_max_alpha = alpha_sum + blockDim.x;
//     float* last_alpha_sum = last_max_alpha + blockDim.x;
//     float* feat_tmp = last_alpha_sum + blockDim.x;
//     float feat[out_feats], temp_x[out_feats];
//     for (int i = 0; i < out_feats; i++)
//         feat[i] = 0.f;

//     int cur_addr = __cvta_generic_to_shared(att_i_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_i[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
//     cur_addr = __cvta_generic_to_shared(att_j_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
//     asm volatile("cp.async.commit_group;\n"::);

//     int id = blockIdx.x * (blockDim.x / num_heads) + (threadIdx.x / num_heads);
//     int edge_begin = count[min(id, total_size-1)];
//     alpha_init[threadIdx.x] = 0.0f;
//     max_alpha[threadIdx.x] = std::numeric_limits<float>::lowest();
//     alpha_sum[threadIdx.x] = 0.f;
//     last_max_alpha[threadIdx.x] = 0.f;
//     last_alpha_sum[threadIdx.x] = 0.f;

//     asm volatile("cp.async.wait_group 0;\n"::);
//     __syncthreads();

//     for (int i = 0; i < (out_feats / 4); i++) {
//         float4 tmp = FLOAT4(x[edge_index_1[out[edge_begin]] * num_heads * out_feats + i * 4 + (threadIdx.x & (num_heads - 1)) * out_feats]);
//         float4 att = FLOAT4(att_i_param[i * 4 + (threadIdx.x & (num_heads - 1)) * out_feats]);
//         alpha_init[threadIdx.x] += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//     }

//     int* edge_off = (int*)(feat_tmp + blockDim.x);
//     edge_off[threadIdx.x] = count[min(id+1, total_size)];
//     __syncwarp();
//     int start = __shfl_sync(FULL_MASK, edge_begin, 0);
//     int start_ = start + ((threadIdx.x & 31) / num_heads);
//     int end = edge_off[(threadIdx.x & 224) + 31];
//     int warp_node_num = 32 / num_heads;
//     int end_ = ((end - start) + (warp_node_num - 1)) / warp_node_num * warp_node_num + start;
//     for (int i = start_; i < end_; i += warp_node_num) {
//         int idx = min(i, end-1);
//         int map = out[idx];
//         int block_delta = (threadIdx.x & (blockDim.x - 32)), delta = (threadIdx.x & (num_heads - 1));
//         for (int j = (threadIdx.x & (num_heads - 1)); j < 32; j += num_heads) {
//             delta += ((idx >= edge_off[(threadIdx.x & 224) + j]) ? 1 : 0) * num_heads;
//         }
//         block_delta += delta;
//         float alpha = alpha_init[block_delta];
//         for (int j = 0; j < (out_feats / 4); j++) {
//             float4 tmp = FLOAT4(x[edge_index_0[map] * num_heads * out_feats + j * 4 + (threadIdx.x & (num_heads - 1)) * out_feats]);
//             float4 att = FLOAT4(att_j_param[j * 4 + (threadIdx.x & (num_heads - 1)) * out_feats]);
//             FLOAT4(temp_x[j * 4]) = tmp;
//             alpha += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//         }
//         u_int agg_mask = __match_any_sync(FULL_MASK, delta);
//         u_int max_ = __reduce_max_sync(agg_mask, __float_as_uint(alpha));
//         max_alpha[block_delta] = max(max_alpha[block_delta], __uint_as_float(max_));
//         // atomicMax(max_alpha + block_delta, alpha);
//         __syncwarp();
//         alpha_sum[threadIdx.x] *= __expf(last_max_alpha[threadIdx.x] - max_alpha[threadIdx.x]);
//         __syncwarp();
//         u_int sum_ = __reduce_add_sync(agg_mask, __float_as_uint(((i > end) ? 0.f : __expf(alpha - max_alpha[block_delta]))));
//         alpha_sum[block_delta] = __uint_as_float(sum_);
//         // atomicAdd(alpha_sum + block_delta, ((i > end) ? 0.f : __expf(alpha - max_alpha[block_delta])));
//         __syncwarp();
//         float s = ((alpha_sum[threadIdx.x] > 0.f) ? 
//             (__expf(last_max_alpha[threadIdx.x] - max_alpha[threadIdx.x]) * (last_alpha_sum[threadIdx.x] + 1e-16f) / (alpha_sum[threadIdx.x] + 1e-16f)) : 0.f);
//         scale(feat, s, out_feats);
//         __syncwarp();
//         float w = __expf(alpha - max_alpha[block_delta]) / (alpha_sum[threadIdx.x] + 1e-16f);
//         feat_tmp[threadIdx.x] = 0.f;
//         for (int j = 0; j < out_feats; j++) {
//             float val = (i >= end) ? 0.f : (w * temp_x[j]);
//             u_int sum = __reduce_add_sync(agg_mask, __float_as_uint(val));
//             feat_tmp[block_delta] = __uint_as_float(sum);
//             __syncwarp();
//             feat[j] += feat_tmp[threadIdx.x];
//         }
//         last_max_alpha[block_delta] = max_alpha[block_delta];
//         last_alpha_sum[block_delta] = alpha_sum[block_delta];
//         __syncwarp(); //maybe not necessary
//     }
//     if (id < total_size) {
//         debug(feat, output + edge_index_1[out[edge_begin]] * num_heads * out_feats + (threadIdx.x & (num_heads - 1)) * out_feats, out_feats);
//     }

// }

__global__ __launch_bounds__(256, 6) void gat(
    int* edge_index_0,
    int* edge_index_1,
    int* count,
    int* out,
    float* x,
    float* output,

    float* att_i,
    float* att_j,

    int num_heads,
    int out_feats,
    int total_size,
    int edge_len
) {
    extern __shared__ float parameters[];
    float* alpha_init = parameters;
    int n = out_feats / 8;
    float* max_alpha = alpha_init + (blockDim.x / n);
    float* last_max_alpha = max_alpha + (blockDim.x / n);
    float* alpha_sum = last_max_alpha + (blockDim.x / n);
    int* edge_off = (int*)(alpha_sum + (blockDim.x / n));
    float* att_j_param = alpha_sum + 2 * (blockDim.x / n);
    float* att_i_param = att_j_param + num_heads * out_feats;
    float* feat = att_i_param + num_heads * out_feats;

    float temp[8];

    int cur_addr = __cvta_generic_to_shared(att_i_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_i[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
    cur_addr = __cvta_generic_to_shared(att_j_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
    for (int i = 0; i < 2; i++) {
        int idx = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (2 * num_heads * n) + i * (blockDim.x / (2 * num_heads * n)));
        cur_addr = __cvta_generic_to_shared(feat) + (threadIdx.x * 4 + i * (blockDim.x * 4 + (blockDim.x >> 2)) + (threadIdx.x >> 3) * 4) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[edge_index_1[out[idx]] * num_heads * out_feats + (threadIdx.x & (2 * num_heads * n - 1)) * 4]));
    } 
    asm volatile("cp.async.commit_group;\n"::);

    int id = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (num_heads * n));
    int edge_begin = count[min(id, total_size-1)];
    alpha_init[(threadIdx.x / n)] = 0.0f;
    max_alpha[(threadIdx.x / n)] = std::numeric_limits<float>::lowest();
    last_max_alpha[(threadIdx.x / n)] = 0.f;
    alpha_sum[(threadIdx.x / n)] = 0.f;

    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    for (int i = 0; i < 2; i++) {
        // float4 tmp = FLOAT4(x[edge_index_1[out[edge_begin]] * num_heads * out_feats + i * 4 + (threadIdx.x & (n - 1)) * 8 + ((threadIdx.x / n) & (num_heads - 1)) * out_feats]);
        float4 tmp = FLOAT4(feat[(threadIdx.x >> 2) * 36 + i * 4 + (threadIdx.x & 3) * 8]);
        float4 att = FLOAT4(att_i_param[i * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
        alpha_init[threadIdx.x/n] += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
    }
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
        int map = out[idx];
        int block_delta = (threadIdx.x & (blockDim.x - 32)) / n, delta = ((threadIdx.x/n) & (num_heads - 1));
        for (int j = delta; j < (32/n); j += num_heads) {
            delta += ((idx >= edge_off[block_delta + j]) ? 1 : 0) * num_heads;
        }
        block_delta += delta;
        float alpha = alpha_init[block_delta];
        for (int j = 0; j < 2; j++) {
            float4 tmp = FLOAT4(x[edge_index_0[map] * num_heads * out_feats + j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
            float4 att = FLOAT4(att_j_param[j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
            FLOAT4(temp[j * 4]) = tmp;
            alpha += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
        }
        alpha = __expf(alpha - 0.99f * min(0.0f, alpha));
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
        debug(feat + threadIdx.x * 9, output + edge_index_1[out[edge_begin]] * num_heads * out_feats + ((threadIdx.x / n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8, 8);
    }
}

// __global__ __launch_bounds__(256, 6) void gat(
//     int* edge_index_0,
//     int* edge_index_1,
//     int* count,
//     int* out,
//     float* x,
//     float* output,

//     float* att_i,
//     float* att_j,

//     int num_heads,
//     int out_feats,
//     int total_size,
//     int edge_len
// ) {
//     extern __shared__ float parameters[];
//     float* alpha_init = parameters;
//     int n = out_feats / 16;
//     float* max_alpha = alpha_init + (blockDim.x / n);
//     float* last_max_alpha = max_alpha + (blockDim.x / n);
//     float* alpha_sum = last_max_alpha + (blockDim.x / n);
//     int* edge_off = (int*)(alpha_sum + (blockDim.x / n));
//     float* att_j_param = alpha_sum + 2 * (blockDim.x / n);
//     float* att_i_param = att_j_param + num_heads * out_feats;
//     float* feat = att_i_param + num_heads * out_feats;

//     float temp[16];

//     int cur_addr = __cvta_generic_to_shared(att_i_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_i[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
//     cur_addr = __cvta_generic_to_shared(att_j_param) + (min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4) * sizeof(float);
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j[min(threadIdx.x >> 4, num_heads-1) * 64 + (threadIdx.x & 15) * 4]));
//     for (int i = 0; i < 4; i++) {
//         int idx = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (out_feats * num_heads / 4) + i * (blockDim.x / (out_feats * num_heads / 4)));
//         cur_addr = __cvta_generic_to_shared(feat) + (threadIdx.x * 4 + i * (blockDim.x + (blockDim.x >> 3)) * 4 + (threadIdx.x >> 3) * 4) * sizeof(float);
//         asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[edge_index_1[out[idx]] * num_heads * out_feats + (threadIdx.x & (out_feats * num_heads / 4 - 1)) * 4]));
//     } 
//     asm volatile("cp.async.commit_group;\n"::);

//     int id = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (num_heads * n));
//     int edge_begin = count[min(id, total_size-1)];
//     alpha_init[(threadIdx.x / n)] = 0.0f;
//     max_alpha[(threadIdx.x / n)] = std::numeric_limits<float>::lowest();
//     last_max_alpha[(threadIdx.x / n)] = 0.f;
//     alpha_sum[(threadIdx.x / n)] = 0.f;

//     asm volatile("cp.async.wait_group 0;\n"::);
//     __syncthreads();

//     for (int i = 0; i < 4; i++) {
//         // float4 tmp = FLOAT4(x[edge_index_1[out[edge_begin]] * num_heads * out_feats + i * 4 + (threadIdx.x & (n - 1)) * 8 + ((threadIdx.x / n) & (num_heads - 1)) * out_feats]);
//         float4 tmp = FLOAT4(feat[(threadIdx.x >> 1) * 36 + i * 4 + (threadIdx.x & 1) * 16]);
//         float4 att = FLOAT4(att_i_param[i * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 16]);
//         alpha_init[threadIdx.x/n] += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//     }
//     edge_off[threadIdx.x / n] = count[min(id+1, total_size)];
//     int start = __shfl_sync(FULL_MASK, edge_begin, 0);
//     int start_ = start + ((threadIdx.x & 31) / (num_heads * n));
//     int end = edge_off[(threadIdx.x & (blockDim.x - 32)) / n + (32 / n) - 1];
//     int warp_node_num = 256 / (num_heads * out_feats);
//     int end_ = ((end - start) + (warp_node_num - 1)) / warp_node_num * warp_node_num + start;
//     u_int filter = (1 << (threadIdx.x & (n - 1)));
//     for (int j = 1; j < (32 / n); j++) {
//         filter |= (filter << n);
//     }
//     for (int i = 0; i < 16; i++)
//         feat[threadIdx.x * 17 + i] = 0.f;
//     for (int i = start_; i < end_; i += warp_node_num) {
//         int idx = min(i, end-1);
//         int map = out[idx];
//         int block_delta = (threadIdx.x & (blockDim.x - 32)) / n, delta = ((threadIdx.x/n) & (num_heads - 1));
//         for (int j = delta; j < (32/n); j += num_heads) {
//             delta += ((idx >= edge_off[block_delta + j]) ? 1 : 0) * num_heads;
//         }
//         block_delta += delta;
//         float alpha = alpha_init[block_delta];
//         for (int j = 0; j < 4; j++) {
//             float4 tmp = FLOAT4(x[edge_index_0[map] * num_heads * out_feats + j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 16]);
//             float4 att = FLOAT4(att_j_param[j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 16]);
//             FLOAT4(temp[j * 4]) = tmp;
//             alpha += (tmp.x * att.x + tmp.y * att.y + tmp.z * att.z + tmp.w * att.w);
//         }
//         alpha = __expf(max(0.0f, alpha) + 0.01f * min(0.0f, alpha));
//         int iter = n;
//         while (iter > 1) {
//             alpha += __shfl_xor_sync(FULL_MASK, alpha, iter-1);
//             iter /= 2;
//         }
//         u_int agg_mask = (__match_any_sync(FULL_MASK, delta) & filter);
//         int num = __popc(agg_mask), offset = __ffs(agg_mask) - 1;
//         iter = num;
//         int iter_offset = offset;
//         float max_ = max(max_alpha[block_delta], alpha);
//         while (iter > 1) {
//             max_ = max(__shfl_sync(agg_mask, max_, 2 * iter_offset + iter - 1 - (threadIdx.x & 31)), max_);
//             iter = (iter + 1) >> 1;
//             iter_offset = ((threadIdx.x & 31) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//         }
//         float sum_ = alpha_sum[block_delta] * __expf(last_max_alpha[block_delta] - max_);
//         float agg_sum = (i > end) ? 0.f : __expf(alpha - max_);
//         iter = num;
//         iter_offset = offset;
//         while(iter > 1) {
//             agg_sum += (2 * iter_offset + iter_offset - 1) == ((threadIdx.x & 31) << 2) ? 0.f : __shfl_sync(agg_mask, agg_sum, 2 * iter_offset + iter - 1 - (threadIdx.x & 31));
//             iter = (iter + 1) >> 1;
//             iter_offset = ((threadIdx.x & 31) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//         }
//         sum_ += agg_sum;
//         float s = ((sum_ > 0.f) ? (__expf(last_max_alpha[block_delta] - max_) * (alpha_sum[block_delta] + 1e-16f) / (sum_ + 1e-16f)) : 0.f);
//         float w = (i >= end) ? 0.f : __expf(alpha - max_) / (sum_ + 1e-16f);
//         for (int j = 0; j < 16; j++) {
//             float tmp = s * feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 17 + j];
//             float agg_tmp = w * temp[j];
//             iter = num;
//             iter_offset = offset;
//             while(iter > 1) {
//                 agg_tmp += (2 * iter_offset + iter_offset - 1) == ((threadIdx.x & 31) << 2) ? 0.f : __shfl_sync(agg_mask, agg_tmp, 2 * iter_offset + iter - 1 - (threadIdx.x & 31));
//                 iter = (iter + 1) >> 1;
//                 iter_offset = ((threadIdx.x & 31) - iter_offset) >= iter ? (iter_offset + iter - 1) : iter_offset;
//             }
//             feat[block_delta * (out_feats + n) + (threadIdx.x & (n - 1)) * 17 + j] = agg_tmp + tmp;
//         }
//         max_alpha[block_delta] = max_;
//         last_max_alpha[block_delta] = max_;
//         alpha_sum[block_delta] = sum_;
//         __syncwarp();
//     }
//     if (id < total_size) {
//         debug(feat + threadIdx.x * 17, output + edge_index_1[out[edge_begin]] * num_heads * out_feats + ((threadIdx.x / n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 16, 16);
//     }
// }


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
    auto feats = torch::empty({size[0], out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    auto output = torch::empty({size[0], num_heads * out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    int grid_num = (size[0] + TILESIZE_X - 1) / TILESIZE_X;
    dim3 BlockDim(256, 1, 1);
    dim3 GridDim(grid_num, 1, 1);
    // auto time = Timer();
    // time.tik();
    
    linear<<<GridDim, BlockDim>>>(
        lin_weight.data_ptr<float>(), 
        lin_bias.data_ptr<float>(), 
        x.data_ptr<float>(), 
        feats.data_ptr<float>(), 
        size[0], 
        size[1], 
        false);
    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "linear time: " << time.get_time() << std::endl;

    auto total_size = counts.sizes().vec();
    int block_node_num = 256 * 8 / (num_heads * out_feats);
    grid_num = (total_size[0] + block_node_num - 2) / block_node_num;
    GridDim = dim3(grid_num, 1, 1);
    // time.tik();
    cudaFuncSetAttribute(gat, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    // gat<<<GridDim, BlockDim, ((2 + block_node_num / 8 * 9) * num_heads * out_feats + 5 * num_heads * block_node_num) * sizeof(float)>>>(
    //     edge_index_0.data_ptr<int>(), 
    //     edge_index_1.data_ptr<int>(),
    //     counts.data_ptr<int>(),
    //     out.data_ptr<int>(),
    //     feats.data_ptr<float>(),
    //     output.data_ptr<float>(),
        
    //     att_i.data_ptr<float>(),
    //     att_j.data_ptr<float>(),
    //     num_heads,
    //     out_feats,
    //     total_size[0]-1,
    //     edge_len);
    gat<<<GridDim, BlockDim, (2 * num_heads * out_feats) * sizeof(float)>>>(
        edge_index_0.data_ptr<int>(), 
        edge_index_1.data_ptr<int>(),
        counts.data_ptr<int>(),
        out.data_ptr<int>(),
        feats.data_ptr<float>(),
        output.data_ptr<float>(),
        
        att_i.data_ptr<float>(),
        att_j.data_ptr<float>(),
        num_heads,
        out_feats,
        total_size[0]-1,
        edge_len);
    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "gat time: " << time.get_time() << std::endl;
    // grid_num = (total_size[0] + 62) / 64;
    // GridDim = dim3(grid_num, 1, 1);
    // BlockDim = dim3(64, 1, 1);
    // cudaFuncSetAttribute(gat, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    // gat<64><<<GridDim, BlockDim, (2 * num_heads * out_feats + 7 * 64) * sizeof(float)>>>(
    //     edge_index_0.data_ptr<int>(), 
    //     edge_index_1.data_ptr<int>(),
    //     counts.data_ptr<int>(),
    //     out.data_ptr<int>(),
    //     feats.data_ptr<float>(),
    //     output.data_ptr<float>(),
        
    //     att_i.data_ptr<float>(),
    //     att_j.data_ptr<float>(),
    //     num_heads,
    //     total_size[0]-1,
    //     edge_len);

    return output;  
}
