#include "gat.cuh"
#include "assert.h"

#define TILESIZE_X 64
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__global__ void scatter_linear(
    const float* X,
    const float* weight,
    const float* att_j_,
    const int* edge_index_0,
    const int* out,
    float* edge_feat,
    float* edge_alpha,
    int edge_len,
    int K) {
    
    const int BK = 32;
    __shared__ float a_tmp[TILESIZE_X * (2 * BK + 1)], b_tmp[TILESIZE_Y * (2 * BK + 1)], att_j[64];
    float c_tmp[8], alphaj = 0.f;
    for (int i = 0; i < 8; i++) {
        c_tmp[i] = 0.f;
    }
    int block_offset = blockIdx.x * TILESIZE_X;
    for (int i = 0; i < (BK / 8); i++) {
        int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1))) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[edge_index_0[out[min(edge_len - 1, block_offset + (threadIdx.x / BK) + i * (blockDim.x / BK))]] * K + (threadIdx.x & 31)]));
        cur_addr = __cvta_generic_to_shared(b_tmp) + ((threadIdx.x & 63) * (2 * BK + 1) + (threadIdx.x >> 6) + i * 8) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[((threadIdx.x >> 6) + i * 8) * 64 + (threadIdx.x & 63)]));
    }

    int cur_addr = __cvta_generic_to_shared(att_j) + (threadIdx.x & 63) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&att_j_[threadIdx.x & 63]));

    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    for (int k = 1; k < (((K - 1) >> 5) + 1); k++) {
        for (int i = 0; i < 4; i++) {
            int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1)) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[edge_index_0[min(edge_len - 1, block_offset + (threadIdx.x / BK) + i * (blockDim.x / BK))] * K + min(k * BK + (threadIdx.x & (BK - 1)), K - 1)]));
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
        edge_feat[min(edge_len - 1, block_offset + (threadIdx.x >> 3)) * 64 + (threadIdx.x & 7) +  i * 8] = c_tmp[i];
        alphaj += c_tmp[i] * att_j[(threadIdx.x & 7) + i * 8];
    }
    int i = 8;
    while (i > 1) {
        alphaj += __shfl_xor_sync(FULL_MASK, alphaj,  i - 1);
        i >>= 1;
    }
    edge_alpha[min(edge_len - 1, block_offset + (threadIdx.x >> 3))] = alphaj;

    }

__global__ void linear_reduce(
    const float* X,
    const float* weight,
    const float* att_i_,
    float* node_alpha,
    int node_len,
    int K) {
    
    const int BK = 32;
    __shared__ float a_tmp[TILESIZE_X * (2 * BK + 1)], b_tmp[TILESIZE_Y * (2 * BK + 1)], att_i[64];
    float c_tmp[8], alphai = 0.f;
    for (int i = 0; i < 8; i++) {
        c_tmp[i] = 0.f;
    }
    int block_offset = blockIdx.x * TILESIZE_X;
    for (int i = 0; i < 4; i++) {
        int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x >> 5) + i * 16) * (2 * BK + 1) + (threadIdx.x & 31)) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[min(node_len - 1, block_offset + (threadIdx.x >> 5) + i * 16) * K + (threadIdx.x & 31)]));
        cur_addr = __cvta_generic_to_shared(b_tmp) + ((threadIdx.x & 63) * (2 * BK + 1) + (threadIdx.x >> 6) + i * 8) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[((threadIdx.x >> 6) + i * 8) * 64 + (threadIdx.x & 63)]));
    }
    int cur_addr = __cvta_generic_to_shared(att_i) + (threadIdx.x & 63) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&att_i_[threadIdx.x & 63]));

    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    for (int k = 1; k < (((K - 1) >> 5) + 1); k++) {
        for (int i = 0; i < 4; i++) {
            int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x >> 5) + i * 16) * (2 * BK + 1) + (threadIdx.x & 31) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[min(node_len - 1, block_offset + (threadIdx.x >> 5) + i * 16) * K + min(k * BK + (threadIdx.x & 31), K - 1)]));
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
        alphai += c_tmp[i] * att_i[(threadIdx.x & 7) + i * 8];
    }
    int i = 8;
    while (i > 1) {
        alphai += __shfl_xor_sync(FULL_MASK, alphai,  i - 1);
        i >>= 1;
    }
    node_alpha[min(node_len - 1, block_offset + (threadIdx.x >> 3))] = alphai;
    }

__global__ void gat_kernel(
    int* edge_index_1,
    int* count,
    int* out,
    float* edge_feat,
    const float* edge_alpha,
    const float* node_alpha,
    float* output,

    int num_heads,
    int out_feats,
    int node_len,
    int edge_len) {
    extern __shared__ float param[];
    int n = out_feats / 8;
    float* max_alpha = param;
    float* last_max_alpha = param + (blockDim.x / n);
    float* alpha_sum = last_max_alpha + (blockDim.x / n);
    int* edge_off = (int*) (alpha_sum + (blockDim.x / n));
    float* feat = alpha_sum + 2 * (blockDim.x / n);

    float temp[8], alpha_init = 0.f;

    int id = blockIdx.x * blockDim.x / (num_heads * n) + (threadIdx.x / (num_heads * n));
    int edge_begin = count[min(id, node_len-1)];

    max_alpha[(threadIdx.x / n)] = std::numeric_limits<float>::lowest();
    last_max_alpha[(threadIdx.x / n)] = 0.f;
    alpha_sum[(threadIdx.x / n)] = 0.f;
    int map1 = edge_index_1[out[edge_begin]];

    alpha_init += node_alpha[map1];

    edge_off[threadIdx.x / n] = count[min(id+1, node_len)];
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
        int block_delta = (threadIdx.x & (blockDim.x - 32)) / n, delta = ((threadIdx.x/n) & (num_heads - 1));
        for (int j = delta; j < (32/n); j += num_heads) {
            delta += ((idx >= edge_off[block_delta + j]) ? 1 : 0) * num_heads;
        }
        block_delta += delta;
        float alpha = alpha_init + edge_alpha[idx];
        for (int j = 0; j < 2; j++) {
            FLOAT4(temp[j * 4]) = FLOAT4(edge_feat[idx * num_heads * out_feats + j * 4 + ((threadIdx.x/n) & (num_heads - 1)) * out_feats + (threadIdx.x & (n - 1)) * 8]);
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
    if (id < node_len) {
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
    torch::Tensor att_i,
    torch::Tensor att_j,
    int num_heads,
    int out_feats
) {
    auto size = x.sizes().vec();
    int edge_len = edge_index_0.sizes().vec()[0];
    auto output = torch::empty({size[0], num_heads * out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    float* edge_feat, *edge_alpha;
    cudaMalloc(&edge_feat, edge_len * num_heads * out_feats * sizeof(float));
    cudaMalloc(&edge_alpha, edge_len * sizeof(float));
    int grid = (edge_len + 63) / 64;
    scatter_linear<<<grid, 512>>>(
        x.data_ptr<float>(),
        lin_weight.data_ptr<float>(),
        att_j.data_ptr<float>(),
        edge_index_0.data_ptr<int>(),
        out.data_ptr<int>(),
        edge_feat,
        edge_alpha,
        edge_len,
        size[1]
    );
    grid = (size[0] + 63) / 64;
    float* node_alpha;
    cudaMalloc(&node_alpha, size[0] * sizeof(float));
    linear_reduce<<<grid, 512>>>(
        x.data_ptr<float>(),
        lin_weight.data_ptr<float>(),
        att_i.data_ptr<float>(),
        node_alpha,
        size[0],
        size[1]
    );
    auto total_size = counts.sizes().vec();
    int block_node_num = 128 * 8 / (num_heads * out_feats);
    grid = (total_size[0] + block_node_num - 2) / block_node_num;
    cudaFuncSetAttribute(gat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);

    gat_kernel<<<grid, 128, (128 * 9 + 4 * block_node_num * num_heads) * sizeof(float)>>>(
        edge_index_1.data_ptr<int>(),
        counts.data_ptr<int>(),
        out.data_ptr<int>(),
        edge_feat,
        edge_alpha,
        node_alpha,
        output.data_ptr<float>(),
        num_heads,
        out_feats,
        total_size[0] - 1,
        edge_len
    );
    cudaFree(edge_feat);
    cudaFree(edge_alpha);
    cudaFree(node_alpha);
    return output;
}
