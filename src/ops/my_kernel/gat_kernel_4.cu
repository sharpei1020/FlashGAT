#include "gat.cuh"
#include "assert.h"
#include "time.cuh"

#define TILESIZE_X 64
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__device__ static inline void atomicMax(float *address, const float val) {
    unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
    unsigned int old = *address_as_ui;                       // NOLINT
    unsigned int assumed;                                    // NOLINT
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);  // NOLINT
}

__device__ static inline void copy(float* src, float* dst, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] = src[(i / 32) + i];
    }
}

__global__ void linear_reduce(
    const float* X,
    const float* weight,
    const float* att_i_,
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
        for (int i = 0; i < (BK / 8); i++) {
            int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1)) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[min(node_len - 1, block_offset + (threadIdx.x >> 5) + i * 16) * K + min(k * BK + (threadIdx.x & (BK - 1)), K - 1)]));
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
    const int* offset,
    const int* out,
    const float* feats,
    const float* alphai,
    const float* att_j_,
    float* output,
    int group,
    int num_heads,
    int out_feats,
    int num_nodes,
    int edge_len
) {
    extern __shared__ float param[];
    int block_offset = blockIdx.x * group;
    int block_start = block_offset > 0 ? offset[block_offset - 1] : 0;
    int block_end = offset[min(block_offset + group, edge_len) - 1];
    if ((block_end - block_start) == 0) return;
    float* att_j = param;
    int cur_addr = __cvta_generic_to_shared(att_j) + threadIdx.x * 2 * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&att_j_[threadIdx.x * 2]));
    float* alpha_init = att_j + 64;
    cur_addr = __cvta_generic_to_shared(alpha_init) + (threadIdx.x % group) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&alphai[min(block_offset + (threadIdx.x % group), num_nodes - 1)]));
    float* max_alpha = alpha_init + group;
    float* alpha_sum = max_alpha + group * num_heads;
    float* last_max_alpha = alpha_sum + group * num_heads;
    float* last_alpha_sum = last_max_alpha + group * num_heads;
    float* out_feat = last_alpha_sum + group * num_heads;
    float* feat = out_feat + group * (out_feats / 32 * 33);
    int k = 256 / out_feats;
    for (int i = 0; i < k * (out_feats / 32); i++) {
        cur_addr = __cvta_generic_to_shared(feat) + (i * 33 + threadIdx.x) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feats[edge_index_0[out[block_start + (i * 32 / out_feats)]] * out_feats + (i % (out_feats / 32)) * 32 + threadIdx.x]));
    }
    max_alpha[threadIdx.x%(group*num_heads)] = std::numeric_limits<float>::lowest();
    alpha_sum[threadIdx.x%(group*num_heads)] = 0.f;
    last_max_alpha[threadIdx.x%(group*num_heads)] = 0.f;
    last_alpha_sum[threadIdx.x%(group*num_heads)] = 0.f;
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncwarp();
    int idx = threadIdx.x * 8 / out_feats;
    for (int l = 1; l < (block_end - block_start + k - 1) / k; l++) {
        for (int i = 0; i < k * (out_feats / 32); i++) {
            cur_addr = __cvta_generic_to_shared(feat) + ((l & 1) * k * out_feats / 32 * 33 + (i * 33 + threadIdx.x)) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&feats[edge_index_0[out[min(block_start + (i * 32 / out_feats) + l * k, block_end - 1)]] * out_feats + (i % (out_feats / 32)) * 32 + threadIdx.x]));
        }
        asm volatile("cp.async.commit_group;\n"::); 
        int group_delta = edge_index_1[out[min(idx + (l - 1) * k + block_start, block_end - 1)]] % group;
        int param_delta = group_delta * num_heads + (threadIdx.x % num_heads);
        int feat_delta = group_delta * out_feats / 32 * 33 + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8;
        float alpha = 0.f;
        for (int i = 0; i < 8; i++)
            alpha += att_j[(threadIdx.x % (out_feats / 8)) * 8 + i] * feats[idx * (out_feats / 32 * 33) + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8 + i];
        for (int i = (out_feats / (8 * num_heads)); i > 1; i >>= 1)
            alpha += __shfl_xor_sync(FULL_MASK, alpha, (i - 1) * num_heads);
        alpha += alpha_init[group_delta];
        atomicMax(&max_alpha[param_delta], alpha);
        float a_s = alpha_sum[param_delta];
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncwarp();
        a_s *= __expf(last_max_alpha[param_delta] - max_alpha[param_delta]);
        alpha_sum[param_delta] = a_s;
        __syncwarp();
        a_s = __expf(alpha - max_alpha[param_delta]);
        atomicAdd(&alpha_sum[param_delta], a_s);
        __syncwarp();
        a_s = ((alpha_sum[param_delta] > 0.f) ? 
            (__expf(last_max_alpha[param_delta] - max_alpha[param_delta]) * (last_alpha_sum[param_delta] + 1e-16f) / (alpha_sum[param_delta] + 1e-16f)) : 0.f);
        float w = __expf(alpha - max_alpha[param_delta]) / (alpha_sum[param_delta] + 1e-16f);
        u_int32_t mask = __match_any_sync(FULL_MASK, feat_delta);
        for (int i = 0; i < 8; i++) {
            float f = out_feat[feat_delta + i] * a_s;
            float val = w * feats[idx * (out_feats / 32 * 33) + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8 + i];
            f += __uint_as_float(__reduce_add_sync(mask, __float_as_uint(val)));
            out_feat[feat_delta + i] = f;
        }
        last_max_alpha[param_delta] = max_alpha[param_delta];
        last_alpha_sum[param_delta] = alpha_sum[param_delta];
        __syncwarp();            
    }
    int l = (block_end - block_start + k - 1) / k;
    int group_delta = edge_index_1[out[min(idx + (l - 1) * k + block_start, block_end - 1)]] % group;
    int param_delta = group_delta * num_heads + (threadIdx.x % num_heads);
    int feat_delta = group_delta * out_feats / 32 * 33 + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8;
    float alpha = 0.f;
    for (int i = 0; i < 8; i++)
        alpha += att_j[(threadIdx.x % (out_feats / 8)) * 8 + i] * feats[idx * (out_feats / 32 * 33) + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8 + i];
    for (int i = (out_feats / (8 * num_heads)); i > 1; i >>= 1)
        alpha += __shfl_xor_sync(FULL_MASK, alpha, (i - 1) * num_heads);
    alpha += alpha_init[group_delta];
    atomicMax(&max_alpha[param_delta], alpha);
    float a_s = alpha_sum[param_delta];
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncwarp();
    a_s *= __expf(last_max_alpha[param_delta] - max_alpha[param_delta]);
    alpha_sum[param_delta] = a_s;
    __syncwarp();
    a_s = (idx + (l - 1) * k + block_start < block_end ? 1.f : 0.f) * __expf(alpha - max_alpha[param_delta]);
    atomicAdd(&alpha_sum[param_delta], a_s);
    __syncwarp();
    a_s = ((alpha_sum[param_delta] > 0.f) ? 
        (__expf(last_max_alpha[param_delta] - max_alpha[param_delta]) * (last_alpha_sum[param_delta] + 1e-16f) / (alpha_sum[param_delta] + 1e-16f)) : 0.f);
    float w = __expf(alpha - max_alpha[param_delta]) / (alpha_sum[param_delta] + 1e-16f);
    u_int32_t mask = __match_any_sync(FULL_MASK, feat_delta);
    for (int i = 0; i < 8; i++) {
        float f = out_feat[feat_delta + i] * a_s;
        float val = (idx + (l - 1) * k + block_start) < block_end ? w * feats[idx * (out_feats / 32 * 33) + (threadIdx.x % (out_feats / 8)) / 4 * 33 + (threadIdx.x & 3) * 8 + i] : 0.f;
        f += __uint_as_float(__reduce_add_sync(mask, __float_as_uint(val)));
        out_feat[feat_delta + i] = f;
    }
    __syncwarp();
    copy(out_feat + (threadIdx.x % group) * out_feats / 32 * 33, output + min(block_offset + (threadIdx.x % group), num_nodes - 1) * out_feats, out_feats);
}

at::Tensor GAT(
    at::Tensor x,
    at::Tensor edge_index_0,
    at::Tensor edge_index_1,
    at::Tensor offsets,
    at::Tensor out,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    int num_heads,
    int out_feats,
    int group
) {
    auto size = x.sizes().vec();
    int edge_len = edge_index_0.sizes().vec()[0];
    auto output = torch::empty({size[0], out_feats}, device(torch::kCUDA)).to(torch::kFloat32);
    float *node_feats, *node_alpha;
    cudaMalloc(&node_feats, size[0] * out_feats * sizeof(float));
    cudaMalloc(&node_alpha, size[0] * sizeof(float));
    int grid = (size[0] + 63) / 64;
    linear_reduce<<<grid, 512>>>(
        x.data_ptr<float>(),
        lin_weight.data_ptr<float>(),
        att_i.data_ptr<float>(),
        node_feats,
        node_alpha,
        size[0],
        size[1]
    );

    grid = (size[0] + group - 1) / group;
    cudaFuncSetAttribute(gat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (64 + group * (1 + 4 * num_heads + 33 * (out_feats / 32)) + 264) * sizeof(float));
    gat_kernel<<<grid, 32, (64 + group * (1 + 4 * num_heads + 33 * (out_feats / 32)) + 264) * sizeof(float)>>>(
        edge_index_0.data_ptr<int>(),
        edge_index_1.data_ptr<int>(),
        offsets.data_ptr<int>(),
        out.data_ptr<int>(),
        node_feats,
        node_alpha,
        att_j.data_ptr<float>(),
        output.data_ptr<float>(),
        group,
        num_heads,
        out_feats,
        size[0],
        edge_len);

    cudaFree(node_feats);
    cudaFree(node_alpha);
    return output;
}