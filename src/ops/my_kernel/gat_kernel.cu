#include "gat.cuh"
#include "assert.h"

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float leaky_relu(float x) {
    return x - 0.99f * min(0.f, x);
}

__global__ void gat_kernel(
    const int* __restrict__ RowWindowOffset,
    const int64_t* __restrict__ TCOffset,
    const uint8_t* __restrict__ BlockMask,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ node_idx,
    const float* __restrict__ alphai,
    const float* __restrict__ alphaj,
    const float* __restrict__ feat,
    float* output,
    const int block_high,
    const int block_width,
    const int node_len
) {
    const int bid = blockIdx.x;
    const int warp_id = (threadIdx.x >> 5);
    const int tid = (threadIdx.x & 31);
    const int64_t e_start = __ldg(&TCOffset[bid]);
    const int64_t e_end = __ldg(&TCOffset[bid + 1]);
    const int block_offset = __ldg(&RowWindowOffset[bid]);
    if (e_start == e_end) return;
    const int iter = (e_end - e_start + block_width - 1) / block_width;
    const int dense_rowid = (threadIdx.x >> 4);
    const int dense_colid = (threadIdx.x & 15);
    const int shuffled_dense_colid = ((dense_rowid + dense_colid) & 15);
    const int sparse_rowid = (threadIdx.x >> 3);
    const int sparse_colid = (threadIdx.x & 7);
    const int shuffled_sparse_colid = ((sparse_rowid + sparse_colid) & 7);
    const int warp_row = (tid >> 2);
    const int warp_col = (tid & 3);

    __shared__ float dense_X[2][8][64];
    __shared__ float sparse_A[2][16][8];
    __shared__ float D[16][64];
    __shared__ float softmax[2][16][4];

    const float alpha_i = __ldg(&alphai[node_idx[min(bid * 16 + sparse_rowid, node_len - 1)]]);
    softmax[0][threadIdx.x & 15][0] = std::numeric_limits<float>::lowest();
    softmax[0][threadIdx.x & 15][1] = 0.f;

    //load dense_X(1/2)
    int cur_addr = __cvta_generic_to_shared(&dense_X[0][0][0]) + (dense_rowid * 64 + shuffled_dense_colid * 4) * sizeof(float);
    int dense_row_id = SparseAToX[min(e_start + (int64_t)dense_rowid, e_end - 1)];
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[dense_row_id * 64 + dense_colid * 4]));
    asm volatile("cp.async.commit_group;\n"::);
    //calculate sparseA(1/2)
    {
        float alpha_j = __ldg(&alphaj[SparseAToX[min(e_start + (int64_t)sparse_colid, e_end - 1)]]);
        uint8_t mask = __ldg(&BlockMask[block_offset * 16 + sparse_rowid]);
        bool is_valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
        float alpha = is_valid ? leaky_relu(alpha_i + alpha_j) : std::numeric_limits<float>::lowest();
        float alpha_max = alpha;
        for (int i = 1; i < 8; i *= 2) {
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * i - 1));
        }
        alpha_max = max(alpha_max, softmax[0][sparse_rowid][0]);
        softmax[0][sparse_rowid][2] = alpha_max;
        float alpha_sum = static_cast<float>(is_valid) * __expf(alpha - alpha_max);
        float upper = alpha_sum;
        for (int i = 1; i < 8; i *= 2) {
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * i - 1);
        }
        // bool is_init = softmax[0][sparse_rowid][0] == std::numeric_limits<float>::lowest();
        // float s = max(static_cast<float>(is_init), );
        alpha_sum = alpha_sum + softmax[0][sparse_rowid][1] * __expf(softmax[0][sparse_rowid][0] - alpha_max);
        softmax[0][sparse_rowid][3] = alpha_sum;
        sparse_A[0][sparse_rowid][shuffled_sparse_colid] = upper / (alpha_sum + 1e-16f);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    float frag_D[8]={0.f}; // (warp_row + (i & 2) * 4, warp_col + (i & 1) * 4 + (i & 4) * 2)
    for (int i = 0; i < iter - 1; i++) {
        //load dense_X(2/2)
        int cur_addr = __cvta_generic_to_shared(&dense_X[(i+1)&1][0][0]) + (dense_rowid * 64 + shuffled_dense_colid * 4) * sizeof(float);
        int dense_row_id = SparseAToX[min(e_start + (int64_t)dense_rowid + (int64_t)((i + 1) * block_width), e_end - 1)];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[dense_row_id * 64 + dense_colid * 4]));
        asm volatile("cp.async.commit_group;\n"::);
        //calculate sparseA(2/2)
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            float alpha_j = __ldg(&alphaj[SparseAToX[min(e_start + (int64_t)sparse_colid + (int64_t)((i + 1) * block_width), e_end - 1)]]);
            uint8_t mask = BlockMask[(block_offset + i + 1) * 16 + sparse_rowid];
            bool is_valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = is_valid ? leaky_relu(alpha_i + alpha_j) : std::numeric_limits<float>::lowest();
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
            }
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(is_valid) * __expf(alpha - alpha_max);
            float upper = alpha_sum;
            for (int j = 1; j < 8; j *= 2) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * j - 1);
            }
            // bool is_init = softmax[cur_rid][sparse_rowid][0] == std::numeric_limits<float>::lowest();
            // float s = max(static_cast<float>(is_init), );
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1] * __expf(softmax[cur_rid][sparse_rowid][0] - alpha_max);
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_A[cur_rid][sparse_rowid][shuffled_sparse_colid] = upper / (alpha_sum + 1e-16f);
        }
        //spmm
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][warp_row][0]), FLOAT4(softmax[i&1][warp_row + 8][0])};
            float s[2];
            for (int j = 0; j < 2; j++) {
                float is_init = static_cast<float>(softmax_param[j].y > 0.f);
                s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);

            }
            for (int j = 0; j < 8; j++) {
                int id = (j & 2) >> 1;
                frag_D[j] *= s[id];
            }
            uint32_t frag_A[4], frag_B[4];
            for (int j = 0; j < 4; j++) {
                int s_row = warp_row + (j & 1) * 8;
                int s_col = (warp_col + warp_row + (j & 2) * 2) & 7;
                int d_row = warp_col + (j & 2) * 2;
                int d_col = (warp_id * 16 + warp_row + (j & 1) * 8 + d_row * 4) & 63;

                // int p = (i > 0) ? (i % 3) : 0;
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_A[i & 1][s_row][s_col]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[i & 1][d_row][d_col]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
                    "r"(frag_B[0]), "r"(frag_B[2]),
                    "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
                    "r"(frag_B[1]), "r"(frag_B[3]),
                    "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    //spmm(-1)
    int i = iter - 1;
    {
        float4 softmax_param[2] = {FLOAT4(softmax[i&1][warp_row][0]), FLOAT4(softmax[i&1][warp_row + 8][0])};
        float s[2];
        for (int j = 0; j < 2; j++) {
            float is_init = static_cast<float>(softmax_param[j].y > 0.f);
            s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);

        }
        for (int j = 0; j < 8; j++) {
            int id = (j & 2) >> 1;
            frag_D[j] *= s[id];
        }
        uint32_t frag_A[4], frag_B[4];
        for (int j = 0; j < 4; j++) {
            int s_row = warp_row + (j & 1) * 8;
            int s_col = (warp_col + warp_row + (j & 2) * 2) & 7;
            int d_row = warp_col + (j & 2) * 2;
            int d_col = (warp_id * 16 + warp_row + (j & 1) * 8 + d_row * 4) & 63;

            // int p = (i > 0) ? (i % 3) : 0;
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_A[i & 1][s_row][s_col]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[i & 1][d_row][d_col]));
        }

        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
              "r"(frag_B[0]), "r"(frag_B[2]),
              "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
            : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
              "r"(frag_B[1]), "r"(frag_B[3]),
              "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
    }
    // store result
    for (int i = 0; i < 8; i++){
        int row = warp_row + 4 * (i & 2);
        int col = (warp_col * 2 + (i & 1) + 2 * (i & 4) + warp_id * 16 + 4 * warp_row) & 63;
        D[row][col] = frag_D[i];
    }
    __syncthreads();
    for (int i = 0; i < 2; i++)
        if ((bid * block_high + dense_rowid + i * 8) < node_len)
            FLOAT4(output[node_idx[(bid * block_high + dense_rowid + i * 8)] * 64 + dense_colid * 4]) = FLOAT4(D[dense_rowid + i * 8][shuffled_dense_colid * 4]);

}

at::Tensor GAT(
    at::Tensor feature,
    at::Tensor RowWindowOffset,
    at::Tensor TCOffset,
    at::Tensor BlockMask,
    at::Tensor SparseAToX,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    at::Tensor node_idx,
    int num_heads,
    int out_feats
) {
    int num_nodes = feature.size(0);
    auto feats = torch::mm(feature, lin_weight.t());
    auto alpha_i = torch::matmul(feats, att_i.squeeze());
    auto alpha_j = torch::matmul(feats, att_j.squeeze());
    auto output = at::empty({num_nodes, out_feats}, feature.options()).fill_(0.f);

    int threads = 128;
    int blocks = (num_nodes + 15) / 16;
    gat_kernel<<<blocks, threads>>>(
        RowWindowOffset.data_ptr<int>(),
        TCOffset.data_ptr<int64_t>(),
        BlockMask.data_ptr<uint8_t>(),
        SparseAToX.data_ptr<int>(),
        node_idx.data_ptr<int>(),
        alpha_i.data_ptr<float>(),
        alpha_j.data_ptr<float>(),
        feats.data_ptr<float>(),
        output.data_ptr<float>(),
        16, 8, num_nodes);
    return output;
}