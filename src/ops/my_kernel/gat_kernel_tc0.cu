#include "gat.cuh"
#include "assert.h"
#include "time.cuh"

#define TILESIZE_X 64
#define TILESIZE_Y 64
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float leaky_relu(float x) {
    return x - 0.99f * min(0.f, x);
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
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[min(node_len - 1, block_offset + (threadIdx.x / BK) + i * (blockDim.x / BK)) * K + (threadIdx.x & (BK - 1))]));
        cur_addr = __cvta_generic_to_shared(b_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1))) * sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[((threadIdx.x / BK) + i * (blockDim.x / BK)) * K + (threadIdx.x & (BK - 1))]));
    }

    int cur_addr = __cvta_generic_to_shared(att_i) + (threadIdx.x & 63) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&att_i_[threadIdx.x & 63]));

    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    for (int k = 1; k < ((K + 15) >> 4); k++) {
        for (int i = 0; i < (BK / 8); i++) {
            int cur_addr = __cvta_generic_to_shared(a_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1)) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&X[min(node_len - 1, block_offset + (threadIdx.x / BK) + i * (blockDim.x / BK)) * K + min(k * BK + (threadIdx.x & (BK - 1)), K - 1)]));
            cur_addr = __cvta_generic_to_shared(b_tmp) + (((threadIdx.x / BK) + i * (blockDim.x / BK)) * (2 * BK + 1) + (threadIdx.x & (BK - 1)) + (k & 1) * BK) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&weight[((threadIdx.x / BK) + i * (blockDim.x / BK)) * K + min(k * BK + (threadIdx.x & (BK - 1)), K - 1)]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        
        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < 8; j++)
                c_tmp[j] += a_tmp[(threadIdx.x >> 3) * (2 * BK + 1) + ((k - 1) & 1) * BK + i] * b_tmp[((threadIdx.x & 7) + j * 8) * (2 * BK + 1) + ((k - 1) & 1) * BK + i];
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();

    }
    int k = ((K + 15) >> 4);
    for (int i = 0; i < BK; i++) {
        for (int j = 0; j < 8; j++)
            c_tmp[j] += (((k - 1) * BK + i) < K ? 1.f : 0.f) * a_tmp[(threadIdx.x >> 3) * (2 * BK + 1) + ((k - 1) & 1) * BK + i] * b_tmp[((threadIdx.x & 7) + j * 8) * (2 * BK + 1) + ((k - 1) & 1) * BK + i];
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
    const int* __restrict__ RowWindowOffset,
    const int64_t* __restrict__ TCOffset,
    const uint8_t* __restrict__ BlockMask,
    const int* __restrict__ SparseAToX,
    const int* __restrict__ node_idx,
    const float* __restrict__ alphai,
    float* feat,
    float* att_j_,
    float* output,
    const int block_high,
    const int block_width,
    const int node_len
) {
    //warp_num = 4, stage_num = 3, 3 * data_buffer, 2 * SparseA
    
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

    // __shared__ float alpha_i[16];
    // __shared__ float att_j[64];
    __shared__ float dense_X[3][8][64];
    __shared__ float sparse_A[2][16][8];
    __shared__ float D[16][64];
    __shared__ float alpha_j[16];
    __shared__ float softmax[2][16][4]; // | last_alpha_max | last_alpha_sum | alpha_max | alpha_sum |

    //load alpha_i, att_j(once), init softmax
    const float alpha_i = __ldg(&alphai[node_idx[min(bid * 16 + sparse_rowid, node_len - 1)]]);
    // int cur_addr = __cvta_generic_to_shared(alpha_i) + (threadIdx.x & 15) * sizeof(float);
    // asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(cur_addr), "l"(&alphai[min(bid * 16 + (threadIdx.x & 15), node_len - 1)]));
    // cur_addr = __cvta_generic_to_shared(att_j) + dense_colid * 4 * sizeof(float);
    // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&att_j_[dense_colid * 4]));
    softmax[0][threadIdx.x & 15][0] = std::numeric_limits<float>::lowest();
    softmax[0][threadIdx.x & 15][1] = 0.f;
    
    //load dense_X(1/3)
    // cur_addr = __cvta_generic_to_shared(&dense_X[0][0][0]) + (dense_rowid * 64 + shuffled_dense_colid * 4) * sizeof(float);
    // int dense_row_id = SparseAToX[min(e_start + (int64_t)dense_rowid, e_end - 1)];
    // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[dense_row_id * 64 + dense_colid * 4]));
    // asm volatile("cp.async.commit_group;\n"::);

    // asm volatile("cp.async.wait_group 0;\n"::);
    // __syncthreads();

    //calculate alpha_j, load dense_X(1/3)
    int dense_row_id = __ldg(&SparseAToX[min(e_start + (uint32_t)dense_rowid, e_end - 1)]);
    float4 tmp = FLOAT4(feat[dense_row_id * 64 + dense_colid * 4]);
    // float4 tmp = FLOAT4(dense_X[0][dense_rowid][shuffled_dense_colid * 4]);
    float4 att_j_vec = FLOAT4(att_j_[dense_colid * 4]);
    float a_j = tmp.x * att_j_vec.x + tmp.y * att_j_vec.y + tmp.z * att_j_vec.z + tmp.w * att_j_vec.w;
    for (int i = 1; i < 16; i *= 2) {
        a_j += __shfl_xor_sync(FULL_MASK, a_j, i * 2 - 1);
    }
    alpha_j[dense_rowid] = a_j;
    FLOAT4(dense_X[0][dense_rowid][shuffled_dense_colid * 4]) = tmp; 

    // //load dense_x(2/3)
    // cur_addr = __cvta_generic_to_shared(&dense_X[1][0][0]) + threadIdx.x * 4 * sizeof(float);
    // dense_row_id = SparseAToX[min(e_start + (uint32_t)(dense_rowid + block_width), e_end - 1)];
    // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[dense_row_id * 64 + dence_colid]));
    // asm volatile("cp.async.commit_group;\n"::);
    // asm volatile("cp.async.wait_group 0;\n"::);
    // __syncthreads();

    //calculate alpha_j, load dense_X(2/3)
    dense_row_id = __ldg(&SparseAToX[min(e_start + (int64_t)(dense_rowid + block_width), e_end - 1)]);
    tmp = FLOAT4(feat[dense_row_id * 64 + dense_colid * 4]);
    // tmp = FLOAT4(dense_X[1][dense_rowid][dence_colid]);
    a_j = tmp.x * att_j_vec.x + tmp.y * att_j_vec.y + tmp.z * att_j_vec.z + tmp.w * att_j_vec.w;
    for (int i = 1; i < 16; i *= 2) {
        a_j += __shfl_xor_sync(FULL_MASK, a_j, i * 2 - 1);
    }
    alpha_j[dense_rowid + block_width] = a_j;
    FLOAT4(dense_X[1][dense_rowid][shuffled_dense_colid * 4]) = tmp;

    __syncthreads();


    //calculate sparseA
    {  
        uint8_t mask = BlockMask[block_offset * 16 + sparse_rowid];
        bool is_valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
        float alpha = is_valid ? leaky_relu(alpha_i + alpha_j[sparse_colid]) : std::numeric_limits<float>::lowest();
        float alpha_max = alpha;
        for (int i = 1; i < 8; i *= 2) {
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * i - 1));
        }
        alpha_max = max(alpha_max, softmax[0][sparse_rowid][0]);
        softmax[0][sparse_rowid][2] = alpha_max;
        float alpha_sum = static_cast<float>(is_valid) * __expf(alpha - alpha_max);
        for (int i = 1; i < 8; i *= 2) {
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * i - 1);
        }
        alpha_sum = alpha_sum + softmax[0][sparse_rowid][1] * __expf(softmax[0][sparse_rowid][0] - alpha_max);
        softmax[0][sparse_rowid][3] = alpha_sum;
        sparse_A[0][sparse_rowid][shuffled_sparse_colid] = static_cast<float>(is_valid) * __expf(alpha - alpha_max) / (alpha_sum + 1e-16f);
    }

    __syncthreads();

    float frag_D[8]; // (warp_row + (i & 2) * 4, warp_col + (i & 1) * 4 + (i & 4) * 2)

    for (int i = 0; i < iter - 2; i++) {
        // // load dense_X(3/3)
        // cur_addr = __cvta_generic_to_shared(&dense_X[(i + 2) % 3][0][0]) + threadIdx.x * 4 * sizeof(float);
        // dense_row_id = SparseAToX[min(e_start + (uint32_t)(dense_rowid + (i + 2) * block_width), e_end - 1)];
        // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&feat[dense_row_id * 64 + dence_colid]));
        // asm volatile("cp.async.commit_group;\n"::);
        // asm volatile("cp.async.wait_group 0;\n"::);
        // __syncthreads();
        // calculate alpha_j, load dense_X(3/3)
        dense_row_id = __ldg(&SparseAToX[min(e_start + (int64_t)(dense_rowid + (i + 2) * block_width), e_end - 1)]);
        tmp = FLOAT4(feat[dense_row_id * 64 + dense_colid * 4]);
        // tmp = FLOAT4(dense_X[(i + 2) % 3][dense_rowid][dence_colid]);
        a_j = tmp.x * att_j_vec.x + tmp.y * att_j_vec.y + tmp.z * att_j_vec.z + tmp.w * att_j_vec.w;
        for (int j = 1; j < 16; j *= 2) {
            a_j += __shfl_xor_sync(FULL_MASK, a_j, j * 2 - 1);
        }
        alpha_j[dense_rowid + (i & 1) * block_width] = a_j;
        FLOAT4(dense_X[(i + 2) % 3][dense_rowid][shuffled_dense_colid * 4]) = tmp;
        // calculate sparseA
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            uint8_t mask = BlockMask[(block_offset + i + 1) * 16 + sparse_rowid];
            bool is_valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = is_valid ? leaky_relu(alpha_i + alpha_j[sparse_colid + cur_rid * block_width]) : std::numeric_limits<float>::lowest();
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
            }
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(is_valid) * __expf(alpha - alpha_max);
            for (int j = 1; j < 8; j *= 2) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * j - 1);
            }
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1];
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_A[cur_rid][sparse_rowid][shuffled_sparse_colid] = static_cast<float>(is_valid) * __expf(alpha - alpha_max) / (alpha_sum + 1e-16f);
        }
        // spmm
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][warp_row][0]), FLOAT4(softmax[i&1][warp_row + 8][0])};
            for (int j = 0; j < 8; j++) {
                int id = (j & 2) >> 1;
                float is_init = softmax_param[id].y > 0.f;
                float s = is_init * __expf(softmax_param[id].x - softmax_param[id].z) * (softmax_param[id].y + 1e-16f) / (softmax_param[id].w + 1e-16f);
                frag_D[j] *= s;
            }
            uint32_t frag_A[4], frag_B[4];
            for (int j = 0; j < 4; j++) {
                int s_row = warp_row + (j & 1) * 8;
                int s_col = (warp_col + warp_row + (j & 2) * 2) & 7;
                int d_row = warp_col + (j & 2) * 2;
                int d_col = (warp_id * 16 + warp_row + (j & 1) * 8 + d_row * 4) & 63;

                // int p = (i > 0) ? (i % 3) : 0;
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_A[i & 1][s_row][s_col]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[(i % 3)][d_row][d_col]));
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

            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[0]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
            //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[1]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[2]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
            //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[3]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
        }
        __syncthreads();
    }
    int i = iter - 2;
    if (i >= 0) {
        // calculate sparse
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            uint8_t mask = BlockMask[(block_offset + i + 1) * 16 + sparse_rowid];
            bool is_valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = is_valid ? leaky_relu(alpha_i + alpha_j[sparse_colid + cur_rid * block_width]) : std::numeric_limits<float>::lowest();
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
            }
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(is_valid) * __expf(alpha - alpha_max);
            for (int j = 1; j < 8; j *= 2) {
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * i - 1);
            }
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1];
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_A[cur_rid][sparse_rowid][shuffled_sparse_colid] = static_cast<float>(is_valid) * __expf(alpha - alpha_max) / (alpha_sum + 1e-16f);
        }
        // spmm(-2)
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][warp_row][0]), FLOAT4(softmax[i&1][warp_row + 8][0])};
            for (int j = 0; j < 8; j++) {
                int id = (j & 2) >> 1;
                float is_init = softmax_param[id].y > 0.f;
                float s = is_init * __expf(softmax_param[id].x - softmax_param[id].z) * (softmax_param[id].y + 1e-16f) / (softmax_param[id].w + 1e-16f);
                frag_D[j] *= s;
            }
            uint32_t frag_A[4], frag_B[4];
            for (int j = 0; j < 4; j++) {
                int s_row = warp_row + (j & 1) * 8;
                int s_col = (warp_col + warp_row + (j & 2) * 2) & 7;
                int d_row = warp_col + (j & 2) * 2;
                int d_col = (warp_id * 16 + warp_row + (j & 1) * 8 + d_row * 4) & 63;

                // int p = (i > 0) ? (i % 3) : 0;
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_A[i & 1][s_row][s_col]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[(i % 3)][d_row][d_col]));
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

            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[0]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
            //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[1]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[2]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
            // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
            //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[3]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
        }
        __syncthreads();
    }
    //spmm(-1)
    i = iter - 1;
    {
        float4 softmax_param[2] = {FLOAT4(softmax[i&1][warp_row][0]), FLOAT4(softmax[i&1][warp_row + 8][0])};
        for (int j = 0; j < 8; j++) {
            int id = (j & 2) >> 1;
            float is_init = softmax_param[id].y > 0.f;
            float s = is_init * __expf(softmax_param[id].x - softmax_param[id].z) * (softmax_param[id].y + 1e-16f) / (softmax_param[id].w + 1e-16f);
            frag_D[j] *= s;
        }
        uint32_t frag_A[4], frag_B[4];
        for (int j = 0; j < 4; j++) {
            int s_row = warp_row + (j & 1) * 8;
            int s_col = (warp_col + warp_row + (j & 2) * 2) & 7;
            int d_row = warp_col + (j & 2) * 2;
            int d_col = (warp_id * 16 + warp_row + (j & 1) * 8 + d_row * 4) & 63;

            // int p = (i > 0) ? (i % 3) : 0;
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_A[i & 1][s_row][s_col]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[(i % 3)][d_row][d_col]));
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

        // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
        //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[0]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
        //     : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_B[1]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
        // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        //     : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
        //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[2]), "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        //     : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
        //     : "r"(frag_A[2]), "r"(frag_A[3]), "r"(frag_B[3]), "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7]));
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
    int K = feature.size(1);
    auto output = at::empty({num_nodes, out_feats}, feature.options());
    float* alpha_i, *feats;
    cudaMalloc(&feats, num_nodes * out_feats * sizeof(float));
    cudaMalloc(&alpha_i, num_nodes * sizeof(float));

    // auto time = Timer();
    // time.tik();

    int threads = 512;
    int blocks = (num_nodes + 63) / 64;
    linear_reduce<<<blocks, threads>>>(
        feature.data_ptr<float>(),
        lin_weight.data_ptr<float>(),
        att_i.data_ptr<float>(),
        feats,
        alpha_i,
        num_nodes,
        K
    );

    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "linear time: " << time.get_time() << std::endl;

    // time.tik();

    threads = 128;
    blocks = (num_nodes + 15) / 16;
    gat_kernel<<<blocks, threads>>>(
        RowWindowOffset.data_ptr<int>(),
        TCOffset.data_ptr<int64_t>(),
        BlockMask.data_ptr<uint8_t>(),
        SparseAToX.data_ptr<int>(),
        node_idx.data_ptr<int>(),
        alpha_i,
        feats,
        att_j.data_ptr<float>(),
        output.data_ptr<float>(),
        16,
        8,
        num_nodes
    );

    // cudaDeviceSynchronize();
    // time.tok();
    // std::cout << "gat time: " << time.get_time() << std::endl;

    cudaFree(alpha_i);
    cudaFree(feats);
    return output;
}