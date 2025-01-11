#include "agnn.cuh"
#include "time.cuh"

#define FULL_MASK 0xffffffff

__global__ void agnn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ x_norm,
    const int* __restrict__ RowWindowOffset,
    const int64_t* __restrict__ TCOffset,
    const uint8_t* __restrict__ BlockMask,
    const int* __restrict__ SparseAToX,
    const float* __restrict__ beta,
    float* output,
    const int block_high,
    const int block_width,
    const int node_len
){
    const int bid = blockIdx.x;
    const int warp_id = (threadIdx.x >> 5);
    const int lane_id = (threadIdx.x & 31);
    const int64_t e_start = __ldg(&TCOffset[bid]);
    const int64_t e_end = __ldg(&TCOffset[bid+1]);
    const int b_start = __ldg(&RowWindowOffset[bid]);
    const int b_end = __ldg(&RowWindowOffset[bid + 1]);
    if (e_start == e_end) return;
    const int iter = b_end - b_start;
    const int dense_rowid = (threadIdx.x >> 4);
    const int dense_colid = (threadIdx.x & 15);
    const int sparse_rowid = (threadIdx.x >> 3);
    const int sparse_colid = (threadIdx.x & 7);
    const int shuffled_dense_colid = (dense_rowid * 2 + dense_colid) & 15;
    const int shuffled_sparse_colid = (sparse_rowid + sparse_colid) & 7;
    const int lane_front = lane_id >> 2;
    const int lane_back = lane_id & 3;
    const int shuffle = lane_front >> 2;
    const int spmm_shuffle = (lane_id & 4)|(lane_id >> 3);

    //buffer size : dense_X 4 / sparse_X 3 / softmax 2 / D 1
    __shared__ float dense_X[4][8][32];
    __shared__ float sparse_X[4][16][8];
    __shared__ float softmax[2][16][4];
    __shared__ float D[16][32];
    __shared__ float norm_j[2][8];

    softmax[0][sparse_rowid][0] = -3.0f;
    softmax[0][sparse_rowid][1] = 0.f;
    const float b = __ldg(&beta[0]);
    float x_norm_i[2];
    x_norm_i[0] = __ldg(&x_norm[bid * 16 + lane_front]);
    x_norm_i[1] = __ldg(&x_norm[bid * 16 + lane_front + 8]);
    // float x_norm_j[3];

    //load dense_X(1/4)
    int cur_addr = __cvta_generic_to_shared(&dense_X[0][0][0]) + (dense_rowid * 32 + shuffled_dense_colid * 2) * sizeof(float);
    int rowid = SparseAToX[min(e_start + (int64_t)dense_rowid, e_end - 1)];
    norm_j[0][dense_rowid] = x_norm[rowid];
    sparse_X[0][sparse_rowid][sparse_colid] = 0.f;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&x[rowid * 32 + dense_colid * 2]));
    cur_addr = __cvta_generic_to_shared(&D[0][0]) + (sparse_rowid * 32 + shuffled_sparse_colid * 4) * sizeof(float);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(cur_addr), "l"(&x[(bid * 16 + sparse_rowid) * 32 + sparse_colid * 4]));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    //load dense_X(2/4)
    cur_addr = __cvta_generic_to_shared(&dense_X[1][0][0]) + (dense_rowid * 32 + shuffled_dense_colid * 2) * sizeof(float);
    rowid = SparseAToX[min(e_start + (int64_t)(dense_rowid + block_width), e_end - 1)];
    norm_j[1][dense_rowid] = x_norm[rowid];
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&x[rowid * 32 + dense_colid * 2]));
    sparse_X[1][sparse_rowid][sparse_colid] = 0.f;
    asm volatile("cp.async.commit_group;\n"::);

    //sddmm(1/3)
    uint32_t A[4];
    for (int i = 0; i < 4; i++) 
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(A[i]) : "f"(D[lane_front+(i&1)*8][(lane_back+warp_id*8+(i&2)*2+lane_front*4)&31] / x_norm_i[(i&1)]));
    {
        uint32_t frag_B[2];
        float frag_C[4] = {0.f};
        for (int i = 0; i < 2; i++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[i]) : "f"(dense_X[0][lane_front][(lane_back+i*4+warp_id*8+lane_front*4)&31] / norm_j[0][lane_front]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_C[0]), "=f"(frag_C[1]), "=f"(frag_C[2]), "=f"(frag_C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
              "r"(frag_B[0]), "r"(frag_B[1]), 
              "f"(frag_C[0]), "f"(frag_C[1]), "f"(frag_C[2]), "f"(frag_C[3]));
        for (int i = 0; i < 4; i++) 
            atomicAdd(&sparse_X[0][lane_front+(i&2)*4][(lane_back*2+(i&1)+shuffle)&7], frag_C[i]);
    }
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    //load dense_X(3/4)
    cur_addr = __cvta_generic_to_shared(&dense_X[2][0][0]) + (dense_rowid * 32 + shuffled_dense_colid * 2) * sizeof(float);
    rowid = SparseAToX[min(e_start + (int64_t)(dense_rowid + block_width*2), e_end - 1)];
    norm_j[0][dense_rowid] = x_norm[rowid];
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&x[rowid * 32 + dense_colid * 2]));
    sparse_X[2][sparse_rowid][sparse_colid] = 0.f;
    asm volatile("cp.async.commit_group;\n"::);

    //sddmm(2/3)
    {
        uint32_t frag_B[2];
        float frag_C[4] = {0.f};
        for (int i = 0; i < 2; i++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[i]) : "f"(dense_X[1][lane_front][(lane_back+i*4+warp_id*8+lane_front*4)&31] / norm_j[1][lane_front]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_C[0]), "=f"(frag_C[1]), "=f"(frag_C[2]), "=f"(frag_C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
              "r"(frag_B[0]), "r"(frag_B[1]), 
              "f"(frag_C[0]), "f"(frag_C[1]), "f"(frag_C[2]), "f"(frag_C[3]));
        for (int i = 0; i < 4; i++) 
            atomicAdd(&sparse_X[1][lane_front+(i&2)*4][(lane_back*2+(i&1)+shuffle)&7], frag_C[i]);
    }

    //compute softmax(1/2)
    {
        uint8_t mask = __ldg(&BlockMask[b_start * 16 + sparse_rowid]);
        bool valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
        float alpha = min(static_cast<float>(valid) * 6.f - 3.f, sparse_X[0][sparse_rowid][(sparse_colid+(warp_id&1))&7]) * b;
        float alpha_max = alpha;
        for (int i = 1; i < 8; i *= 2) {
            // alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * i - 1));
            alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, i));
        }
        alpha_max = __shfl_sync(FULL_MASK, alpha_max, lane_id&24);
        alpha_max = max(alpha_max, softmax[0][sparse_rowid][0]);
        softmax[0][sparse_rowid][2] = alpha_max;
        float alpha_sum = static_cast<float>(valid) * __expf(alpha - alpha_max);
        float upper = alpha_sum;
        for (int i = 1; i < 8; i *= 2) {
            // alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * i - 1);
            alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, i);
        } 
        alpha_sum = __shfl_sync(FULL_MASK, alpha_sum, lane_id&24);
        alpha_sum = alpha_sum + softmax[0][sparse_rowid][1] * __expf(softmax[0][sparse_rowid][0] - alpha_max);
        softmax[0][sparse_rowid][3] = alpha_sum;
        sparse_X[0][sparse_rowid][(sparse_colid+(warp_id&1))&7] = upper / (alpha_sum + 1e-16f);
    }

    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    float frag_D[4] = {0.f};

    for (int i = 0; i < iter - 3; i++) {
        //load dense_X(4/4)
        cur_addr = __cvta_generic_to_shared(&dense_X[(i+3)&3][0][0]) + (dense_rowid * 32 + shuffled_dense_colid * 2) * sizeof(float);
        rowid = SparseAToX[min(e_start + (int64_t)(dense_rowid + block_width*(i+3)), e_end - 1)];
        norm_j[(i+1)&1][dense_rowid] = x_norm[rowid];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(cur_addr), "l"(&x[rowid * 32 + dense_colid * 2]));
        sparse_X[(i+3)&3][sparse_rowid][sparse_colid] = 0.f;
        asm volatile("cp.async.commit_group;\n"::);

        //sddmm(3/3)
        {
            uint32_t frag_B[2];
            float frag_C[4] = {0.f};
            for (int j = 0; j < 2; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[(i+2)&3][lane_front][(lane_back+j*4+warp_id*8+lane_front*4)&31] / norm_j[i&1][lane_front]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_C[0]), "=f"(frag_C[1]), "=f"(frag_C[2]), "=f"(frag_C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(frag_B[0]), "r"(frag_B[1]), 
                "f"(frag_C[0]), "f"(frag_C[1]), "f"(frag_C[2]), "f"(frag_C[3]));
            for (int j = 0; j < 4; j++) 
                atomicAdd(&sparse_X[(i+2)&3][lane_front+(j&2)*4][(lane_back*2+(j&1)+shuffle)&7], frag_C[j]);
        }

        //compute softmax(2/2)
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            uint8_t mask = __ldg(&BlockMask[(b_start + i + 1) * 16 + sparse_rowid]);
            bool valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = min(static_cast<float>(valid) * 6.f - 3.f, sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7]) * b;
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, j));
            }
            alpha_max = __shfl_sync(FULL_MASK, alpha_max, lane_id&24);
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(valid) * __expf(alpha - alpha_max);
            float upper = alpha_sum;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * j - 1);
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, j);
            } 
            alpha_sum = __shfl_sync(FULL_MASK, alpha_sum, lane_id&24);
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1] * __expf(softmax[cur_rid][sparse_rowid][0] - alpha_max);
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7] = upper / (alpha_sum + 1e-16f);
        }

        //spmm
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][lane_front][0]), FLOAT4(softmax[i&1][lane_front+8][0])};
            float s[2];
            for (int j = 0; j < 2; j++) {
                float is_init = static_cast<float>(softmax_param[j].y > 0.f);
                s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);
            }
            for (int j = 0; j < 4; j++) {
                int id = j >> 1;
                frag_D[j] *= s[id];
            }
            uint32_t frag_A[4], B[2];
            for (int j = 0; j < 2; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_X[(i&3)][lane_front+j*8][(lane_back*2+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j+2]) : "f"(sparse_X[(i&3)][lane_front+j*8][(lane_back*2+1+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(B[j]) : "f"(dense_X[i&3][lane_back*2+j][(spmm_shuffle+warp_id*8+(lane_back*2+j)*4)&31]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        }
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }
    int i = iter - 3;
    if (i >= 0){
        //sddmm
        {
            uint32_t frag_B[2];
            float frag_C[4] = {0.f};
            for (int j = 0; j < 2; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_B[j]) : "f"(dense_X[(i+2)&3][lane_front][(lane_back+j*4+warp_id*8+lane_front*4)&31] / norm_j[i&1][lane_front]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_C[0]), "=f"(frag_C[1]), "=f"(frag_C[2]), "=f"(frag_C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
                "r"(frag_B[0]), "r"(frag_B[1]), 
                "f"(frag_C[0]), "f"(frag_C[1]), "f"(frag_C[2]), "f"(frag_C[3]));
            for (int j = 0; j < 4; j++) 
                atomicAdd(&sparse_X[(i+2)&3][lane_front+(j&2)*4][(lane_back*2+(j&1)+shuffle)&7], frag_C[j]);
        }

        //compute softmax
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            uint8_t mask = __ldg(&BlockMask[(b_start + i + 1) * 16 + sparse_rowid]);
            bool valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = min(static_cast<float>(valid) * 6.f - 3.f, sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7]) * b;
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, j));
            }
            alpha_max = __shfl_sync(FULL_MASK, alpha_max, lane_id&24);
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(valid) * __expf(alpha - alpha_max);
            float upper = alpha_sum;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * j - 1);
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, j);
            } 
            alpha_sum = __shfl_sync(FULL_MASK, alpha_sum, lane_id&24);
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1] * __expf(softmax[cur_rid][sparse_rowid][0] - alpha_max);
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7] = upper / (alpha_sum + 1e-16f);
        }

        //spmm
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][lane_front][0]), FLOAT4(softmax[i&1][lane_front+8][0])};
            float s[2];
            for (int j = 0; j < 2; j++) {
                float is_init = static_cast<float>(softmax_param[j].y > 0.f);
                s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);
            }
            for (int j = 0; j < 4; j++) {
                int id = j >> 1;
                frag_D[j] *= s[id];
            }
            uint32_t frag_A[4], B[2];
            // int p = (i == 0) ? 0 : (i % 3);
            for (int j = 0; j < 2; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j+2]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+1+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(B[j]) : "f"(dense_X[i&3][lane_back*2+j][(spmm_shuffle+warp_id*8+(lane_back*2+j)*4)&31]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        }
        __syncthreads();
    }
    i = iter - 2;
    if (i >= 0){
        //compute softmax
        {
            int cur_rid = (i+1)&1;
            softmax[cur_rid][sparse_rowid][0] = softmax[i&1][sparse_rowid][2];
            softmax[cur_rid][sparse_rowid][1] = softmax[i&1][sparse_rowid][3];
            uint8_t mask = __ldg(&BlockMask[(b_start + i + 1) * 16 + sparse_rowid]);
            bool valid = (mask & ((uint8_t)(1 << sparse_colid))) > 0;
            float alpha = min(static_cast<float>(valid) * 6.f - 3.f, sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7]) * b;
            float alpha_max = alpha;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, 2 * j - 1));
                alpha_max = max(alpha_max, __shfl_xor_sync(FULL_MASK, alpha_max, j));
            }
            alpha_max = __shfl_sync(FULL_MASK, alpha_max, lane_id&24);
            alpha_max = max(alpha_max, softmax[cur_rid][sparse_rowid][0]);
            softmax[cur_rid][sparse_rowid][2] = alpha_max;
            float alpha_sum = static_cast<float>(valid) * __expf(alpha - alpha_max);
            float upper = alpha_sum;
            for (int j = 1; j < 8; j *= 2) {
                // alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, 2 * j - 1);
                alpha_sum += __shfl_xor_sync(FULL_MASK, alpha_sum, j);
            } 
            alpha_sum = __shfl_sync(FULL_MASK, alpha_sum, lane_id&24);
            alpha_sum = alpha_sum + softmax[cur_rid][sparse_rowid][1] * __expf(softmax[cur_rid][sparse_rowid][0] - alpha_max);
            softmax[cur_rid][sparse_rowid][3] = alpha_sum;
            sparse_X[(i+1)&3][sparse_rowid][(sparse_colid+(warp_id&1))&7] = upper / (alpha_sum + 1e-16f);
        }

        //spmm
        {
            float4 softmax_param[2] = {FLOAT4(softmax[i&1][lane_front][0]), FLOAT4(softmax[i&1][lane_front+8][0])};
            float s[2];
            for (int j = 0; j < 2; j++) {
                float is_init = static_cast<float>(softmax_param[j].y > 0.f);
                s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);
            }
            for (int j = 0; j < 4; j++) {
                int id = j >> 1;
                frag_D[j] *= s[id];
            }
            uint32_t frag_A[4], B[2];
            // int p = (i == 0) ? 0 : (i % 3);
            for (int j = 0; j < 2; j++) {
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j+2]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+1+shuffle)&7]));
                asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(B[j]) : "f"(dense_X[i&3][lane_back*2+j][(spmm_shuffle+warp_id*8+(lane_back*2+j)*4)&31]));
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
                "r"(B[0]), "r"(B[1]), 
                "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        }
        __syncthreads();
    }
    i = iter - 1;
    //spmm
    {
        float4 softmax_param[2] = {FLOAT4(softmax[i&1][lane_front][0]), FLOAT4(softmax[i&1][lane_front+8][0])};
        float s[2];
        for (int j = 0; j < 2; j++) {
            float is_init = static_cast<float>(softmax_param[j].y > 0.f);
            s[j] = is_init * __expf(softmax_param[j].x - softmax_param[j].z) * (softmax_param[j].y + 1e-16f) / (softmax_param[j].w + 1e-16f);
        }
        for (int j = 0; j < 4; j++) {
            int id = j >> 1;
            frag_D[j] *= s[id];
        }
        uint32_t frag_A[4], B[2];
        // int p = (i == 0) ? 0 : (i % 3);
        for (int j = 0; j < 2; j++) {
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+shuffle)&7]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(frag_A[j+2]) : "f"(sparse_X[i&3][lane_front+j*8][(lane_back*2+1+shuffle)&7]));
            asm volatile("cvt.rna.tf32.f32 %0, %1;\n": "=r"(B[j]) : "f"(dense_X[i&3][lane_back*2+j][(spmm_shuffle+warp_id*8+(lane_back*2+j)*4)&31]));
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(B[0]), "r"(B[1]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
    }
    for (int i = 0; i < 4; i++) {
        D[lane_front+(i&2)*4][(lane_back+(i&1)*4+warp_id*8+lane_front*4)&31] = frag_D[i];
    }
    __syncthreads();
    if (bid * block_high + sparse_rowid < node_len)
        FLOAT4(output[(bid * block_high + sparse_rowid) * 32 + sparse_colid * 4]) = FLOAT4(D[sparse_rowid][shuffled_sparse_colid * 4]);
}

at::Tensor AGNN(
    at::Tensor x,
    at::Tensor RowWindowOffset,
    at::Tensor TCOffset,
    at::Tensor BlockMask,
    at::Tensor SparseAToX,
    at::Tensor beta,
    int out_feats
) {
    int num_nodes = x.size(0);
    auto x_norm = x.norm(2, -1).clamp_min(1e-12);
    auto output = at::empty({num_nodes, out_feats}, x.options());

    int threads = 128;
    int blocks = (num_nodes + 15) / 16;
    // Timer timer;
    // timer.tik();
    
    agnn_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        RowWindowOffset.data_ptr<int>(),
        TCOffset.data_ptr<int64_t>(),
        BlockMask.data_ptr<uint8_t>(),
        SparseAToX.data_ptr<int>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        16,
        8,
        num_nodes);
    // cudaDeviceSynchronize();
    // timer.tok();
    // printf("AGNN time: %f ms\n", timer.get_time());
    // printf("x_norm_size: %d\n", x_norm.size(0));
    return output;
}