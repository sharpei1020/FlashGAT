#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__device__ static inline void scale_sum(float *output, float *input, float scale, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] += input[i] * scale;
    }
}

__device__ static inline void scale(float *temp, float scale, int temp_size)
{
    for (int i = 0; i < temp_size; i++)
        temp[i] *= scale;
}

__device__ static inline void debug(float *temp, float *test_output, int embed)
{
    for (int i = 0; i < embed; i++)
    {
        test_output[i] = temp[i];
    }
}

// __global__ void linear(
//     const float* __restrict__ weight, 
//     const float* __restrict__ bias, 
//     const float* __restrict__ X,
//     const float* __restrict__ att_i,
//     const float* __restrict__ att_j, 
//     float*  __restrict__ Z, 
//     float*  __restrict__ alpha_i,
//     float*  __restrict__ alpha_j,
//     int M, 
//     int K, 
//     bool bias_flag);

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
//     int edge_len);

at::Tensor GAT(
    at::Tensor feature,
    at::Tensor RowWindowOffset,
    at::Tensor TCOffset,
    at::Tensor BlockMask,
    at::Tensor SparseAToX,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    int num_heads,
    int out_feats
);

at::Tensor GAT_balance(
    at::Tensor feature,
    at::Tensor RowWindowOffset,     // RowWindow-Row nonzero-block Offset
    at::Tensor RowWindowRowOffset,  // RowWindow Row offset
    at::Tensor TCOffset,            // RowWindow-Row nonzero-element Offset
    at::Tensor BlockMask,           // TC-Block Mask
    at::Tensor SparseAToX,          // SparseA-X nonzero-element colidx
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    int num_heads,
    int out_feats
);

at::Tensor GAT_short(
    at::Tensor feature,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor lin_weight,
    at::Tensor att_i, 
    at::Tensor att_j,
    int num_heads,
    int out_feats,
    int block_high,
    int block_width
);

at::Tensor sputnik_GAT(
    at::Tensor feature,
    at::Tensor row_idx,
    at::Tensor row_offset,
    at::Tensor col_idx,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    int num_heads,
    int out_feats
);