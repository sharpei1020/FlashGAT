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
    bool bias_flag);

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
    int edge_len);

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
);