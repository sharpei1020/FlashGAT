#include "preprocess.cuh"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform_scan.h>
#include <omp.h>



std::vector<at::Tensor> preprocess_CSR(at::Tensor edge_index, at::Tensor counts, int group, int num_nodes)
{
    auto offset = at::empty({num_nodes}, device(torch::kCUDA)).to(torch::kInt);
    thrust::inclusive_scan(thrust::device, counts.data_ptr<int>(), counts.data_ptr<int>() + num_nodes, offset.data_ptr<int>());
    int num_edges = edge_index.size(1);
    auto out_edge_index = at::empty({num_edges}, device(torch::kCUDA)).to(torch::kInt);
    thrust::sequence(thrust::device, out_edge_index.data_ptr<int>(), out_edge_index.data_ptr<int>() + num_edges);
    u_int32_t* value;
    cudaMalloc(&value, num_edges * sizeof(u_int32_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), 
                      edge_index.data_ptr<int>() + num_edges, edge_index.data_ptr<int>() + num_edges,
                      value, [=] __device__ (int e0, int e1) {return ((uint32_t)(e1 / group) * (uint32_t)num_nodes + (uint32_t)e0);});
    thrust::stable_sort_by_key(thrust::device, value, value + num_edges, out_edge_index.data_ptr<int>());
    cudaFree(value);
    return {offset, out_edge_index};
}

std::vector<at::Tensor> get_graph_set(at::Tensor edge_index, int numwarps, int numnodes)
{
    int num_edges = edge_index.size(1);
    auto value = at::empty({num_edges}, device(torch::kCUDA)).to(torch::kInt64);
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), 
                      edge_index.data_ptr<int>() + num_edges, edge_index.data_ptr<int>() + num_edges,
                      value.data_ptr<int64_t>(), [=] __device__ (int e0, int e1) {return (int64_t)e1 * (int64_t)numnodes + (int64_t)e0;});             
    int groups = (numnodes + numwarps - 1) / numwarps;
    std::vector<at::Tensor> graph_set(groups), graph_mask(groups);
    std::vector<int64_t> graph_size(groups + 1);
    graph_size[0] = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < groups; i++) {
        graph_set[i] = edge_index.slice(0, 0, 1).masked_select(value >= (int64_t)i * (int64_t)numwarps * (int64_t)numnodes & value < (int64_t)(i + 1) * (int64_t)numwarps * (int64_t)numnodes);
        graph_set[i] = std::get<0>(torch::_unique(graph_set[i]));
        // printf("graph_set[%d] size: %d\n", i, graph_set[i].size(0));
        graph_size[i + 1] = graph_set[i].size(0);
        // auto new_end = thrust::unique(thrust::device, graph_set[i].data_ptr<int>(), graph_set[i].data_ptr<int>() + graph_set[i].size(0));
        // graph_size[i + 1] = new_end - graph_set[i].data_ptr<int>();
        graph_mask[i] = at::empty({numwarps, graph_size[i + 1]}, device(torch::kCUDA)).to(torch::kBool);
        for (int j = 0; j < numwarps; j++) {
            auto temp_mask = at::empty({numnodes}, device(torch::kCUDA)).to(torch::kBool).fill_(false);
            auto temp_idx = edge_index.slice(0, 0, 1).masked_select(value >= (int64_t)(i * numwarps + j) * (int64_t)numnodes & value < (int64_t)(i * numwarps + j + 1) * (int64_t)numnodes).to(torch::kInt64);
            temp_mask.index_fill_(0, temp_idx, true); 
            // printf("%d \n", graph_size[i + 1]);
            thrust::gather(thrust::device, graph_set[i].data_ptr<int>(), graph_set[i].data_ptr<int>() + graph_size[i + 1], 
                            temp_mask.data_ptr<bool>(), graph_mask[i].data_ptr<bool>() + j * graph_size[i + 1]);
        }
    }
    printf("get_graph_set done\n");
    for (int i = 0; i < groups; i++) 
        graph_size[i + 1] += graph_size[i];
    auto graphset = at::empty({graph_size[groups]}, device(torch::kCUDA)).to(torch::kInt);
    auto graphmask = at::empty({graph_size[groups], numwarps}, device(torch::kCUDA)).to(torch::kBool);
    auto graphoffset = at::empty({groups + 1}, device(torch::kCUDA)).to(torch::kInt64);
    cudaMemcpy(graphoffset.data_ptr<int64_t>(), &graph_size[0], sizeof(int64_t), cudaMemcpyHostToDevice);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < groups; i++) {
        auto temp_mask = graph_mask[i].transpose(0, 1).contiguous();
        thrust::copy(thrust::device, graph_set[i].data_ptr<int>(), graph_set[i].data_ptr<int>() + graph_set[i].size(0), 
                    graphset.data_ptr<int>() + graph_size[i]);
        thrust::copy(thrust::device, temp_mask.data_ptr<bool>(), temp_mask.data_ptr<bool>() + temp_mask.size(0) * temp_mask.size(1), 
                    graphmask.data_ptr<bool>() + graph_size[i] * numwarps);
        cudaMemcpy(graphoffset.data_ptr<int64_t>() + i + 1, &graph_size[i + 1], sizeof(int64_t), cudaMemcpyHostToDevice);
    }
    return {graphset, graphmask, graphoffset};
}

__global__ void BlockMaskKernel(
    int* edge_index_0,
    int* edge_index_1,
    u_int32_t* TCOffset,
    uint8_t* BlockMask,
    int block_high,
    int block_width,
    int block_num
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= block_num) return;
    u_int32_t start = TCOffset[tid];
    u_int32_t end = TCOffset[tid + 1];
    uint8_t mask[16] = {0};
     
    for (uint32_t i = start; i < end; i++) {
        int block_col = ((edge_index_0[i] - 1) % block_width);
        int block_row = (edge_index_1[i] % block_high);
        mask[block_row] += (uint8_t)(1 << block_col);
    }
    *((int4 *)(&BlockMask[tid * 16])) = *((int4 *)(&mask[0]));
}

std::vector<torch::Tensor> process_DTC(at::Tensor edge_index, at::Tensor dev_idx, int block_high, int block_width, int num_nodes)
{
    uint32_t* value, *value1;
    // , *value2
    int* rowwindow_value, *X_col_id, *edge_idx1;
    cudaMalloc(&value, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&value1, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&edge_idx1, edge_index.size(1) * sizeof(int));
    thrust::gather(thrust::device, edge_index.data_ptr<int>() + edge_index.size(1), edge_index.data_ptr<int>() + 2 * edge_index.size(1), dev_idx.data_ptr<int>(), edge_idx1);
    // cudaMalloc(&value2, edge_index.size(1) * sizeof(uint32_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), 
                      edge_index.data_ptr<int>() + edge_index.size(1), edge_idx1,
                      value, [=] __device__ (int e0, int e1) {return ((uint32_t)(e1 / block_high) * (uint32_t)(num_nodes) + (uint32_t)(e0));});
    thrust::copy(thrust::device, value, value + edge_index.size(1), value1);
    // thrust::copy(thrust::device, value, value + edge_index.size(1), value2);
    
    thrust::stable_sort_by_key(thrust::device, value, value + edge_index.size(1), edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value1, value1 + edge_index.size(1), edge_idx1);
    // thrust::stable_sort_by_key(thrust::device, value2, value + edge_index.size(1), rowwindow_value);
    cudaFree(value);
    cudaFree(value1);
    cudaMalloc(&rowwindow_value, edge_index.size(1) * sizeof(int));

    thrust::transform(thrust::device, edge_idx1, edge_idx1 + edge_index.size(1),
                      rowwindow_value, [=] __device__ (int e) {return e / block_high;});

    int* SparseAidx;
    cudaMalloc(&SparseAidx, edge_index.size(1) * sizeof(int));
    thrust::adjacent_difference(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>() + edge_index.size(1), 
                                SparseAidx, [=]__device__(int x1, int x0) {return (x1 == x0) ? 0 : 1;});
    cudaMemset(SparseAidx, 1, sizeof(int));
    thrust::inclusive_scan_by_key(thrust::device, rowwindow_value, rowwindow_value + edge_index.size(1), SparseAidx, SparseAidx, 
                                thrust::equal_to<int>(), thrust::plus<int>());

    cudaMalloc(&X_col_id, edge_index.size(1) * sizeof(int));
    thrust::copy(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>() + edge_index.size(1), X_col_id);
    auto unique_rowwindow_end = thrust::unique_by_key(thrust::device, X_col_id, X_col_id + edge_index.size(1), 
                            rowwindow_value, thrust::equal_to<int>());
    //SparseAToX
    auto SparseAToX = torch::from_blob(X_col_id, {unique_rowwindow_end.first - X_col_id}, device(torch::kCUDA).dtype(torch::kInt32)).clone();
    int* RowWindowMask, *RowWindowNum_temp, *RowWindowNum, *unique_rowwindow_value;
    cudaMalloc(&RowWindowMask, SparseAToX.size(0) * sizeof(int));
    cudaMalloc(&RowWindowNum_temp, SparseAToX.size(0) * sizeof(int));
    cudaMalloc(&RowWindowNum, (num_nodes + block_high - 1) / block_high * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowNum, (num_nodes + block_high - 1) / block_high, 0);
    cudaMalloc(&unique_rowwindow_value, SparseAToX.size(0) * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowMask, SparseAToX.size(0), 1);
    // printf("SparseAToX size: %d\n", SparseAToX.size(0));
    auto RowWindowColOffset = at::empty({1 + (num_nodes + block_high - 1) / block_high}, device(torch::kCUDA).dtype(torch::kInt64)).fill_(0);
    auto new_end_ = thrust::reduce_by_key(thrust::device, rowwindow_value, unique_rowwindow_end.second, RowWindowMask,
                            unique_rowwindow_value, RowWindowNum_temp, thrust::equal_to<int>(), thrust::plus<int>());
    thrust::scatter(thrust::device, RowWindowNum_temp, new_end_.second, unique_rowwindow_value, RowWindowNum);
    thrust::inclusive_scan(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowColOffset.data_ptr<int64_t>() + 1);
    thrust::transform(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowNum, [=]__device__(int x) {return (x + block_width - 1) / block_width;});
    auto RowWindowOffset = at::empty({(num_nodes + block_high - 1) / block_high + 1}, device(torch::kCUDA).dtype(torch::kInt32)).fill_(0);
    //RowWindowOffset
    thrust::inclusive_scan(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowOffset.data_ptr<int>() + 1);
    cudaFree(X_col_id);
    cudaFree(RowWindowMask);
    cudaFree(RowWindowNum_temp);
    cudaFree(RowWindowNum);
    cudaFree(unique_rowwindow_value);
    
    int block_num = RowWindowOffset[-1].item().to<int>();
    // printf("block_num: %d\n", block_num);
    // auto TCOffset = at::empty({block_num + 1}, device(torch::kCUDA).dtype(torch::kInt64)).fill_(0);
    uint32_t *block_value, *TCOffset, *BlockElementMask, *BlockElementNum;
    cudaMalloc(&TCOffset, (block_num + 1) * sizeof(uint32_t));
    cudaMalloc(&block_value, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&BlockElementMask, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&BlockElementNum, block_num * sizeof(uint32_t));
    cudaMemset(TCOffset, 0, sizeof(uint32_t));
    thrust::fill_n(thrust::device, BlockElementMask, edge_index.size(1), 1);
    thrust::transform(thrust::device, SparseAidx, SparseAidx + edge_index.size(1), edge_idx1,
                    block_value, 
                    [=] __device__ (int e0, int e1) {return ((uint32_t)(e1 / block_high) * (uint32_t)(num_nodes) + (uint32_t)((e0-1) / block_width));});
    auto new_end = thrust::reduce_by_key(thrust::device, block_value, block_value + (int)edge_index.size(1), BlockElementMask,
                            block_value, BlockElementNum, thrust::equal_to<uint32_t>(), thrust::plus<uint32_t>());
    thrust::inclusive_scan(thrust::device, BlockElementNum, BlockElementNum + block_num, TCOffset + 1);
    
    //BlockMask
    auto BlockMask = at::empty({16 * block_num}, device(torch::kCUDA).dtype(torch::kUInt8));
    int grid = (block_num + 127) / 128;
    BlockMaskKernel<<<grid, 128>>>(SparseAidx, 
                                    edge_idx1, 
                                    TCOffset, 
                                    BlockMask.data_ptr<uint8_t>(), 
                                    block_high, block_width, block_num);
    cudaFree(rowwindow_value);
    cudaFree(SparseAidx);
    cudaFree(TCOffset);
    cudaFree(block_value);
    cudaFree(BlockElementMask);
    cudaFree(BlockElementNum);
    return {RowWindowOffset, RowWindowColOffset, BlockMask, SparseAToX};
}
    