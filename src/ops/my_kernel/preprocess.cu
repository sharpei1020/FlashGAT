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
    auto offset = at::empty({num_nodes + 1}, device(torch::kCUDA).dtype(torch::kInt));
    thrust::inclusive_scan(thrust::device, counts.data_ptr<int>(), counts.data_ptr<int>() + num_nodes, offset.data_ptr<int>() + 1);
    int num_edges = edge_index.size(1);
    auto out_edge_index = at::empty({num_edges}, device(torch::kCUDA).dtype(torch::kInt));
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

std::vector<at::Tensor> process_CSR(at::Tensor edge_index, int group, int num_nodes)
{
    int num_edges = edge_index.size(1);
    auto out_edge_index = at::empty({num_edges}, device(torch::kCUDA).dtype(torch::kInt));
    thrust::sequence(thrust::device, out_edge_index.data_ptr<int>(), out_edge_index.data_ptr<int>() + num_edges);
    int *edge_index_1;
    cudaMalloc(&edge_index_1, num_edges * sizeof(int));
    thrust::copy(thrust::device, edge_index.data_ptr<int>() + num_edges, edge_index.data_ptr<int>() + 2 * num_edges, edge_index_1);
    u_int32_t* value, *value1;
    cudaMalloc(&value, num_edges * sizeof(u_int32_t));
    cudaMalloc(&value1, num_edges * sizeof(u_int32_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>() + num_edges,
                       edge_index.data_ptr<int>() + num_edges, value,
                       [=] __device__ (int e0, int e1) {return ((uint32_t)(e1 / group) * (uint32_t)num_nodes + (uint32_t)e0);});
    thrust::copy(thrust::device, value, value + num_edges, value1);
    thrust::stable_sort_by_key(thrust::device, value, value + num_edges, out_edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value1, value1 + num_edges, edge_index_1);
    cudaFree(value);
    cudaFree(value1);
    auto row_idx = torch::from_blob(edge_index_1, {num_edges}, device(torch::kCUDA).dtype(torch::kInt32)).clone();
    return {row_idx, out_edge_index};
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

__global__ void RowWindowKernel(
    const int* __restrict__ RowWindowRowOffset,  
    const int* __restrict__ RowWindowNum,        
    int64_t* __restrict__ RowWindowColOffset,      //RowWindowColOffset(RowWindow nonzero offset)
    int* __restrict__ RowWindowOffset,     //RowWindowOffset(RowWindow nonero block offset)
    int average,
    int block_width,
    int num_row
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_row) return;
    int start = RowWindowRowOffset[tid];
    int end = RowWindowRowOffset[tid + 1];
    int nonzero_num = RowWindowNum[tid];
    int segment_num = average * block_width;
    for (int i = start; i < end; i++) {
        RowWindowColOffset[i] = (int64_t)min(segment_num, nonzero_num - (i - start) * segment_num);
        RowWindowOffset[i] = min(average, (nonzero_num - (i - start) * segment_num + block_width - 1) / block_width);
    }
}

__global__ void BlockMaskKernel(
    const int* __restrict__ edge_index_0,
    const int* __restrict__ edge_index_1,
    const u_int32_t* __restrict__ TCOffset,
    uint8_t* __restrict__ BlockMask,
    const int block_high,
    const int block_width,
    const int block_num
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= block_num) return;
    u_int32_t start = TCOffset[tid];
    u_int32_t end = TCOffset[tid + 1];
    uint32_t mask[4] = {0};
    for (uint32_t i = start; i < end; i++) {
        int block_col = ((edge_index_0[i] - 1) & (block_width - 1));
        int block_row = (edge_index_1[i] & (block_high - 1));
        mask[block_row>>2] |= (1 << (block_col + (block_row & 3) * 8));
    }
    *((uint4 *)(&BlockMask[tid * 16])) = *((uint4 *)(&mask[0]));
}


__global__ void BlockMaskShortKernel(
    const int* __restrict__ edge_index_0,
    const int* __restrict__ edge_index_1,
    const uint32_t* __restrict__ TCOffset,
    const int* __restrict__ BitMask_RowOffset,
    uint8_t* __restrict__ BlockMaskCol,
    uint8_t* __restrict__ BlockMaskRow,
    const int block_high,
    const int block_width,
    const int block_num
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= block_num) return;
    uint32_t start = TCOffset[tid];
    uint32_t end = TCOffset[tid + 1];
    int write_start = BitMask_RowOffset[tid];
    int write_end = BitMask_RowOffset[tid + 1];
    int width_num = block_width / 8;
    uint8_t col_mask[4] = {0};
    uint8_t row_mask[64] = {0};
    for (uint32_t i = start; i < end; i++) {
        int block_col = ((edge_index_0[i] - 1) & (block_width - 1));
        int block_row = (edge_index_1[i] & (block_high - 1));
        col_mask[block_row>>3] |= (1 << (block_row & 7));
        row_mask[block_row*width_num + (block_col >> 3)] |= (1 << (block_col & 7));
    }
    if (block_high == 8) 
        *((uint8_t*)(&BlockMaskCol[tid])) = col_mask[0];
    else if (block_high == 16) 
        *((uint16_t*)(&BlockMaskCol[tid*2])) = *(uint16_t*)(&col_mask[0]);
    else if (block_high == 32)
        *((uint32_t*)(&BlockMaskCol[tid*4])) = *(uint32_t*)(&col_mask[0]);
    int m = __popc(*(uint32_t*)(&col_mask[0]));
    for (int i = 1; i <= m; i++) {
        int j = __fns(*(uint32_t*)(&col_mask[0]), 0, i);
        if (width_num == 1)
            *(uint8_t*)(&BlockMaskRow[write_start + i - 1]) = row_mask[j];
        else if (width_num == 2)
            *(uint16_t*)(&BlockMaskRow[(write_start + i - 1) * 2]) = *(uint16_t*)(&row_mask[j*2]);
        else if (width_num == 4)
            *(uint32_t*)(&BlockMaskRow[(write_start + i - 1) * 4]) = *(uint32_t*)(&row_mask[j*4]);
    }
    //uint16_t col_mask = 0;
    // uint8_t row_mask[16] = {0};
    // for (uint32_t i = start; i < end; i++) {
    //     int block_col = ((edge_index_0[i] - 1) & (block_width - 1));
    //     int block_row = (edge_index_1[i] & (block_high - 1));

        // col_mask |= (1 << block_row);
        // row_mask[block_row] |= (1 << block_col);
    // }
    // *((uint16_t*)(&BlockMaskCol[tid*2])) = col_mask;
    // int m = __popc((uint32_t)col_mask);
    // for (int i = 1; i <= m; i++) {
    //     int j = __fns(col_mask, 0, i);
    //     BlockMaskRow[write_start + i - 1] = row_mask[j];
    // }
}

std::vector<torch::Tensor> process_DTC(at::Tensor edge_index, at::Tensor dev_idx, int block_high, int block_width, int num_nodes, bool balance)
{
    uint64_t* value, *value1;
    int* rowwindow_value, *X_col_id, *edge_idx1;
    cudaMalloc(&value, edge_index.size(1) * sizeof(uint64_t));
    cudaMalloc(&value1, edge_index.size(1) * sizeof(uint64_t));
    cudaMalloc(&edge_idx1, edge_index.size(1) * sizeof(int));
    thrust::gather(thrust::device, edge_index.data_ptr<int>() + edge_index.size(1), 
                    edge_index.data_ptr<int>() + 2 * edge_index.size(1), dev_idx.data_ptr<int>(), edge_idx1);
    // after get edge_index_1

    thrust::transform(thrust::device, edge_index.data_ptr<int>(), 
                      edge_index.data_ptr<int>() + edge_index.size(1), edge_index.data_ptr<int>() + edge_index.size(1),
                      value, [=] __device__ (int e0, int e1) {return ((uint64_t)(e1 / block_high) * (uint64_t)(num_nodes) + (uint64_t)(e0));});
    // printf("after transform\n");
    thrust::copy(thrust::device, value, value + edge_index.size(1), value1);
    thrust::stable_sort_by_key(thrust::device, value, value + edge_index.size(1), edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value1, value1 + edge_index.size(1), edge_idx1);
    cudaMalloc(&rowwindow_value, edge_index.size(1) * sizeof(int));
    thrust::transform(thrust::device, edge_idx1, edge_idx1 + edge_index.size(1),
                      rowwindow_value, [=] __device__ (int e) {return e / block_high;});
    // printf("after stable_sort_by_key and transform.\n");
    cudaFree(value);
    cudaFree(value1);
    // after sort edge_index_0 and edge_index_1 according to rowwindow_id(rowwindow_value) and col_id
    int* SparseAidx, *SparseAidx_temp, *SparseAidx_temp1;
    cudaMalloc(&SparseAidx, edge_index.size(1) * sizeof(int));
    cudaMalloc(&SparseAidx_temp, edge_index.size(1) * sizeof(int));
    thrust::adjacent_difference(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>() + edge_index.size(1), 
                                SparseAidx, [=]__device__(int x1, int x0) {return (x1 == x0) ? 0 : 1;});
    thrust::adjacent_difference(thrust::device, rowwindow_value, rowwindow_value + edge_index.size(1), SparseAidx_temp,
                                [=]__device__(int x1, int x0) {return (x1 == x0) ? 0 : 1;});
    thrust::transform(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), SparseAidx, SparseAidx, 
                    [=]__device__(int x0, int x1) {return (x0 | x1);});
    // printf("after adjacent_difference and transform.\n");
    cudaMemset(SparseAidx, 1, 1);
    thrust::copy_n(thrust::device, SparseAidx, edge_index.size(1), SparseAidx_temp);
    thrust::inclusive_scan_by_key(thrust::device, rowwindow_value, rowwindow_value + edge_index.size(1), SparseAidx, SparseAidx, 
                                thrust::equal_to<int>(), thrust::plus<int>());
    thrust::inclusive_scan(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), SparseAidx_temp);
    // printf("after inclusive_scan.\n");
    cudaMalloc(&SparseAidx_temp1, edge_index.size(1) * sizeof(int));
    thrust::copy_n(thrust::device, SparseAidx_temp, edge_index.size(1), SparseAidx_temp1);
    // after record nonzero-element's col_id in sparse matrix
    cudaMalloc(&X_col_id, edge_index.size(1) * sizeof(int));
    thrust::copy_n(thrust::device, edge_index.data_ptr<int>(), edge_index.size(1), X_col_id);
    auto unique_X_col_id_end = thrust::unique_by_key(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), 
                            X_col_id, thrust::equal_to<int>());
    auto unique_rowwindow_end = thrust::unique_by_key(thrust::device, SparseAidx_temp1, SparseAidx_temp1 + edge_index.size(1), 
                            rowwindow_value, thrust::equal_to<int>());
    cudaFree(SparseAidx_temp);
    cudaFree(SparseAidx_temp1);
    //SparseAToX
    auto SparseAToX = torch::from_blob(X_col_id, {unique_X_col_id_end.second - X_col_id}, device(torch::kCUDA).dtype(torch::kInt32)).clone();
    // printf("after getting SparseAToX.\n");
    //after de-duplicate col_id
    int* RowWindowMask, *RowWindowNum_temp, *RowWindowNum, *unique_rowwindow_value;
    cudaMalloc(&RowWindowMask, SparseAToX.size(0) * sizeof(int));
    cudaMalloc(&RowWindowNum_temp, SparseAToX.size(0) * sizeof(int));
    cudaMalloc(&RowWindowNum, (num_nodes + block_high - 1) / block_high * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowNum, (num_nodes + block_high - 1) / block_high, 0);
    cudaMalloc(&unique_rowwindow_value, SparseAToX.size(0) * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowMask, SparseAToX.size(0), 1);
    // printf("SparseAToX size: %d\n", SparseAToX.size(0));
    at::Tensor RowWindowColOffset, RowWindowOffset, RowWindowRowOffset;
    int block_num;
    auto new_end_ = thrust::reduce_by_key(thrust::device, rowwindow_value, unique_rowwindow_end.second, RowWindowMask,
                            unique_rowwindow_value, RowWindowNum_temp, thrust::equal_to<int>(), thrust::plus<int>());
    thrust::scatter(thrust::device, RowWindowNum_temp, new_end_.second, unique_rowwindow_value, RowWindowNum);
    // printf("after scatter and reduce_by_key.\n");
    if (balance) {
        int* RowWindowNum_temp1;
        cudaMalloc(&RowWindowNum_temp1, (num_nodes + block_high - 1) / block_high * sizeof(int));
        cudaMalloc(&RowWindowNum_temp, (num_nodes + block_high - 1) / block_high * sizeof(int));
        thrust::transform(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowNum_temp, 
                        [=]__device__(int x) {return (x + block_width - 1) / block_width;});
        thrust::inclusive_scan(thrust::device, RowWindowNum_temp, RowWindowNum_temp + (num_nodes + block_high - 1) / block_high, RowWindowNum_temp1);
        cudaMemcpy(&block_num, RowWindowNum_temp1 + (num_nodes + block_high - 1) / block_high - 1, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("block_num: %d\n", block_num);
        cudaFree(RowWindowNum_temp1);
        int average_block_num = block_num * block_high / (num_nodes + block_high - 1);
        // printf("average_block_num: %d\n", average_block_num);
        // int first_RowWindow_num;
        // cudaMemcpy(&first_RowWindow_num, RowWindowNum, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("first_RowWindow_num: %d\n", first_RowWindow_num);
        //RowWindowRowOffset(RowWindow Row offset)
        RowWindowRowOffset = at::empty({1 + (num_nodes + block_high - 1) / block_high}, device(torch::kCUDA).dtype(torch::kInt32));
        cudaMemset(RowWindowRowOffset.data_ptr<int>(), 0, sizeof(int));
        thrust::transform(thrust::device, RowWindowNum_temp, RowWindowNum_temp + (num_nodes + block_high - 1) / block_high, 
                    RowWindowRowOffset.data_ptr<int>() + 1, [=]__device__(int x) {return std::max(1, (x + average_block_num - 1) / average_block_num);});
        thrust::inclusive_scan(thrust::device, RowWindowRowOffset.data_ptr<int>() + 1, RowWindowRowOffset.data_ptr<int>() + 1 + (num_nodes + block_high - 1) / block_high, 
                    RowWindowRowOffset.data_ptr<int>() + 1);
        int row_num;
        cudaMemcpy(&row_num, RowWindowRowOffset.data_ptr<int>() + (num_nodes + block_high - 1) / block_high, sizeof(int), cudaMemcpyDeviceToHost);
        RowWindowColOffset = at::empty({1 + row_num}, device(torch::kCUDA).dtype(torch::kInt64)).fill_(0);
        RowWindowOffset = at::empty({1 + row_num}, device(torch::kCUDA).dtype(torch::kInt32)).fill_(0);
        int grid = ((num_nodes + block_high - 1) / block_high + 127) / 128;
        RowWindowKernel<<<grid, 128>>>(RowWindowRowOffset.data_ptr<int>(), RowWindowNum, RowWindowColOffset.data_ptr<int64_t>() + 1, 
                                        RowWindowOffset.data_ptr<int>() + 1, average_block_num, block_width, (num_nodes + block_high - 1) / block_high);
        thrust::inclusive_scan(thrust::device, RowWindowOffset.data_ptr<int>() + 1, RowWindowOffset.data_ptr<int>() + 1 + row_num, 
                    RowWindowOffset.data_ptr<int>() + 1);
        thrust::inclusive_scan(thrust::device, RowWindowColOffset.data_ptr<int64_t>() + 1, RowWindowColOffset.data_ptr<int64_t>() + 1 + row_num, 
                    RowWindowColOffset.data_ptr<int64_t>() + 1);
    } else {
        //RowWindowRowOffset(RowWindow Row offset)
        RowWindowRowOffset = at::empty({1 + (num_nodes + block_high - 1) / block_high}, device(torch::kCUDA).dtype(torch::kInt32));
        thrust::sequence(thrust::device, RowWindowRowOffset.data_ptr<int>(), RowWindowRowOffset.data_ptr<int>() + 1 + (num_nodes + block_high - 1) / block_high);
        //RowWindowColOffset(RowWindow-Row nonzero offset)
        RowWindowColOffset = at::empty({1 + (num_nodes + block_high - 1) / block_high}, device(torch::kCUDA).dtype(torch::kInt64)).fill_(0);
        thrust::inclusive_scan(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, 
                                RowWindowColOffset.data_ptr<int64_t>() + 1);
        thrust::transform(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowNum, 
                        [=]__device__(int x) {return (x + block_width - 1) / block_width;});
        //RowWindowOffset(RowWindow-Row nonero block offset)
        RowWindowOffset = at::empty({(num_nodes + block_high - 1) / block_high + 1}, device(torch::kCUDA).dtype(torch::kInt32)).fill_(0);
        thrust::inclusive_scan(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high, RowWindowOffset.data_ptr<int>() + 1);
        block_num = RowWindowOffset[-1].item().to<int>();
    }
    cudaFree(X_col_id);
    cudaFree(RowWindowMask);
    cudaFree(RowWindowNum_temp);
    cudaFree(RowWindowNum);
    cudaFree(unique_rowwindow_value);
    
    // printf("block_num: %d\n", block_num);
    // auto TCOffset = at::empty({block_num + 1}, device(torch::kCUDA).dtype(torch::kInt64)).fill_(0);
    uint32_t *block_value, *TCOffset, *BlockElementMask, *BlockElementNum;
    cudaMalloc(&TCOffset, (block_num + 1) * sizeof(uint32_t));
    cudaMalloc(&block_value, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&BlockElementMask, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&BlockElementNum, block_num * sizeof(uint32_t));
    cudaMemset(TCOffset, 0, sizeof(uint32_t));
    thrust::fill_n(thrust::device, BlockElementMask, edge_index.size(1), 1);
    thrust::transform(thrust::device, SparseAidx, SparseAidx + edge_index.size(1), edge_idx1, block_value, 
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
    return {RowWindowOffset, RowWindowRowOffset, RowWindowColOffset, BlockMask, SparseAToX};
}

std::vector<at::Tensor> process_DTC_short_mask(at::Tensor edge_index, int block_high, int block_width, int num_nodes, bool balance)
{
    uint64_t *value0, *value1;
    int* rowwindow_value, *X_col_id, *edge_idx1;
    cudaMalloc(&value0, edge_index.size(1) * sizeof(uint64_t));
    cudaMalloc(&value1, edge_index.size(1) * sizeof(uint64_t));
    cudaMalloc(&edge_idx1, edge_index.size(1) * sizeof(int));
    thrust::copy_n(thrust::device, edge_index.data_ptr<int>() + edge_index.size(1), 
                    edge_index.size(1), edge_idx1);
    // after get edge_index_1

    thrust::transform(thrust::device, edge_index.data_ptr<int>(), 
                      edge_index.data_ptr<int>() + edge_index.size(1), edge_index.data_ptr<int>() + edge_index.size(1),
                      value0, [=] __device__ (int e0, int e1) {return ((uint64_t)(e1 / block_high) * (uint64_t)(num_nodes) + (uint64_t)(e0));});
    thrust::copy_n(thrust::device, value0, edge_index.size(1), value1);
    thrust::stable_sort_by_key(thrust::device, value0, value0 + edge_index.size(1), edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value1, value1 + edge_index.size(1), edge_idx1);
    cudaMalloc(&rowwindow_value, edge_index.size(1) * sizeof(int));
    thrust::transform(thrust::device, edge_idx1, edge_idx1 + edge_index.size(1),
                      rowwindow_value, [=] __device__ (int e) {return e / block_high;});
    cudaFree(value0);
    cudaFree(value1);
    // after sort edge_index_0 and edge_index_1 according to rowwindow_id(rowwindow_value) and col_id
    int* SparseAidx, *SparseAidx_temp, *SparseAidx_temp1;
    cudaMalloc(&SparseAidx, edge_index.size(1) * sizeof(int));
    cudaMalloc(&SparseAidx_temp, edge_index.size(1) * sizeof(int));
    thrust::adjacent_difference(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>() + edge_index.size(1), 
                                SparseAidx, [=]__device__(int x1, int x0) {return (x1 == x0) ? 0 : 1;});
    thrust::adjacent_difference(thrust::device, rowwindow_value, rowwindow_value + edge_index.size(1), SparseAidx_temp,
                                [=]__device__(int x1, int x0) {return (x1 == x0) ? 0 : 1;});
    thrust::transform(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), SparseAidx, SparseAidx, 
                    [=]__device__(int x0, int x1) {return (x0 | x1);});
    cudaMemset(SparseAidx, 1, 1);
    thrust::copy_n(thrust::device, SparseAidx, edge_index.size(1), SparseAidx_temp);
    thrust::inclusive_scan_by_key(thrust::device, rowwindow_value, rowwindow_value + edge_index.size(1), SparseAidx, SparseAidx, 
                                thrust::equal_to<int>(), thrust::plus<int>());
    cudaMalloc(&SparseAidx_temp1, edge_index.size(1) * sizeof(int));
    thrust::inclusive_scan(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), SparseAidx_temp1);
    thrust::copy_n(thrust::device, SparseAidx_temp1, edge_index.size(1), SparseAidx_temp);
    // after record nonzero-element's col_id in sparse matrix
    cudaMalloc(&X_col_id, edge_index.size(1) * sizeof(int));
    thrust::copy_n(thrust::device, edge_index.data_ptr<int>(), edge_index.size(1), X_col_id);
    auto unique_X_col_id_end = thrust::unique_by_key(thrust::device, SparseAidx_temp1, SparseAidx_temp1 + edge_index.size(1), 
                            X_col_id, thrust::equal_to<int>());
    auto unique_rowwindow_end = thrust::unique_by_key(thrust::device, SparseAidx_temp, SparseAidx_temp + edge_index.size(1), 
                            rowwindow_value, thrust::equal_to<int>());
    // after de-duplicate X_col_id
    int rowwindow_value_len = unique_rowwindow_end.second - rowwindow_value;
    // printf("current rowwindow_value_len: %d\n", rowwindow_value_len);
    int *RowWindowMask, *RowWindowNum_temp, *RowWindowNum, *unique_rowwindow_value;
    cudaMalloc(&RowWindowMask, rowwindow_value_len * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowMask, rowwindow_value_len, 1);
    cudaMalloc(&RowWindowNum_temp, rowwindow_value_len * sizeof(int));
    cudaMalloc(&unique_rowwindow_value, rowwindow_value_len * sizeof(int));
    auto new_end = thrust::reduce_by_key(thrust::device, rowwindow_value, unique_rowwindow_end.second, 
                            RowWindowMask, unique_rowwindow_value, RowWindowNum_temp, thrust::equal_to<int>(), 
                            thrust::plus<int>());
    //////////////////////////////////////////////////////////////////////////
    // int RowWindowNum_temp_len = new_end.second - RowWindowNum_temp;
    // printf("current RowWindowNum_temp_len: %d\n", RowWindowNum_temp_len);
    // auto max_rowwindow_value = thrust::max_element(thrust::device, unique_rowwindow_value, unique_rowwindow_value + RowWindowNum_temp_len);
    // int max_rowwindow_value_value, last_rowwindow_value;
    // cudaMemcpy(&max_rowwindow_value_value, max_rowwindow_value, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&last_rowwindow_value, unique_rowwindow_value + RowWindowNum_temp_len - 1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_rowwindow_value_value: %d, last_rowwindow_value: %d\n", max_rowwindow_value_value, last_rowwindow_value);
    ///////////////////////////////////////////////////////////////////////
    cudaMalloc(&RowWindowNum, (num_nodes + block_high - 1) / block_high * sizeof(int));
    thrust::fill_n(thrust::device, RowWindowNum, (num_nodes + block_high - 1) / block_high, 0);
    thrust::scatter(thrust::device, RowWindowNum_temp, new_end.second, unique_rowwindow_value, RowWindowNum);
    at::Tensor RowWindowOffset = at::empty({1 + (num_nodes + block_high - 1) / block_high}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(RowWindowOffset.data_ptr<int>(), 0, sizeof(int));
    thrust::transform(thrust::device, RowWindowNum, RowWindowNum + (num_nodes + block_high - 1) / block_high,
                    RowWindowOffset.data_ptr<int>() + 1, [=]__device__(int x) {return (x + block_width - 1) / block_width;});
    thrust::inclusive_scan(thrust::device, RowWindowOffset.data_ptr<int>() + 1, RowWindowOffset.data_ptr<int>() + 1 + (num_nodes + block_high - 1) / block_high,
                    RowWindowOffset.data_ptr<int>() + 1);
    // printf("after getting RowWindowOffset\n");
    int block_num;
    cudaMemcpy(&block_num, RowWindowOffset.data_ptr<int>() + (num_nodes + block_high - 1) / block_high, sizeof(int), cudaMemcpyDeviceToHost);
    int *align_offset, *align_idx;
    cudaMalloc(&align_offset, (num_nodes + block_high - 1) / block_high * sizeof(int));
    cudaMalloc(&align_idx, rowwindow_value_len * sizeof(int));
    thrust::fill_n(thrust::device, align_idx, rowwindow_value_len, 0);
    // printf("rowwindow_value_len: %d, SparseAidx_temp_len: %d\n", rowwindow_value_len, unique_rowwindow_end.first-SparseAidx_temp);
    // printf("RowWindowNum_temp_len: %d, align_offset_len: %d\n", new_end.second - RowWindowNum_temp, (num_nodes + block_high - 1) / block_high);
    thrust::transform(thrust::device, RowWindowNum_temp, new_end.second, align_offset, 
                    [=]__device__(int x) {return (x + block_width - 1) / block_width * block_width - x;});
    // printf("after getting align_offset\n");
    thrust::inclusive_scan(thrust::device, RowWindowNum_temp, new_end.second, RowWindowNum_temp);
    // int last_WindowNum_temp;
    // cudaMemcpy(&last_WindowNum_temp, new_end.second-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("after last RowWindowNum_temp: %d\n", last_WindowNum_temp);
    thrust::scatter(thrust::device, align_offset, align_offset + (num_nodes + block_high - 1) / block_high - 1, 
                    RowWindowNum_temp, align_idx);
    // printf("after scatter.\n");
    thrust::inclusive_scan(thrust::device, align_idx, align_idx + rowwindow_value_len, align_idx);
    // printf("after inclusive_scan.\n");
    thrust::transform(thrust::device, align_idx, align_idx + rowwindow_value_len, SparseAidx_temp, align_idx,
                    [=]__device__(int x0, int x1) {return (x0 + x1 - 1);});
    // printf("after getting align_idx\n");
    at::Tensor SparseAToX = at::empty({block_num * block_width}, device(torch::kCUDA).dtype(torch::kInt32));
    thrust::fill_n(thrust::device, SparseAToX.data_ptr<int>(), block_width * block_num, num_nodes);
////////////////////////////////////////////////////////////////////////////////////////////
    // auto max_align_idx = thrust::max_element(thrust::device, align_idx, align_idx + rowwindow_value_len);
    // int max_align_idx_value;
    // cudaMemcpy(&max_align_idx_value, max_align_idx, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_align_idx_value: %d, SparseAToX_len: %d\n", max_align_idx_value, block_num * block_width);
////////////////////////////////////////////////////////////////////////////////////////////
    thrust::scatter(thrust::device, X_col_id, unique_X_col_id_end.second, align_idx, SparseAToX.data_ptr<int>());
    // printf("after scatter\n");
    cudaFree(SparseAidx_temp);
    cudaFree(SparseAidx_temp1);
    cudaFree(X_col_id);
    cudaFree(rowwindow_value);
    cudaFree(unique_rowwindow_value);
    cudaFree(RowWindowMask);
    cudaFree(RowWindowNum_temp);
    cudaFree(RowWindowNum);
    //after generate aligned SparseAToX and RowWindowOffset
    at::Tensor BitMask_col = at::empty({block_num*(block_high/8)}, device(torch::kCUDA).dtype(torch::kUInt8));
    at::Tensor BitMask_RowOffset = at::empty({block_num+1}, device(torch::kCUDA).dtype(torch::kInt32));
    // printf("current block_num: %d\n", block_num);
    uint32_t *block_value, *BlockElementMask, *TCOffset;
    cudaMalloc(&block_value, edge_index.size(1) * sizeof(uint32_t));
    cudaMemset(BitMask_RowOffset.data_ptr<int>(), 0, sizeof(uint32_t));
    thrust::transform(thrust::device, SparseAidx, SparseAidx + edge_index.size(1), edge_idx1, block_value, 
                    [=] __device__ (int e0, int e1) {return ((uint32_t) (e1 / block_high) * (uint32_t) (num_nodes * block_high) + (uint32_t)((e0-1) / block_width * block_high) + (e1 % block_high));});
    thrust::sort(thrust::device, block_value, block_value + edge_index.size(1));
    auto block_value_end = thrust::unique(thrust::device, block_value, block_value + edge_index.size(1));
    int block_value_len = block_value_end - block_value;
    cudaMalloc(&BlockElementMask, block_value_len * sizeof(uint32_t));
    thrust::fill_n(thrust::device, BlockElementMask, block_value_len, 1);
    thrust::transform(thrust::device, block_value, block_value_end, block_value, [=]__device__(uint32_t x) {return (x / block_high);});
    thrust::reduce_by_key(thrust::device, block_value, block_value_end, BlockElementMask,
                            block_value, BitMask_RowOffset.data_ptr<int>() + 1, thrust::equal_to<uint32_t>(), 
                            [=]__device__ (uint32_t x1, uint32_t x0) {return (int)(x1 + x0);});
    // printf("after reduce_by_key 0\n");
    thrust::inclusive_scan(thrust::device, BitMask_RowOffset.data_ptr<int>() + 1, 
                            BitMask_RowOffset.data_ptr<int>() + 1 + block_num, BitMask_RowOffset.data_ptr<int>() + 1);
    cudaMalloc(&BlockElementMask, edge_index.size(1) * sizeof(uint32_t));
    cudaMalloc(&block_value, edge_index.size(1) * sizeof(uint32_t));
    thrust::transform(thrust::device, SparseAidx, SparseAidx + edge_index.size(1), edge_idx1, block_value,
                    [=] __device__ (int e0, int e1) {return ((uint32_t)(e1 / block_high) * (uint32_t) num_nodes + (uint32_t)((e0-1) / block_width));});
    thrust::fill_n(thrust::device, BlockElementMask, edge_index.size(1), 1);
    cudaMalloc(&TCOffset, (block_num + 1) * sizeof(uint32_t));
    cudaMemset(TCOffset, 0, sizeof (uint32_t));
    thrust::reduce_by_key(thrust::device, block_value, block_value + edge_index.size(1), BlockElementMask,
                            block_value, TCOffset + 1, thrust::equal_to<uint32_t>(), thrust::plus<uint32_t>());
    // printf("after reduce_by_key 1\n");
    thrust::inclusive_scan(thrust::device, TCOffset + 1, TCOffset + 1 + block_num, TCOffset + 1);
    int mask_num;
    cudaMemcpy(&mask_num, BitMask_RowOffset.data_ptr<int>() + block_num, sizeof(int), cudaMemcpyDeviceToHost);
    //BlockColMask
    // printf("current mask_num: %d\n", mask_num);
    at::Tensor BitMask_row = at::empty({mask_num*(block_width/8)}, device(torch::kCUDA).dtype(torch::kUInt8));
    int grid = (block_num + 127) / 128;
    // block_high, block_width must be power of 2 (both >= 8) (block_high * block_width <= 512)
    BlockMaskShortKernel<<<grid, 128>>>(SparseAidx, 
                                    edge_idx1, 
                                    TCOffset, 
                                    BitMask_RowOffset.data_ptr<int>(), 
                                    BitMask_col.data_ptr<uint8_t>(),
                                    BitMask_row.data_ptr<uint8_t>(),
                                    block_high, block_width, block_num);
    cudaFree(block_value);
    cudaFree(BlockElementMask);
    cudaFree(TCOffset);
    cudaFree(SparseAidx);
    cudaFree(edge_idx1);
    return {RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, SparseAToX};
}

__global__ void sgt_short_mask_kernel(
    const int* __restrict__ row_pointers,
    const int* __restrict__ column_index,
    const int* __restrict__ RowWindowOffset,
    const int* __restrict__ edgeToColumn,
    const int* __restrict__ edgeToRow,
    int* __restrict__ BitMask_RowOffset,
    uint8_t* __restrict__ BitMask_col,
    int* __restrict__ SparseAToX,
    int block_high,
    int block_width,
    int node_num
) {
    int bid = blockIdx.x;
    int element_start = row_pointers[bid*16];
    int elemnet_end = row_pointers[min(bid*16+16, node_num)];
    int block_start = RowWindowOffset[bid];
    int block_end = RowWindowOffset[bid+1];
    int col_num = block_high / 8;
    int row_num = block_width / 8;
    __shared__ uint8_t col_mask[4];
    __shared__ uint8_t row_mask[64];
    for (int i = 0; i < (block_end - block_start); i++) {
        if(threadIdx.x < col_num) {
            col_mask[threadIdx.x] = 0;
        }
        if (threadIdx.x < row_num * block_high) {
            row_mask[threadIdx.x] = 0;
        }
        __syncthreads();
        for (int idx = element_start + threadIdx.x; idx < elemnet_end; idx += blockDim.x) {
            int col = edgeToColumn[idx];
            if (col >= i * block_width && col < (i+1) * block_width) {
                int local_row = edgeToRow[idx] % block_high;
                atomicOr((uint32_t*)&col_mask[0], (1<<local_row));
                int local_col = edgeToColumn[idx] % block_width;
                SparseAToX[(i+block_start)*block_width+local_col] = column_index[idx];
                int col = (local_row * block_width + local_col) & 31;
                int row = (local_row * block_width + local_col) >> 5;
                atomicOr((uint32_t*)&row_mask[row*4], (1<<col));
            }
        }
        __syncthreads();
        if(threadIdx.x < col_num) {
            BitMask_col[(block_start+i)*col_num+threadIdx.x] = col_mask[threadIdx.x];
        }
        int row_num = __popc(*(uint32_t*)&col_mask[0]);
        BitMask_RowOffset[block_start+i+1] = row_num;
    }
}

__global__ void gen_rowmask_kernel(
    const int* __restrict__ row_pointers,
    const int* __restrict__ column_index,
    const int* __restrict__ RowWindowOffset,
    const int* __restrict__ edgeToColumn,
    const int* __restrict__ edgeToRow,
    const int* __restrict__ BitMask_RowOffset,
    const uint8_t* __restrict__ BitMask_col,
    uint8_t* __restrict__ BitMask_row,
    int block_high,
    int block_width,
    int node_num
){
    int bid = blockIdx.x;
    int element_start = row_pointers[bid*16];
    int elemnet_end = row_pointers[min(bid*16+16, node_num)];
    int block_start = RowWindowOffset[bid];
    int block_end = RowWindowOffset[bid+1];
    int col_num = block_high / 8;
    int row_num = block_width / 8;
    __shared__ uint8_t row_mask[64];
    __shared__ uint8_t col_mask[4];
    for (int i = 0; i < (block_end - block_start); i++) {
        if (threadIdx.x < row_num * block_high) {
            row_mask[threadIdx.x] = 0;
        }
        __syncthreads();
        for (int idx = element_start + threadIdx.x; idx < elemnet_end; idx += blockDim.x) {
            int col = edgeToColumn[idx];
            if (col >= i * block_width && col < (i+1) * block_width) {
                int local_row = edgeToRow[idx] % block_high;
                int col = (local_row * block_width + col) & 31;
                int row = (local_row * block_width + col) >> 5;
                atomicOr((uint32_t*)&row_mask[row*4], (1<<col));
            }
        }
        if (threadIdx.x < col_num) 
            col_mask[threadIdx.x] = BitMask_col[(block_start+i)*col_num+threadIdx.x];
        __syncthreads();
        if (threadIdx.x < row_num * block_high) {
            int row = threadIdx.x / row_num;
            int offset = __popc(*(uint32_t*)&col_mask[0]&((1<<row)-1));
            if (*(uint32_t*)&col_mask[0]&(1<<row))
                BitMask_row[BitMask_RowOffset[block_start+i]+offset] = row_mask[threadIdx.x];
        }
    }
}

std::vector<at::Tensor> SGT_short_Mask(
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor blockPartition,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow,
    int block_high,
    int block_width
) {
    int num_nodes = row_pointers.size(0) - 1;
    int num_edges = column_index.size(0);

    auto RowWindowOffset = at::empty({1+blockPartition.size(0)}, blockPartition.options().dtype(torch::kInt));
    cudaMemset(RowWindowOffset.data_ptr<int>(), 0, sizeof(int));
    thrust::inclusive_scan(thrust::device, blockPartition.data_ptr<int>(), blockPartition.data_ptr<int>() + blockPartition.size(0), 
                            RowWindowOffset.data_ptr<int>() + 1);
    int block_num;
    cudaMemcpy(&block_num, RowWindowOffset.data_ptr<int>() + blockPartition.size(0), sizeof(int), cudaMemcpyDeviceToHost);
    auto BitMask_RowOffset = at::empty({block_num+1}, blockPartition.options().dtype(torch::kInt));
    int row_num = block_high / 8;
    auto BitMask_col = at::empty({row_num*block_num}, device(torch::kCUDA).dtype(torch::kUInt8));
    auto SparseAToX = at::empty({block_num*block_width}, device(torch::kCUDA).dtype(torch::kInt));
    thrust::fill_n(thrust::device, SparseAToX.data_ptr<int>(), block_num*block_width, num_nodes);
    cudaMemset(BitMask_RowOffset.data_ptr<int>(), 0, sizeof(int));
    int blocks = (num_nodes + 15) / 16;
    sgt_short_mask_kernel<<<blocks, 128>>>(
        row_pointers.data_ptr<int>(), column_index.data_ptr<int>(), 
        RowWindowOffset.data_ptr<int>(), edgeToColumn.data_ptr<int>(), 
        edgeToRow.data_ptr<int>(), BitMask_RowOffset.data_ptr<int>(), 
        BitMask_col.data_ptr<uint8_t>(), SparseAToX.data_ptr<int>(),
        block_high, block_width, num_nodes);
    thrust::inclusive_scan(thrust::device, BitMask_RowOffset.data_ptr<int>()+1, BitMask_RowOffset.data_ptr<int>()+1+block_num, 
                            BitMask_RowOffset.data_ptr<int>()+1);
    int row_mask_num;
    cudaMemcpy(&row_mask_num, BitMask_RowOffset.data_ptr<int>()+block_num, sizeof(int), cudaMemcpyDeviceToHost);
    auto BitMask_row = at::empty({row_mask_num}, device(torch::kCUDA).dtype(torch::kUInt8));
    gen_rowmask_kernel<<<blocks, 128>>>(
        row_pointers.data_ptr<int>(), column_index.data_ptr<int>(), 
        RowWindowOffset.data_ptr<int>(), edgeToColumn.data_ptr<int>(), 
        edgeToRow.data_ptr<int>(), BitMask_RowOffset.data_ptr<int>(), 
        BitMask_col.data_ptr<uint8_t>(), BitMask_row.data_ptr<uint8_t>(),
        block_high, block_width, num_nodes);
    return {RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, SparseAToX};
}
