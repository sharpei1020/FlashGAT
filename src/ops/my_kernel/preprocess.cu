#include "preprocess.cuh"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform_scan.h>
#include <omp.h>
#include "time.cuh"

#define FULL_MASK 0xffffffff

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

std::vector<at::Tensor> process_CSR(at::Tensor edge_index, int num_nodes)
{
    int num_edges = edge_index.size(1);
    auto index = at::empty({num_edges}, device(torch::kCUDA).dtype(torch::kInt));
    thrust::copy(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+num_edges, index.data_ptr<int>());
    int *edge_index_1;
    cudaMalloc(&edge_index_1, num_edges*sizeof(int));
    thrust::copy(thrust::device, edge_index.data_ptr<int>()+num_edges, edge_index.data_ptr<int>()+2*num_edges, edge_index_1);
    uint64_t *value, *value0;
    cudaMalloc(&value, num_edges*sizeof(uint64_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+num_edges,
                        edge_index.data_ptr<int>()+num_edges, value, [=]__device__(int e0, int e1) 
                        {return (uint64_t)e1*(uint64_t)num_nodes+(uint64_t)e0;});
    cudaMalloc(&value0, num_edges*sizeof(uint64_t));
    thrust::copy(thrust::device, value, value+num_edges, value0);
    thrust::stable_sort_by_key(thrust::device, value, value+num_edges, index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value0, value0+num_edges, edge_index_1);
    cudaFree(value);
    cudaFree(value0);
    int *mask, *unique_edge_index_1, *row_num;
    cudaMalloc(&mask, num_edges*sizeof(int));
    cudaMalloc(&unique_edge_index_1, num_nodes*sizeof(int));
    cudaMalloc(&row_num, num_nodes*sizeof(int));
    thrust::fill_n(thrust::device, mask, num_edges, 1);
    auto row_offset = at::empty({num_nodes+1}, device(torch::kCUDA).dtype(torch::kInt));
    auto end = thrust::reduce_by_key(thrust::device, edge_index_1, edge_index_1+num_edges, mask, 
                        unique_edge_index_1, row_num);
    thrust::fill_n(thrust::device, row_offset.data_ptr<int>(), num_nodes+1, 0);
    thrust::scatter(thrust::device, row_num, end.second, unique_edge_index_1, row_offset.data_ptr<int>()+1);
    thrust::inclusive_scan(thrust::device, row_offset.data_ptr<int>()+1, row_offset.data_ptr<int>()+1+num_nodes, row_offset.data_ptr<int>()+1);
    cudaFree(mask);
    cudaFree(unique_edge_index_1);
    cudaFree(row_num);
    return {row_offset, index};
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
}

__global__ void BlockMaskShortKernel_2block(
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
    int n = 8 / block_high;
    if (n*tid >= block_num) return;
    uint32_t start = TCOffset[n*tid];
    uint32_t end = TCOffset[min(n*(tid+1),block_num)];
    int write_start = BitMask_RowOffset[n*tid];
    // int write_end = BitMask_RowOffset[min(n*(tid + 1),block_num)];
    int width_num = block_width / 8;
    uint8_t col_mask = 0;
    uint8_t row_mask[64] = {0};
    for (uint32_t i = start; i < end; i++) {
        int block_col = ((edge_index_0[i] - 1) & (block_width - 1));
        int block_row = (edge_index_1[i] & (block_high - 1));
        for (int j = 1; j < n; j++) {
            if ((i < TCOffset[n*tid+j])||(n*tid+j>block_num))
                break;
            block_row += block_high;
        }
        col_mask |= (1 << (block_row & 7));
        row_mask[block_row*width_num + (block_col >> 3)] |= (1 << (block_col & 7));
    }
    BlockMaskCol[tid] = col_mask;
    int m = __popc(col_mask);
    for (int i = 1; i <= m; i++) {
        int j = __fns(col_mask, 0, i);
        if (width_num == 1)
            BlockMaskRow[write_start + i - 1] = row_mask[j];
        else if (width_num == 2)
            *(uint16_t*)(&BlockMaskRow[(write_start + i - 1) * 2]) = *(uint16_t*)(&row_mask[j*2]);
        else if (width_num == 4)
            *(uint32_t*)(&BlockMaskRow[(write_start + i - 1) * 4]) = *(uint32_t*)(&row_mask[j*4]);
    }
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
    // printf("after transform0.\n");
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
    // printf("after transform1.\n");
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
    // auto max_RowWindowNum_temp = thrust::max_element(thrust::device, RowWindowNum_temp, new_end.second);
    // int max_RowWindowNum_temp_value;
    // cudaMemcpy(&max_RowWindowNum_temp_value, max_RowWindowNum_temp, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_RowWindowNum_temp_value: %d\n", max_RowWindowNum_temp_value);
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
    // printf("after transform2.\n");
    thrust::inclusive_scan(thrust::device, RowWindowOffset.data_ptr<int>() + 1, RowWindowOffset.data_ptr<int>() + 1 + (num_nodes + block_high - 1) / block_high,
                    RowWindowOffset.data_ptr<int>() + 1);
    // printf("after getting RowWindowOffset\n");
    int block_num;
    cudaMemcpy(&block_num, RowWindowOffset.data_ptr<int>() + (num_nodes + block_high - 1) / block_high, sizeof(int), cudaMemcpyDeviceToHost);
    /////////////////////////////////////////////////
    // printf("block_num: %d\n", block_num);
    /////////////////////////////////////////////////
    int *align_offset, *align_idx;
    cudaMalloc(&align_offset, (num_nodes + block_high - 1) / block_high * sizeof(int));
    cudaMalloc(&align_idx, rowwindow_value_len * sizeof(int));
    thrust::fill_n(thrust::device, align_idx, rowwindow_value_len, 0);
    // printf("rowwindow_value_len: %d, SparseAidx_temp_len: %d\n", rowwindow_value_len, unique_rowwindow_end.first-SparseAidx_temp);
    // printf("RowWindowNum_temp_len: %d, align_offset_len: %d\n", new_end.second - RowWindowNum_temp, (num_nodes + block_high - 1) / block_high);
    thrust::transform(thrust::device, RowWindowNum_temp, new_end.second, align_offset, 
                    [=]__device__(int x) {return (x + block_width - 1) / block_width * block_width - x;});
    // printf("after transform3.\n");
    // printf("after getting align_offset\n");
    thrust::inclusive_scan(thrust::device, RowWindowNum_temp, new_end.second, RowWindowNum_temp);
    // int last_WindowNum_temp;
    // cudaMemcpy(&last_WindowNum_temp, new_end.second-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("after last RowWindowNum_temp: %d\n", last_WindowNum_temp);
    thrust::scatter(thrust::device, align_offset, align_offset + (new_end.second - RowWindowNum_temp) - 1, 
                    RowWindowNum_temp, align_idx);
    /////////////////////////////////////////////////////////
    // auto max_align_offset = thrust::max_element(thrust::device, align_offset, align_offset+(num_nodes + block_high - 1) / block_high);
    // int RowWindowNum_temp_end, max_align_offset_value;
    // cudaMemcpy(&max_align_offset_value, max_align_offset, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&RowWindowNum_temp_end, new_end.second-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("RowWindowNum_temp_end: %d, max_align_offset: %d, align_idx_len: %d\n", RowWindowNum_temp_end, max_align_offset_value, rowwindow_value_len);
    //////////////////////////////////////////////////////////
    // printf("after scatter.\n");
    thrust::inclusive_scan(thrust::device, align_idx, align_idx + rowwindow_value_len, align_idx);
    /////////////////////////////////////////////
    // int max_align_idx_value0;
    // cudaMemcpy(&max_align_idx_value0, align_idx+rowwindow_value_len-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_align_idx_value: %d\n", max_align_idx_value0);
    //////////////////////////////////////////////
    // printf("after inclusive_scan.\n");
    thrust::transform(thrust::device, align_idx, align_idx + rowwindow_value_len, SparseAidx_temp, align_idx,
                    [=]__device__(int x0, int x1) {return (x0 + x1 - 1);});
    // printf("after transform4.\n");
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
    int colmask_num = (block_high/8)>0?block_num*(block_high/8):(block_num+(8/block_high-1))/(8/block_high);
    at::Tensor BitMask_col = at::empty({colmask_num}, device(torch::kCUDA).dtype(torch::kUInt8));
    at::Tensor BitMask_RowOffset = at::empty({block_num+1}, device(torch::kCUDA).dtype(torch::kInt32));
    // printf("current block_num: %d\n", block_num);
    uint32_t *block_value, *BlockElementMask, *TCOffset;
    cudaMalloc(&block_value, edge_index.size(1) * sizeof(uint32_t));
    cudaMemset(BitMask_RowOffset.data_ptr<int>(), 0, sizeof(uint32_t));
    thrust::transform(thrust::device, SparseAidx, SparseAidx + edge_index.size(1), edge_idx1, block_value, 
                    [=] __device__ (int e0, int e1) {return ((uint32_t) (e1 / block_high) * (uint32_t) (num_nodes * block_high) + (uint32_t)((e0-1) / block_width * block_high) + (e1 % block_high));});
    // printf("after transform5.\n");
    ///////////////////////////////////////////////
    // uint32_t* max_block_value_ptr = thrust::max_element(thrust::device, block_value, block_value+edge_index.size(1));
    // uint32_t max_block_value;
    // cudaMemcpy(&max_block_value, max_block_value_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // printf("max_block_value: %ld\n", max_block_value);
    ///////////////////////////////////////////////
    thrust::sort(thrust::device, block_value, block_value + edge_index.size(1));
    auto block_value_end = thrust::unique(thrust::device, block_value, block_value + edge_index.size(1));
    int block_value_len = block_value_end - block_value;
    cudaMalloc(&BlockElementMask, block_value_len * sizeof(uint32_t));
    thrust::fill_n(thrust::device, BlockElementMask, block_value_len, 1);
    thrust::transform(thrust::device, block_value, block_value_end, block_value, [=]__device__(uint32_t x) {return (x / block_high);});
    // printf("after transform6.\n");
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
    // printf("after transform7.\n");
    thrust::fill_n(thrust::device, BlockElementMask, edge_index.size(1), 1);
    cudaMalloc(&TCOffset, (block_num + 1) * sizeof(uint32_t));
    cudaMemset(TCOffset, 0, sizeof (uint32_t));
    thrust::reduce_by_key(thrust::device, block_value, block_value + edge_index.size(1), BlockElementMask,
                            block_value, TCOffset + 1, thrust::equal_to<uint32_t>(), thrust::plus<uint32_t>());
    // printf("after reduce_by_key 1\n");
    thrust::inclusive_scan(thrust::device, TCOffset + 1, TCOffset + 1 + block_num, TCOffset + 1);
    int mask_num;
    cudaMemcpy(&mask_num, BitMask_RowOffset.data_ptr<int>() + block_num, sizeof(int), cudaMemcpyDeviceToHost);
    ///////////////////////////////////////////////////////////////////
    // int* max_BitMask_RowOffset_ptr = thrust::max_element(thrust::device, BitMask_RowOffset.data_ptr<int>(), BitMask_RowOffset.data_ptr<int>()+block_num+1);
    // int max_BitMask_RowOffset;
    // cudaMemcpy(&max_BitMask_RowOffset, max_BitMask_RowOffset_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_BitMask_RowOffset: %d\n", max_BitMask_RowOffset);
    ///////////////////////////////////////////////////////////////////
    //BlockColMask
    // printf("current mask_num: %d\n", mask_num);
    at::Tensor BitMask_row = at::empty({mask_num*(block_width/8)}, device(torch::kCUDA).dtype(torch::kUInt8));
    if (block_high >= 8) {
        int grid = (block_num + 127) / 128;
        // block_high, block_width must be power of 2 (both >= 8) (block_high * block_width <= 512)
        BlockMaskShortKernel<<<grid, 128>>>(SparseAidx, 
                                    edge_idx1, 
                                    TCOffset, 
                                    BitMask_RowOffset.data_ptr<int>(), 
                                    BitMask_col.data_ptr<uint8_t>(),
                                    BitMask_row.data_ptr<uint8_t>(),
                                    block_high, block_width, block_num);
    } else {
        int n = 8 / block_high;
        int grid = ((block_num + (n-1)) / n + 127) / 128;
        // block_high <= 4
        BlockMaskShortKernel_2block<<<grid, 128>>>(SparseAidx, 
                                    edge_idx1, 
                                    TCOffset, 
                                    BitMask_RowOffset.data_ptr<int>(), 
                                    BitMask_col.data_ptr<uint8_t>(),
                                    BitMask_row.data_ptr<uint8_t>(),
                                    block_high, block_width, block_num);
    }
    cudaFree(block_value);
    cudaFree(BlockElementMask);
    cudaFree(TCOffset);
    cudaFree(SparseAidx);
    cudaFree(edge_idx1);
    return {RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, SparseAToX};
}

__global__ void select_blocksize(
    const int* __restrict__ rowwindow_edgenum_16,
    const int* __restrict__ col_num_16,
    const int* __restrict__ rowwindow_edgenum_8,
    const int* __restrict__ col_num_8,
    uint8_t* __restrict__ block_high_array,
    uint8_t* __restrict__ block_width_array,
    int* __restrict__ mixed_edgenum_array,
    int* __restrict__ mixed_colid_array,
    int* __restrict__ mixed_colid_array_padding,
    int* __restrict__ rowwindow_id,
    int* __restrict__ rowwindow_type,
    float boundary,
    int len, int len_) 
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=len) return;
    int col_16 = col_num_16[idx];
    int col_8_0 = col_num_8[2*idx];
    int col_8_1 = (2*idx+1>=len_)?0:col_num_8[2*idx+1];
    // int block_width_16; 
    // int block_width_8_0, block_width_8_1;
    // if (boundary>7.f) {
    int block_width_16 = 16; //((col_16 + 15) / 16 == 1) ? 8 : 16;
    int block_width_8_0 = 16; //(col_8_0 <= 8) ? 8 : 16;
    int block_width_8_1 = 16; //(col_8_1 <= 8) ? 8 : 16;
    // } else {
    //     block_width_16 = 16;
    //     block_width_8_0 = (col_8_0 > 16) ? 16 : 8;
    //     block_width_8_1 = (col_8_1 > 16) ? 16 : 8;
    // }
    int padding_col_16 = (col_16 + block_width_16 - 1) / block_width_16 * block_width_16;
    int padding_col_8_0 = (col_8_0 + block_width_8_0 - 1) / block_width_8_0 * block_width_8_0;
    int padding_col_8_1 = (col_8_1 + block_width_8_1 - 1) / block_width_8_1 * block_width_8_1;
    float select = (float)(16*col_16-8*(col_8_0+col_8_1))/(float)(col_8_0+col_8_1-col_16);
    if (select>=boundary) {
        block_high_array[2*idx] = (uint8_t)8;
        block_width_array[2*idx] = (uint8_t)block_width_8_0;
        mixed_edgenum_array[2*idx] = rowwindow_edgenum_8[2*idx];
        mixed_colid_array[2*idx] = col_8_0;
        mixed_colid_array_padding[2*idx] = padding_col_8_0;
        rowwindow_id[2*idx] = 1;
        rowwindow_type[2*idx] = block_width_8_0/8-1;
        if (2*idx+1<len_) {
            block_high_array[2*idx+1] = (uint8_t)8;
            block_width_array[2*idx+1] = (uint8_t)block_width_8_1;
            mixed_edgenum_array[2*idx+1] = rowwindow_edgenum_8[2*idx+1];
            mixed_colid_array[2*idx+1] = col_8_1;
            mixed_colid_array_padding[2*idx+1] =  padding_col_8_1;
            rowwindow_id[2*idx+1] = 1;
            rowwindow_type[2*idx+1] = block_width_8_1/8-1;
        }
    } else {
        block_high_array[2*idx] = (uint8_t)16;
        block_width_array[2*idx] = (uint8_t)block_width_16;
        mixed_edgenum_array[2*idx] = rowwindow_edgenum_16[idx];
        mixed_colid_array[2*idx] = col_16;
        mixed_colid_array_padding[2*idx] = padding_col_16;
        rowwindow_id[2*idx] = 1;
        rowwindow_type[2*idx] = (block_width_16/8-1)+2;
    }
}

__global__ void transform_5(
    const int* __restrict__ edge_index_0,
    const int* __restrict__ edge_index_1,
    const int* __restrict__ segment_key,
    const uint8_t* __restrict__ block_high,
    const uint8_t* __restrict__ block_width,
    uint64_t* __restrict__ value_out,
    int node_num_col,
    int len)
{
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid >= len) return;
    value_out[tid] = (uint64_t)segment_key[tid]*(uint64_t)(node_num_col*2)+
                    (uint64_t)((edge_index_0[tid]-1)/(int)block_width[tid])*(uint64_t)(block_high[tid])+
                    (uint64_t)(edge_index_1[tid]%(int)block_high[tid]);
}

__global__ void mixed_BlockMaskShortKernel(
    const int* __restrict__ edge_index_0,
    const int* __restrict__ edge_index_1,
    const int* __restrict__ TCOffset,
    const int* __restrict__ TCColMaskOffset,
    const int* __restrict__ BlockRowMaskOffset,
    const uint8_t* __restrict__ block_high,
    const uint8_t* __restrict__ block_width,
    uint8_t* __restrict__ ColMask,
    uint8_t* __restrict__ RowMask,
    int block_num)
{
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid >= block_num) return;
    int start = TCOffset[tid];
    int end = TCOffset[tid+1];
    int write_start_col = BlockRowMaskOffset[tid];
    int write_start_row = TCColMaskOffset[tid];
    int width_num = block_width[tid]/8;
    uint8_t col_mask[2] = {0};
    uint8_t row_mask[32] = {0};
    int high = block_high[tid];
    int width = block_width[tid];
    for (uint32_t i = start; i < end; i++) {
        int local_row_idx = (edge_index_1[i]&(high-1));
        col_mask[local_row_idx>>3] |= (1<<(local_row_idx&7));
        int local_col_idx = ((edge_index_0[i]-1)&(width-1));
        row_mask[local_row_idx*width_num+(local_col_idx>>3)] |= (1<<(local_col_idx&7));
    }
    if (high == 8) 
        ColMask[write_start_row] = col_mask[0];
    else if (high == 16) {
        ColMask[write_start_row] = col_mask[0];
        ColMask[write_start_row+1] = col_mask[1];
    }
    int m = __popc(*(uint16_t*)(&col_mask[0]));
    for (int i = 1; i <= m; i++) {
        int j = __fns(*(uint16_t*)(&col_mask[0]), 0, i);
        if (width_num == 1)
            RowMask[write_start_col + i - 1] = row_mask[j];
        else if (width_num == 2) {
            RowMask[write_start_col + (i - 1) * 2] = row_mask[j*2];
            RowMask[write_start_col + (i - 1) * 2 + 1] = row_mask[j*2+1];
        }
    }
}   

std::vector<at::Tensor> adaptive_ASC(at::Tensor edge_index, const std::string &model, int node_num_row, int node_num_col) 
{
    // calculate rowwindow_edgenum and col_num for each RowWindow for block_high=16
    int *edge_row_id;
    int edge_num = edge_index.size(1);
    cudaMalloc(&edge_row_id, edge_num*sizeof(int));
    thrust::transform(thrust::device, edge_index.data_ptr<int>()+edge_num, edge_index.data_ptr<int>()+2*edge_num, 
                        edge_row_id, [=]__device__(int row){return row/16;});
    uint64_t *value0;
    cudaMalloc(&value0, edge_num*sizeof(uint64_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+edge_num, edge_index.data_ptr<int>()+edge_num,
                        value0, [=]__device__(int e0, int e1){return (uint64_t)(e1/16)*(uint64_t)node_num_col+(uint64_t)e0;});
    thrust::stable_sort_by_key(thrust::device, value0, value0+edge_num, edge_row_id);
    int *mask, *rowwindow_edgenum_16, *temp, *unique_row_id;
    cudaMalloc(&mask, edge_num*sizeof(int));
    thrust::fill_n(thrust::device, mask, edge_num, 1);
    cudaMalloc(&unique_row_id, ((node_num_row+15)/16)*sizeof(int));
    cudaMalloc(&temp, ((node_num_row+15)/16)*sizeof(int));
    auto unique_row_edgenum = thrust::reduce_by_key(thrust::device, edge_row_id, edge_row_id+edge_num, mask,
                        unique_row_id, temp, thrust::equal_to<int>(), thrust::plus<int>());
    cudaMalloc(&rowwindow_edgenum_16, ((node_num_row+15)/16)*sizeof(int));
    thrust::fill_n(thrust::device, rowwindow_edgenum_16, (node_num_row+15)/16, 0);
    thrust::scatter(thrust::device, temp, unique_row_edgenum.second, unique_row_id, rowwindow_edgenum_16);
    auto unique_value_row = thrust::unique_by_key(thrust::device, value0, value0+edge_num, edge_row_id, thrust::equal_to<uint64_t>());
    int *col_num_16;
    cudaMalloc(&unique_row_id, ((node_num_row+15)/16)*sizeof(int));
    cudaMalloc(&temp, ((node_num_row+15)/16)*sizeof(int));
    auto unique_row_col = thrust::reduce_by_key(thrust::device, edge_row_id, unique_value_row.second, mask,
                        unique_row_id, temp, thrust::equal_to<int>(), thrust::plus<int>());
    cudaMalloc(&col_num_16, ((node_num_row+15)/16)*sizeof(int));
    thrust::fill_n(thrust::device, col_num_16, (node_num_row+15)/16, 0);
    thrust::scatter(thrust::device, temp, unique_row_col.second, unique_row_id, col_num_16);
    // printf("after get rowwindow_edgenum_16 and col_num_16\n");
    // calculate rowwindow_edgenum and col_num for each RowWindow for block_high=8
    cudaMalloc(&edge_row_id, edge_num*sizeof(int));
    thrust::transform(thrust::device, edge_index.data_ptr<int>()+edge_num, edge_index.data_ptr<int>()+2*edge_num, 
                        edge_row_id, [=]__device__(int row){return row/8;});
    cudaMalloc(&value0, edge_num*sizeof(uint64_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+edge_num, edge_index.data_ptr<int>()+edge_num,
                        value0, [=]__device__(int e0, int e1){return (uint64_t)(e1/8)*(uint64_t)node_num_col+(uint64_t)e0;});
    thrust::stable_sort_by_key(thrust::device, value0, value0+edge_num, edge_row_id);
    cudaMalloc(&unique_row_id, ((node_num_row+7)/8)*sizeof(int));
    cudaMalloc(&temp, ((node_num_row+7)/8)*sizeof(int));
    unique_row_edgenum = thrust::reduce_by_key(thrust::device, edge_row_id, edge_row_id+edge_num, mask,
                        unique_row_id, temp, thrust::equal_to<int>(), thrust::plus<int>());
    int *rowwindow_edgenum_8;
    cudaMalloc(&rowwindow_edgenum_8, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, rowwindow_edgenum_8, (node_num_row+7)/8, 0);
    thrust::scatter(thrust::device, temp, unique_row_edgenum.second, unique_row_id, rowwindow_edgenum_8);
    unique_value_row = thrust::unique_by_key(thrust::device, value0, value0+edge_num, edge_row_id, thrust::equal_to<uint64_t>());
    int *col_num_8;
    cudaMalloc(&unique_row_id, ((node_num_row+7)/8)*sizeof(int));
    cudaMalloc(&temp, ((node_num_row+7)/8)*sizeof(int));
    unique_row_col = thrust::reduce_by_key(thrust::device, edge_row_id, unique_value_row.second, mask,
                        unique_row_id, temp, thrust::equal_to<int>(), thrust::plus<int>());
    cudaMalloc(&col_num_8, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, col_num_8, (node_num_row+7)/8, 0);
    thrust::scatter(thrust::device, temp, unique_row_col.second, unique_row_id, col_num_8);
    // printf("after get rowwindow_edgenum_8 and col_num_8\n");
    cudaFree(edge_row_id);
    cudaFree(value0);
    cudaFree(mask);
    cudaFree(temp);
    cudaFree(unique_row_id);
    // select one from 16x16, 16x8, 8x16, 8x8  according to the distribution of ZNN
    int *mixed_edgenum_array, *mixed_colid_array, *mixed_colid_array_padding, *rowwindow_id, *rowwindow_type;
    uint8_t *block_high_array, *block_width_array;
    cudaMalloc(&block_high_array, ((node_num_row+7)/8)*sizeof(uint8_t));
    thrust::fill_n(thrust::device, block_high_array, (node_num_row+7)/8, 0);
    cudaMalloc(&block_width_array, ((node_num_row+7)/8)*sizeof(uint8_t));
    thrust::fill_n(thrust::device, block_width_array, (node_num_row+7)/8, 0);
    cudaMalloc(&mixed_edgenum_array, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, mixed_edgenum_array, (node_num_row+7)/8, 0);
    cudaMalloc(&mixed_colid_array, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, mixed_colid_array, (node_num_row+7)/8, 0);
    cudaMalloc(&mixed_colid_array_padding, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, mixed_colid_array_padding, (node_num_row+7)/8, 0);
    cudaMalloc(&rowwindow_id, ((node_num_row+7)/8)*sizeof(int));
    thrust::fill_n(thrust::device, rowwindow_id, (node_num_row+7)/8, 0);
    cudaMalloc(&rowwindow_type, ((node_num_row+7)/8)*sizeof(int));
    float boundary = (model==std::string("agnn"))?12.13f:6.36f;
    int rows_16 = (node_num_row + 15)/16;
    int rows_8 = (node_num_row + 7)/8;
    select_blocksize<<<(rows_16+511)/512, 512>>>(rowwindow_edgenum_16, col_num_16, rowwindow_edgenum_8, 
                            col_num_8, block_high_array, block_width_array, mixed_edgenum_array, 
                            mixed_colid_array, mixed_colid_array_padding, rowwindow_id, rowwindow_type,
                            boundary, rows_16, rows_8);
    // printf("after selecting blocksize\n");
    cudaFree(rowwindow_edgenum_16);
    cudaFree(col_num_16);
    cudaFree(rowwindow_edgenum_8);
    cudaFree(col_num_8);
    // get RowWindowId_16x16, RowWindowId_16x8, RowWindowId_8x16, RowWindowId_8x8
    thrust::inclusive_scan(thrust::device, rowwindow_id, rowwindow_id+(node_num_row+7)/8, rowwindow_id);
    int *rowwindow_id_;
    cudaMalloc(&rowwindow_id_, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_id, (node_num_row+7)/8, rowwindow_id_);
    auto RowWindowId = thrust::unique_by_key(thrust::device, rowwindow_id, rowwindow_id+(node_num_row+7)/8,
                            rowwindow_type, thrust::equal_to<int>());
    ////////////////////////////////////////////////////////////////
    // printf("rowwindow_id_len: %d\n", RowWindowId.first-rowwindow_id);
    // printf("rowwindow_type_len: %d\n", RowWindowId.second-rowwindow_type);
    // int max_rowwindow_type;
    // auto max_rowwindow_type_ptr = thrust::max_element(thrust::device, rowwindow_type, RowWindowId.second);
    // cudaMemcpy(&max_rowwindow_type, max_rowwindow_type_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_rowwindow_type: %d\n", max_rowwindow_type);
    /////////////////////////////////////////////////////////////////
    thrust::transform(thrust::device, rowwindow_id, RowWindowId.first, rowwindow_id, [=]__device__(int id){return id-1;});
    thrust::stable_sort_by_key(thrust::device, rowwindow_type, RowWindowId.second, rowwindow_id);
    int *rowwindow_mask, *rowwindow_num, *rowwindow_num_, *rowwindow_key;
    cudaMalloc(&rowwindow_mask, (RowWindowId.second-rowwindow_type)*sizeof(int));
    thrust::fill_n(thrust::device, rowwindow_mask, (RowWindowId.second-rowwindow_type), 1);
    cudaMalloc(&rowwindow_num_, 4*sizeof(int));
    cudaMalloc(&rowwindow_num, 4*sizeof(int));
    thrust::fill_n(thrust::device, rowwindow_num, 4, 0);
    cudaMalloc(&rowwindow_key, 4*sizeof(int));
    auto rowwindow_reduce = thrust::reduce_by_key(thrust::device, rowwindow_type, RowWindowId.second, rowwindow_mask, rowwindow_key,
                            rowwindow_num_, thrust::equal_to<int>(), thrust::plus<int>());
    thrust::scatter(thrust::device, rowwindow_num_, rowwindow_reduce.second, rowwindow_key, rowwindow_num);
    int rownum_8x8, rownum_8x16, rownum_16x8, rownum_16x16;
    cudaMemcpy(&rownum_8x8, rowwindow_num, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rownum_8x16, rowwindow_num+1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rownum_16x8, rowwindow_num+2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rownum_16x16, rowwindow_num+3, sizeof(int), cudaMemcpyDeviceToHost);
    printf("rownum_8x8: %d, rownum_8x16: %d, rownum_16x8: %d, rownum_16x16: %d\n", rownum_8x8, rownum_8x16, rownum_16x8, rownum_16x16);
    at::Tensor RowWindowId_8x8 = rownum_8x8 > 0 ? 
                            torch::from_blob(rowwindow_id, {rownum_8x8}, device(torch::kCUDA).dtype(torch::kInt32)).clone():
                            torch::empty({0}, device(torch::kCUDA).dtype(torch::kInt32));
    at::Tensor RowWindowId_8x16 = rownum_8x16 > 0 ?
                            torch::from_blob(rowwindow_id+rownum_8x8, {rownum_8x16}, device(torch::kCUDA).dtype(torch::kInt32)).clone():
                            torch::empty({0}, device(torch::kCUDA).dtype(torch::kInt32));
    at::Tensor RowWindowId_16x8 = rownum_16x8 > 0 ?
                            torch::from_blob(rowwindow_id+rownum_8x8+rownum_8x16, {rownum_16x8}, device(torch::kCUDA).dtype(torch::kInt32)).clone():
                            torch::empty({0}, device(torch::kCUDA).dtype(torch::kInt32));
    at::Tensor RowWindowId_16x16 = rownum_16x16 > 0 ?
                            torch::from_blob(rowwindow_id+rownum_8x8+rownum_8x16+rownum_16x8, {rownum_16x16}, device(torch::kCUDA).dtype(torch::kInt32)).clone():
                            torch::empty({0}, device(torch::kCUDA).dtype(torch::kInt32));
    // printf("after get RowWindowId_16x16, RowWindowId_16x8, RowWindowId_8x16, RowWindowId_8x8\n");
    cudaFree(rowwindow_mask);
    cudaFree(rowwindow_num);
    cudaFree(rowwindow_num_);
    cudaFree(rowwindow_key);
    cudaFree(rowwindow_id);
    cudaFree(rowwindow_type);
    // create RowWindowSparseAToXOffset, RowWindowBlockOffset, RowWindowColMaskOffset, RowWindowRowOffset
    int rownum = rownum_8x8+rownum_8x16+rownum_16x8+rownum_16x16;
    int *rowwindow_blocknum, *colmask_len, *rowwindow_rownum;
    cudaMalloc(&rowwindow_blocknum, (node_num_row+7)/8*sizeof(int));
    thrust::transform(thrust::device, mixed_colid_array_padding, mixed_colid_array_padding+(node_num_row+7)/8, block_width_array,
                            rowwindow_blocknum, [=]__device__(int col_len, uint8_t block_width){return col_len/max(block_width, 1);});
    cudaMalloc(&colmask_len, (node_num_row+7)/8*sizeof(int));
    thrust::transform(thrust::device, rowwindow_blocknum, rowwindow_blocknum+(node_num_row+7)/8, block_high_array,
                            colmask_len, [=]__device__(int blocknum, uint8_t high){return blocknum*(int)high/8;});
    cudaMalloc(&rowwindow_id, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_id_, (node_num_row+7)/8, rowwindow_id);
    auto colmask_offset_end = thrust::unique_by_key(thrust::device, rowwindow_id_, rowwindow_id_+(node_num_row+7)/8, colmask_len);
    cudaFree(rowwindow_id_);
    at::Tensor RowWindowColMaskOffset = torch::empty({rownum+1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(RowWindowColMaskOffset.data_ptr<int>(), 0, sizeof(int));
    thrust::copy(thrust::device, colmask_len, colmask_offset_end.second, RowWindowColMaskOffset.data_ptr<int>()+1);
    thrust::inclusive_scan(thrust::device, RowWindowColMaskOffset.data_ptr<int>()+1, RowWindowColMaskOffset.data_ptr<int>()+1+rownum, 
                            RowWindowColMaskOffset.data_ptr<int>()+1);
    thrust::inclusive_scan(thrust::device, rowwindow_blocknum, rowwindow_blocknum+(node_num_row+7)/8, rowwindow_blocknum);
    ////////////////////////////////////////////////////////////////////
    // int max_rowwindow_blocknum;
    // auto max_rowwindow_blocknum_ptr = thrust::max_element(thrust::device, rowwindow_blocknum, rowwindow_blocknum+(node_num_row+7)/8);
    // cudaMemcpy(&max_rowwindow_blocknum, max_rowwindow_blocknum_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_rowwindow_blocknum: %d\n", max_rowwindow_blocknum);
    //////////////////////////////////////////////////////////////////////
    cudaMalloc(&rowwindow_id_, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_id, (node_num_row+7)/8, rowwindow_id_);
    auto rowwindow_blocknum_end = thrust::unique_by_key(thrust::device, rowwindow_id, rowwindow_id+(node_num_row+7)/8, rowwindow_blocknum);
    cudaFree(rowwindow_id);
    at::Tensor RowWindowBlockOffset = torch::empty({rownum+1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(RowWindowBlockOffset.data_ptr<int>(), 0, sizeof(int));
    ////////////////////////////////////////////////////////////////////////
    // max_rowwindow_blocknum_ptr = thrust::max_element(thrust::device, rowwindow_blocknum, rowwindow_blocknum_end.second);
    // cudaMemcpy(&max_rowwindow_blocknum, max_rowwindow_blocknum_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_rowwindow_blocknum: %d\n", max_rowwindow_blocknum);
    //////////////////////////////////////////////////////////////////////////
    thrust::copy(thrust::device, rowwindow_blocknum, rowwindow_blocknum_end.second, RowWindowBlockOffset.data_ptr<int>()+1);
    cudaMalloc(&rowwindow_id, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_id_, (node_num_row+7)/8, rowwindow_id);
    auto mixed_colid_end = thrust::unique_by_key(thrust::device, rowwindow_id_, rowwindow_id_+(node_num_row+7)/8,
                            mixed_colid_array_padding, thrust::equal_to<int>());
    cudaFree(rowwindow_id_);
    at::Tensor RowWindowSparseAToXOffset = torch::empty({rownum+1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(RowWindowSparseAToXOffset.data_ptr<int>(), 0, sizeof(int));
    thrust::copy(thrust::device, mixed_colid_array_padding, mixed_colid_end.second, RowWindowSparseAToXOffset.data_ptr<int>()+1);
    thrust::inclusive_scan(thrust::device, RowWindowSparseAToXOffset.data_ptr<int>()+1, RowWindowSparseAToXOffset.data_ptr<int>()+rownum+1,
                            RowWindowSparseAToXOffset.data_ptr<int>()+1);
    cudaMalloc(&rowwindow_rownum, (node_num_row+7)/8*sizeof(int));
    thrust::copy(thrust::device, block_high_array, block_high_array+(node_num_row+7)/8, rowwindow_rownum);
    thrust::inclusive_scan(thrust::device, rowwindow_rownum, rowwindow_rownum + (node_num_row+7)/8, rowwindow_rownum);
    cudaMalloc(&rowwindow_id_, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_id, (node_num_row+7)/8, rowwindow_id_);
    auto rowwindow_rownum_end = thrust::unique_by_key(thrust::device, rowwindow_id, rowwindow_id+(node_num_row+7)/8, rowwindow_rownum);
    ////////////////////////////////////////////////////////////
    // printf("block_high_array_len: %d, rowwindow_rownum: %d, rownum: %d\n", (node_num_row+7)/8, rowwindow_rownum_end.second-rowwindow_rownum, rownum);
    // int* max_rowwindow_rownum = thrust::max_element(thrust::device, rowwindow_rownum, rowwindow_rownum_end.second);
    // int max_rowwindow_rownum_value;
    // cudaMemcpy(&max_rowwindow_rownum_value, max_rowwindow_rownum, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_rowwindow_rownum: %d\n", max_rowwindow_rownum_value);
    ////////////////////////////////////////////////////////////
    cudaFree(rowwindow_id);
    at::Tensor RowWindowRowOffset = torch::empty({rownum+1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(RowWindowRowOffset.data_ptr<int>(), 0, sizeof(int));
    thrust::copy_n(thrust::device, rowwindow_rownum, rownum, RowWindowRowOffset.data_ptr<int>()+1);
    // printf("after get RowWindowSparseAToXOffset, RowWindowBlockOffset, RowWindowColMaskOffset\n");
    int SparseAToX_len;
    cudaMemcpy(&SparseAToX_len, RowWindowSparseAToXOffset.data_ptr<int>()+rownum, sizeof(int), cudaMemcpyDeviceToHost);
    // get sorted edge_index
    ////////////////////////////////////////////////////////////////////
    // int mixed_edgenum_array_sum = thrust::reduce(thrust::device, mixed_edgenum_array, mixed_edgenum_array+(node_num_row+7)/8, 0);
    // printf("mixed_edgenum_array_sum: %d\n", mixed_edgenum_array_sum);
    // int block_high_array_sum = thrust::reduce(thrust::device, block_high_array, block_high_array+(node_num_row+7)/8, 0);
    // printf("block_high_array_sum: %d\n", block_high_array_sum);
    ////////////////////////////////////////////////////////////////////
    thrust::inclusive_scan(thrust::device, mixed_edgenum_array, mixed_edgenum_array+(node_num_row+7)/8, mixed_edgenum_array);
    auto unique_offset = thrust::unique_by_key(thrust::device, mixed_edgenum_array, mixed_edgenum_array+(node_num_row+7)/8, 
                                block_high_array, thrust::equal_to<int>());
    int *edge_index_1;
    cudaMalloc(&edge_index_1, edge_num*sizeof(int));
    thrust::fill_n(thrust::device, edge_index_1, edge_num, 0);
    ////////////////////////////////////////////////////
    // int edge_index_1_max, block_high, mixed_edgenum_max;
    // auto edgenum_offset_max = thrust::max_element(thrust::device, mixed_edgenum_array, unique_offset.first-1);
    // cudaMemcpy(&mixed_edgenum_max, edgenum_offset_max, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("edge_num: %d, mixed_edgenum_array_max: %d\n", edge_num, mixed_edgenum_max);
    // auto edge_index_max = thrust::max_element(thrust::device, edge_index_1, edge_index_1+edge_num);
    // cudaMemcpy(&edge_index_1_max, edge_index_max, sizeof(int), cudaMemcpyDeviceToHost);
    // auto block_high_ptr = thrust::min_element(thrust::device, block_high_array, blockhigh_end.second);
    // cudaMemcpy(&block_high, block_high_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("edge_index_1_max: %d, block_high_array[1]: %d, mixed_edgenum_max: %d\n", edge_index_1_max, block_high, mixed_edgenum_max);
    // printf("blockhigh_array_len: %d\n", blockhigh_end.second-block_high_array);
    /////////////////////////////////////////////////////////
    thrust::scatter(thrust::device, block_high_array, unique_offset.second-1, mixed_edgenum_array, edge_index_1);
    thrust::inclusive_scan(thrust::device, edge_index_1, edge_index_1+edge_num, edge_index_1);
    ////////////////////////////////////////////////////
    // int edge_index_1_end;
    // cudaMemcpy(&edge_index_1_end, edge_index_1+edge_num-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("edge_index_1[-1]: %d\n", edge_index_1_end);
    /////////////////////////////////////////////////////////
    uint64_t *value1, *value2;
    cudaMalloc(&value1, edge_num*sizeof(uint64_t));
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+edge_num, edge_index.data_ptr<int>()+edge_num, 
                            value1, [=]__device__(int e0, int e1){return (uint64_t)e1*(uint64_t)node_num_col+(uint64_t)e0;});
    cudaMalloc(&value2, edge_num*sizeof(uint64_t));
    thrust::copy_n(thrust::device, value1, edge_num, value2);
    thrust::stable_sort_by_key(thrust::device, value1, value1+edge_num, edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value2, value2+edge_num, edge_index.data_ptr<int>()+edge_num);
    thrust::transform(thrust::device, edge_index.data_ptr<int>(), edge_index.data_ptr<int>()+edge_num, edge_index_1,
                            value1, [=]__device__(int e0, int e1){return (uint64_t)e1*(uint64_t)node_num_col+(uint64_t)e0;});
    thrust::copy_n(thrust::device, value1, edge_num, value2);
    thrust::stable_sort_by_key(thrust::device, value1, value1+edge_num, edge_index.data_ptr<int>());
    thrust::stable_sort_by_key(thrust::device, value2, value2+edge_num, edge_index.data_ptr<int>()+edge_num);
    int *condensed_edgeindex_0;
    cudaMalloc(&condensed_edgeindex_0, edge_num*sizeof(int));
    thrust::adjacent_difference(thrust::device, value1, value1+edge_num, condensed_edgeindex_0, 
                            [=]__device__(uint64_t v0, uint64_t v1){return (v0==v1)?0:1;});
    thrust::fill_n(thrust::device, condensed_edgeindex_0, 1, 1);
    // printf("after sorting edge_index\n");
    // create SparseAToX
    int *Col_idx, *map, *mixed_colid_array_copy;
    cudaMalloc(&Col_idx, edge_num*sizeof(int));
    thrust::copy_n(thrust::device, edge_index.data_ptr<int>(), edge_num, Col_idx);
    auto unique_col_id = thrust::unique_by_key(thrust::device, value1, value1+edge_num, Col_idx, thrust::equal_to<uint64_t>());
    int SparseAToX_nopadding_len = unique_col_id.second-Col_idx;
    cudaMalloc(&map, SparseAToX_nopadding_len*sizeof(int));
    thrust::fill_n(thrust::device, map, SparseAToX_nopadding_len, 0);
    thrust::inclusive_scan(thrust::device, mixed_colid_array, mixed_colid_array+(node_num_row+7)/8, mixed_colid_array);
    cudaMalloc(&mixed_colid_array_copy, (node_num_row+7)/8*sizeof(int));
    thrust::copy_n(thrust::device, mixed_colid_array, (node_num_row+7)/8, mixed_colid_array_copy);
    auto blockwidth_end = thrust::unique_by_key(thrust::device, mixed_colid_array_copy, mixed_colid_array_copy+(node_num_row+7)/8, block_width_array);
    auto mixed_colid_array_end = thrust::unique_by_key(thrust::device, rowwindow_id_, rowwindow_id_+(node_num_row+7)/8, mixed_colid_array);
    auto mixed_colid_end_ = thrust::unique_by_key(thrust::device, mixed_colid_array, mixed_colid_array_end.second, mixed_colid_array_padding);
    //////////////////////////////////////////////////////////
    // int mixed_colid_array_padding_len, max_mixed_colid_array;
    // mixed_colid_array_padding_len = mixed_colid_end_.second - mixed_colid_array_padding;
    // auto max_mixed_colid_array_ptr = thrust::max_element(thrust::device, mixed_colid_array, mixed_colid_end_.first-1);
    // cudaMemcpy(&max_mixed_colid_array, max_mixed_colid_array_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("mixed_colid_array_padding_len: %d, SparseAToX_nopadding_len: %d, max_mixed_colid_array: %d\n", mixed_colid_array_padding_len, SparseAToX_nopadding_len, max_mixed_colid_array);
    //////////////////////////////////////////////////////////
    thrust::scatter(thrust::device, mixed_colid_array_padding, mixed_colid_end_.second-1, mixed_colid_array, map);
    thrust::inclusive_scan(thrust::device, map, map+SparseAToX_nopadding_len, map);
    auto aligned = thrust::unique_by_key(thrust::device, value2, value2+edge_num, edge_index_1, thrust::equal_to<int>());
    cudaMalloc(&mask, (aligned.second-edge_index_1)*sizeof(int));
    // printf("mask_len: %d, map_len: %d\n", (aligned.second-edge_index_1), SparseAToX_nopadding_len);
    thrust::fill_n(thrust::device, mask, (aligned.second-edge_index_1), 1);
    thrust::exclusive_scan_by_key(thrust::device, edge_index_1, aligned.second, mask, mask);
    thrust::transform(thrust::device, map, map+SparseAToX_nopadding_len, mask, map, [=]__device__(int id0, int id1){return id0+id1;});
    ////////////////////////////////////////////////////////////////////////////
    // int map_end;
    // cudaMemcpy(&map_end, map+SparseAToX_nopadding_len-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("map[-1]: %d, edge_num: %d\n", map_end, edge_num);
    /////////////////////////////////////////////////////////////////////////////////////
    at::Tensor SparseAToX = torch::empty({SparseAToX_len}, device(torch::kCUDA).dtype(torch::kInt32));
    thrust::fill_n(thrust::device, SparseAToX.data_ptr<int>(), SparseAToX_len, node_num_col);
    ///////////////////////////////////////////////////////////////////////////////////
    // int max_map;
    // auto max_map_ptr = thrust::max_element(thrust::device, map, map+SparseAToX_nopadding_len);
    // cudaMemcpy(&max_map, max_map_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("SparseAToX_len: %d, max_map: %d\n", SparseAToX_len, max_map);
    ////////////////////////////////////////////////////////////////////////////////////
    thrust::scatter(thrust::device, Col_idx, unique_col_id.second, map, SparseAToX.data_ptr<int>());
    // printf("after getting SparseAToX\n");
    cudaFree(edge_index_1);
    cudaFree(mixed_colid_array_padding);
    cudaFree(mixed_colid_array);
    cudaFree(mixed_colid_array_copy);
    cudaFree(Col_idx);
    cudaFree(map);
    cudaFree(rowwindow_id_);
    cudaFree(mask);
    // create BlockRowMaskOffset
    int *segment_key;
    uint8_t *block_high_long_array, *block_width_long_array;
    cudaMalloc(&segment_key, edge_num*sizeof(int));
    thrust::fill_n(thrust::device, segment_key, edge_num, 0);
    thrust::scatter(thrust::device, block_high_array, unique_offset.second-1, mixed_edgenum_array, segment_key);
    thrust::inclusive_scan(thrust::device, segment_key, segment_key+edge_num, segment_key);
    cudaMalloc(&block_high_long_array, edge_num*sizeof(uint8_t));
    thrust::fill_n(thrust::device, block_high_long_array, edge_num, 0);
    cudaMemcpy(block_high_long_array, block_high_array, sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaMalloc(&block_width_long_array, edge_num*sizeof(uint8_t));
    thrust::fill_n(thrust::device, block_width_long_array, edge_num, 0);
    cudaMemcpy(block_width_long_array, block_width_array, sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    thrust::scatter(thrust::device, block_high_array+1, unique_offset.second, mixed_edgenum_array, block_high_long_array);
    thrust::scatter(thrust::device, block_width_array+1, blockwidth_end.second, mixed_edgenum_array, block_width_long_array);
    thrust::inclusive_scan_by_key(thrust::device, segment_key, segment_key+edge_num, block_high_long_array, block_high_long_array);
    thrust::inclusive_scan_by_key(thrust::device, segment_key, segment_key+edge_num, block_width_long_array, block_width_long_array);
    cudaFree(block_width_array);
    cudaFree(block_high_array);
    ////////////////////////////////////////////////////////////
    // int max_block_high_long_array, max_block_width_long_array;
    // auto max_block_high_long_array_ptr = thrust::min_element(thrust::device, block_high_long_array, block_high_long_array+edge_num);
    // auto max_block_width_long_array_ptr = thrust::min_element(thrust::device, block_width_long_array, block_width_long_array+edge_num);
    // cudaMemcpy(&max_block_high_long_array, max_block_high_long_array_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&max_block_width_long_array, max_block_width_long_array_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_block_high_long_array: %d, max_block_width_long_array: %d\n", max_block_high_long_array, max_block_width_long_array);
    ////////////////////////////////////////////////////////////
    int blocks = (edge_num+511)/512;
    cudaMalloc(&value1, edge_num*sizeof(uint64_t));
    thrust::inclusive_scan_by_key(thrust::device, segment_key, segment_key+edge_num, condensed_edgeindex_0, condensed_edgeindex_0);
    transform_5<<<blocks, 512>>>(condensed_edgeindex_0, edge_index.data_ptr<int>()+edge_num, segment_key,
                                block_high_long_array, block_width_long_array, value1, node_num_col, edge_num);
    cudaFree(segment_key);
    thrust::stable_sort(thrust::device, value1, value1+edge_num);
    cudaMalloc(&value2, edge_num*sizeof(uint64_t));
    thrust::transform(thrust::device, value1, value1+edge_num, block_high_long_array, value2, 
                                [=]__device__(uint64_t v1, uint8_t high){return (v1&((uint64_t)(0xffffffffffffffff)-(uint64_t)(high-1)))>>3;});
    // printf("after transform.\n");
    uint32_t *value3;
    cudaMalloc(&value3, edge_num*sizeof(uint32_t));
    thrust::transform(thrust::device, value2, value2+edge_num, value3, [=]__device__(uint64_t v){return (uint32_t)((0xffffffff)&v);});
    auto total_row_key = thrust::unique_by_key(thrust::device, value1, value1+edge_num, block_high_long_array);
    auto total_block_key_ = thrust::unique_by_key(thrust::device, value2, value2+edge_num, block_width_long_array);
    /////////////////////////////////////////////////////////
    // printf("transform_v1_len: %d, ", total_row_key.first-value1);
    // printf("transform_v2_len: %d\n", total_row_key.second-block_high_long_array);
    // printf("transform_o1_len: %d, ", total_row_key.first-value1);
    // int max_block_width_long_array;
    // auto max_block_width_long_array_ptr = thrust::min_element(thrust::device, block_width_long_array, total_block_key_.second);
    // cudaMemcpy(&max_block_width_long_array, max_block_width_long_array_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_block_width_long_array: %d\n", max_block_width_long_array);
    //////////////////////////////////////////////////////////
    thrust::transform(thrust::device, value1, total_row_key.first, block_high_long_array, value1,
                                [=]__device__(uint64_t value1, uint8_t block_high){return value1&((uint64_t)(0xffffffffffffffff)-(uint64_t)(block_high-1));});
    int total_row_num = total_row_key.first-value1;
    int block_num = total_block_key_.first-value2;
    uint8_t *mask_;
    cudaMalloc(&mask_, total_row_num*sizeof(uint8_t));
    uint8_t *block_row_num;
    thrust::fill_n(thrust::device, mask_, total_row_num, 1);
    // printf("after init mask_.\n");
    cudaMalloc(&block_row_num, block_num*sizeof(uint8_t));
    thrust::fill_n(thrust::device, block_row_num, block_num, 0);
    auto block_rownum_end = thrust::reduce_by_key(thrust::device, value1, total_row_key.first, mask_, value2,
                                block_row_num, thrust::equal_to<uint64_t>(), thrust::plus<uint8_t>());
    // printf("after get block_rownum.\n");
    auto block_high_end = thrust::unique_by_key(thrust::device, value1, total_row_key.first, block_high_long_array);
    cudaFree(value1);
    cudaFree(mask_);
    /////////////////////////////////////////////////////////////////////////////////////////
    // int min_block_row_num;
    // int block_row_num_len = block_rownum_end.second - block_row_num;
    // auto min_block_row_num_ptr = thrust::min_element(thrust::device, block_row_num, block_row_num+block_num);
    // int block_row_num_sum = thrust::reduce(thrust::device, block_row_num, block_rownum_end.second);
    // cudaMemcpy(&min_block_row_num, min_block_row_num_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("total_row_num: %d, block_row_num_sum: %d\n", total_row_num, block_row_num_sum);
    // printf("min_block_row_num: %d, block_row_num_len: %d, block_num: %d\n", min_block_row_num, block_row_num_len, block_num);
    // printf("block_row_num_len: %d, block_num: %d\n", block_rownum_end.second-block_row_num, block_num);
    /////////////////////////////////////////////////////////////////////////////////
    at::Tensor BlockRowMaskOffset = torch::empty({block_num+1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(BlockRowMaskOffset.data_ptr<int>(), 0, sizeof(int));
    /////////////////////////////////////////////////////////
    // printf("transform_v1_len: %d, ", block_num);
    // printf("transform_v2_len: %d\n", total_block_key_.second-block_width_long_array);
    // printf("transform_o1_len: %d, ", block_num);
    //////////////////////////////////////////////////////////
    thrust::transform(thrust::device, block_row_num, block_row_num+block_num, block_width_long_array, BlockRowMaskOffset.data_ptr<int>()+1,
                                [=]__device__(uint8_t row_num, uint8_t width){return (int)row_num*(int)(width/8);});
    // printf("after transform.\n");
    thrust::inclusive_scan(thrust::device, BlockRowMaskOffset.data_ptr<int>()+1, BlockRowMaskOffset.data_ptr<int>()+block_num+1, 
                                BlockRowMaskOffset.data_ptr<int>()+1);
    // printf("after getting BlockRowMaskOffset\n");
    
    cudaFree(mixed_edgenum_array);
    cudaFree(block_row_num);
    // create ColMask, RowMask
    cudaMalloc(&mask, edge_num*sizeof(int));
    thrust::fill_n(thrust::device, mask, edge_num, 1);
    int *TCOffset, *TCColMaskOffset;
    cudaMalloc(&TCOffset, (block_num+1)*sizeof(int));
    cudaMemset(TCOffset, 0, sizeof(int));
    thrust::reduce_by_key(thrust::device, value3, value3+edge_num, mask, value2, TCOffset+1,
                                thrust::equal_to<uint32_t>(), thrust::plus<int>());
    thrust::inclusive_scan(thrust::device, TCOffset+1, TCOffset+1+block_num, TCOffset+1);
    cudaFree(value2);
    cudaFree(value3);
    cudaMalloc(&TCColMaskOffset, block_num*sizeof(int));
    thrust::fill_n(thrust::device, TCColMaskOffset, block_num, 0);
    auto TCColOffset_end = thrust::unique_by_key(thrust::device, rowwindow_blocknum, rowwindow_blocknum_end.second, colmask_len);
    thrust::scatter(thrust::device, colmask_len, TCColOffset_end.second-1, rowwindow_blocknum, TCColMaskOffset);
    thrust::inclusive_scan(thrust::device, TCColMaskOffset, TCColMaskOffset+block_num, TCColMaskOffset);
    //////////////////////////////////////////////////////////////////////////////////////////
    // int TCColMaskOffset_end;
    // cudaMemcpy(&TCColMaskOffset_end, TCColMaskOffset+block_num-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("TCColMaskOffset_end: %d\n", TCColMaskOffset_end);
    //////////////////////////////////////////////////////////////////////////////////////////
    cudaMalloc(&mask, block_num*sizeof(int));
    thrust::fill_n(thrust::device, mask, block_num, 1);
    thrust::exclusive_scan_by_key(thrust::device, TCColMaskOffset, TCColMaskOffset+block_num, mask, mask);
    /////////////////////////////////////////////////////////
    // printf("transform_v1_len: %d, ", block_num);
    // printf("transform_v2_len: %d\n", block_high_end.second-block_high_long_array);
    // printf("transform_o1_len: %d, ", block_num);
    //////////////////////////////////////////////////////////
    thrust::transform(thrust::device, mask, mask+block_num, block_high_long_array, mask,
                                [=]__device__(int id, uint8_t high){return id*(int)(high/8);});
    /////////////////////////////////////////////////////////
    // printf("transform_v1_len: %d, ", block_num);
    // printf("transform_v2_len: %d\n", block_num);
    // printf("transform_o1_len: %d, ", block_num);
    //////////////////////////////////////////////////////////
    thrust::transform(thrust::device, mask, mask+block_num, TCColMaskOffset, TCColMaskOffset,
                                [=]__device__(int id, int off){return id+off;});
    int ColMask_len, RowMask_len;
    cudaMemcpy(&ColMask_len, RowWindowColMaskOffset.data_ptr<int>()+rownum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&RowMask_len, BlockRowMaskOffset.data_ptr<int>()+block_num, sizeof(int), cudaMemcpyDeviceToHost);
    at::Tensor ColMask = torch::empty({ColMask_len}, device(torch::kCUDA).dtype(torch::kUInt8));
    at::Tensor RowMask = torch::empty({RowMask_len}, device(torch::kCUDA).dtype(torch::kUInt8));
    ////////////////////////////////////////////////////////////////////
    // int BlockRowMaskOffset_end;
    // cudaMemcpy(&TCColMaskOffset_end, TCColMaskOffset+block_num-1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("TCColMaskOffset_end: %d, colMask_len: %d\n", TCColMaskOffset_end, ColMask_len);
    ////////////////////////////////////////////////////////////////////
    blocks = (block_num+255)/256;
    mixed_BlockMaskShortKernel<<<blocks, 256>>>(condensed_edgeindex_0,
                                        edge_index.data_ptr<int>()+edge_num,
                                        TCOffset, TCColMaskOffset,
                                        BlockRowMaskOffset.data_ptr<int>(),
                                        block_high_long_array,
                                        block_width_long_array,
                                        ColMask.data_ptr<uint8_t>(),
                                        RowMask.data_ptr<uint8_t>(), 
                                        block_num);
    // printf("after getting ColMask, RowMask\n");
    cudaFree(rowwindow_blocknum);
    cudaFree(colmask_len);
    cudaFree(block_high_long_array);
    cudaFree(block_width_long_array);
    cudaFree(mask);
    cudaFree(TCOffset);
    cudaFree(TCColMaskOffset);
    cudaFree(condensed_edgeindex_0);
    return {RowWindowId_8x8, RowWindowId_8x16, RowWindowId_16x8, RowWindowId_16x16, RowWindowRowOffset, RowWindowBlockOffset, 
        RowWindowSparseAToXOffset, RowWindowColMaskOffset, BlockRowMaskOffset, ColMask, RowMask, SparseAToX};
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
                int col_ = (local_row * block_width + col) & 31;
                int row_ = (local_row * block_width + col) >> 5;
                atomicOr((uint32_t*)&row_mask[row_*4], (1<<col_));
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

__global__ void get_id_count(
    const int* __restrict__ edge_1,
    int* __restrict__ id_count,
    int* __restrict__ edge_local_id,
    int edge_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_num) return;
    int local = atomicAdd(&id_count[edge_1[idx]], 1);
    edge_local_id[idx] = local;
} 

__global__ void get_row_num(
    const int* __restrict__ id_count,
    int* __restrict__ row_num,
    int block_size,
    int node_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_num = 0;
    if (idx < node_num)
        max_num = max(max_num, id_count[idx]);
    for (int i = 1; i < block_size; i *= 2) {
        max_num = max(max_num, __shfl_xor_sync(FULL_MASK, max_num, 2*i-1));
    }
    if (idx < node_num)
        row_num[idx/block_size] = max_num;
}

__global__ void gen_map(
    const int* __restrict__ edge_1,
    const int* __restrict__ edge_local_id,
    const int* __restrict__ row_offset,
    int* __restrict__ edge_id,
    int block_size,
    int edge_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_num) return;
    int map_id = (row_offset[edge_1[idx]/block_size] + edge_local_id[idx]) * block_size + (edge_1[idx]&(block_size-1));
    edge_id[map_id] = idx;
}

std::vector<at::Tensor> ASC_e2v(at::Tensor edge_index, int block_size, int node_num) {
    Timer time;
    int edge_num = edge_index.size(1);
    int *id_count, *edge_local_id;
    cudaMalloc(&edge_local_id, edge_num * sizeof(int));
    cudaMalloc(&id_count, node_num * sizeof(int));
    cudaMemset(id_count, 0, node_num * sizeof(int));
    int blocks = (edge_num + 1023) / 1024;
    get_id_count<<<blocks, 1024>>>(edge_index.data_ptr<int>() + edge_num, id_count, edge_local_id, edge_num);
    at::Tensor row_offset = torch::empty({(node_num + block_size - 1) / block_size + 1}, device(torch::kCUDA).dtype(torch::kInt32));
    cudaMemset(row_offset.data_ptr<int>(), 0, sizeof(int));
    blocks = (node_num + 255) / 256;
    get_row_num<<<blocks, 256>>>(id_count, row_offset.data_ptr<int>() + 1, block_size, node_num);
    thrust::inclusive_scan(thrust::device, row_offset.data_ptr<int>() + 1, row_offset.data_ptr<int>() + (node_num + block_size - 1) / block_size + 1, 
                        row_offset.data_ptr<int>() + 1);
    int block_num;
    cudaMemcpy(&block_num, row_offset.data_ptr<int>() + (node_num + block_size - 1) / block_size, sizeof(int), cudaMemcpyDeviceToHost);
    at::Tensor edge_id = torch::empty({block_num * block_size}, device(torch::kCUDA).dtype(torch::kInt32));
    thrust::fill_n(thrust::device, edge_id.data_ptr<int>(), block_num * block_size, edge_num);
    blocks = (edge_num + 255) / 256;
    gen_map<<<blocks, 256>>>(edge_index.data_ptr<int>() + edge_num, edge_local_id, row_offset.data_ptr<int>(),
                            edge_id.data_ptr<int>(), block_size, edge_num);
    cudaFree(edge_local_id);
    cudaFree(id_count);
    return {edge_id, row_offset};
}   