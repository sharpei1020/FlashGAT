import torch
from ctypes import cdll
cdll.LoadLibrary('/home/ljq/mine/graphiler/src/build/sputnik/libsputnik.so')
import mygraph

g = torch.randint(0, 40, (500, 500))
g = torch.where(g>=1, torch.zeros(g.shape), torch.ones(g.shape)).type(torch.IntTensor)
res = torch.nonzero(g, as_tuple=True)
adj = torch.vstack((res[1], res[0])).to(torch.int32)

def torchcore_process_DTC(g, block_high, block_width, node_num):
    iter = (node_num + block_high - 1) // block_high
    RowWindowOffset = [0]
    TCOffset = [0]
    BlockMask = []
    SparseAToX = []
    for i in range(iter):
        rowmask = torch.zeros((node_num), dtype=torch.bool)
        for j in range(block_high):
            if i * block_high + j < node_num:
                rowmask = rowmask | g[i * block_high + j, :].to(torch.bool)
        rowAtoX_colid = torch.nonzero(rowmask).view(-1)
        SparseAToX.append(rowAtoX_colid)
        num = len(rowAtoX_colid)
        TCOffset.append(TCOffset[-1] + num)
        RowWindowOffset.append(RowWindowOffset[-1] + (num + block_width - 1) // block_width)
        rowblock_len = (num + block_width - 1) // block_width
        for k in range(rowblock_len):
            BlockMask_elem = torch.zeros((block_high), dtype=torch.uint8)
            for l in range(block_width):
                if k * block_width + l < num:
                    tmp = torch.zeros((block_high), dtype=torch.uint8)
                    tmp[0: min((i + 1) * block_high, node_num) - i * block_high] = g[i * block_high : (i + 1) * block_high, rowAtoX_colid[k * block_width + l]]
                    BlockMask_elem += tmp * (2 ** l)
            BlockMask.append(BlockMask_elem)
    RowWindowOffset = torch.Tensor(RowWindowOffset).to("cuda")
    TCOffset = torch.Tensor(TCOffset).to("cuda")
    BlockMask = torch.stack(BlockMask).to("cuda")
    SparseAToX = torch.cat(SparseAToX).to("cuda")
    return RowWindowOffset, TCOffset, BlockMask, SparseAToX

def torchcore_process_DTC_short_mask(g, block_high, block_width, node_num):
    iter = (node_num + block_high - 1) // block_high
    RowWindowOffset = [0]
    BitMask_RowOffset = [0]
    BitMask_col = [] 
    BitMask_row = []
    SparseAToX = []
    for i in range(iter):
        rowmask = torch.zeros((node_num), dtype=torch.bool)
        for j in range(block_high):
            if i * block_high + j < node_num:
                rowmask = rowmask | g[i * block_high + j, :].to(torch.bool)
        rowAtoX_colid = torch.nonzero(rowmask).view(-1)
        rowblock_len = (len(rowAtoX_colid) + block_width - 1) // block_width
        RowWindowOffset.append(RowWindowOffset[-1] + rowblock_len)
        SparseAToX_part = torch.ones((rowblock_len * block_width), dtype=torch.int32) * node_num
        SparseAToX_part[:len(rowAtoX_colid)] = rowAtoX_colid
        SparseAToX.append(SparseAToX_part)
        for k in range(rowblock_len):
            BitMask_col_elem = torch.zeros((block_high//8), dtype=torch.uint8)
            block_colmask = torch.zeros((block_high), dtype=torch.bool)
            for l in range(block_width):
                if k * block_width + l < len(rowAtoX_colid):
                    block_colmask[:min(node_num - i * block_high, block_high)] |= g[i * block_high:(i + 1) * block_high, rowAtoX_colid[k * block_width + l]].to(torch.bool)
            rowid = torch.nonzero(block_colmask).view(-1)
            BitMask_RowOffset.append(BitMask_RowOffset[-1] + len(rowid))
            block_rowmask = torch.zeros((len(rowid)*(block_width//8)), dtype=torch.uint8).reshape((-1, (block_width//8)))
            for l in range(len(rowid)):
                # cuda数据字节是小端存储
                BitMask_col_elem[rowid[l]//8] += (2 ** (rowid[l]%8))
            for l in range(min(block_width, len(rowAtoX_colid) - k * block_width)):
                block_rowmask[:, l//8] += g[i * block_high + rowid, rowAtoX_colid[k * block_width + l]] * (2 ** (l % 8))
            BitMask_col.append(BitMask_col_elem)
            BitMask_row.append(block_rowmask.reshape((-1)))
    RowWindowOffset = torch.Tensor(RowWindowOffset).to("cuda")
    BitMask_RowOffset = torch.Tensor(BitMask_RowOffset).to("cuda")
    BitMask_col = torch.cat(BitMask_col).to("cuda")
    BitMask_row = torch.cat(BitMask_row).to("cuda")
    SparseAToX = torch.cat(SparseAToX).to("cuda")
    return RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, SparseAToX

RowWindowOffset_test, BitMask_RowOffset_test, BitMask_col_test, BitMask_row_test, SparseAToX_test = torchcore_process_DTC_short_mask(g, 8, 16, 500)
RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, SparseAToX = mygraph.process_DTC_short_mask(adj.to("cuda"), 8, 16, 500, False)
# print(RowWindowOffset_test, RowWindowOffset)
condition_out = torch.all(torch.abs(RowWindowOffset_test - RowWindowOffset) < 1e-5, dim=-1).cpu()
print(condition_out)
# print(SparseAToX_test, SparseAToX)
condition_out = torch.all(torch.abs(SparseAToX_test - SparseAToX) < 1e-5, dim=-1).cpu()
print(condition_out)
# print(TCOffset_test, TCOffset)
# print(BitMask_RowOffset_test, BitMask_RowOffset)
condition_out = torch.all(torch.abs(BitMask_RowOffset_test - BitMask_RowOffset) < 1e-5, dim=-1).cpu()
print(condition_out)
# print(BitMask_col_test, BitMask_col)
condition_out = torch.all(torch.abs(BitMask_col_test - BitMask_col) < 1e-5, dim=-1).cpu()
print(condition_out)
print(BitMask_row_test, BitMask_row)
condition_out = torch.all(torch.abs(BitMask_row_test - BitMask_row) < 1e-5, dim=-1).cpu()
print(condition_out)
# print(torch.nonzero(~condition_out))


    
# RowWindowOffset_test, TCOffset_test, BlockMask_test, SparseAToX_test = torchcore_process_DTC(g, 16, 8, 100)
# print(adj.shape, adj, res)
# RowWindowOffset, TCOffset, BlockMask, SparseAToX = mygraph.process_DTC(adj.to("cuda"), 16, 8, 100)

# print(RowWindowOffset_test, RowWindowOffset)
# condition_out = torch.all(torch.abs(RowWindowOffset_test - RowWindowOffset) < 1e-5, dim=-1).cpu()
# print(condition_out)
# print(TCOffset_test, TCOffset)
# condition_out = torch.all(torch.abs(TCOffset_test - TCOffset) < 1e-5, dim=-1).cpu()
# print(condition_out)
# condition_out = torch.all(torch.abs(BlockMask_test - BlockMask.reshape((-1, 16))) < 1e-5, dim=-1).cpu()
# print(torch.nonzero(~condition_out))
# condition_out = torch.all(torch.abs(SparseAToX_test - SparseAToX) < 1e-5, dim=-1).cpu()
# print(condition_out)

# idxs = torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape)))
# print(out_[idxs], out[idxs])
# print(idxs, idxs.shape[0])
# assert(torch.all(condition_out).item())


