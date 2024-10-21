import torch

import mygraph

g = torch.randint(0, 2, (100, 100))
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
    
RowWindowOffset_test, TCOffset_test, BlockMask_test, SparseAToX_test = torchcore_process_DTC(g, 16, 8, 100)
print(adj.shape, adj, res)
RowWindowOffset, TCOffset, BlockMask, SparseAToX = mygraph.process_DTC(adj.to("cuda"), 16, 8, 100)

print(RowWindowOffset_test, RowWindowOffset)
condition_out = torch.all(torch.abs(RowWindowOffset_test - RowWindowOffset) < 1e-5, dim=-1).cpu()
print(condition_out)
print(TCOffset_test, TCOffset)
condition_out = torch.all(torch.abs(TCOffset_test - TCOffset) < 1e-5, dim=-1).cpu()
print(condition_out)
condition_out = torch.all(torch.abs(BlockMask_test - BlockMask.reshape((-1, 16))) < 1e-5, dim=-1).cpu()
print(torch.nonzero(~condition_out))
condition_out = torch.all(torch.abs(SparseAToX_test - SparseAToX) < 1e-5, dim=-1).cpu()
print(condition_out)

# idxs = torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape)))
# print(out_[idxs], out[idxs])
# print(idxs, idxs.shape[0])
# assert(torch.all(condition_out).item())


