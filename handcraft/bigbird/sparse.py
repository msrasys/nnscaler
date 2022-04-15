"""
BigBird paper
https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf

Understanding blog:
https://github.com/huggingface/blog/blob/main/big-bird.md


OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/bigbird/sparse.py \
        --hidden-size 5120 --heads 32 --seqlen 12288 \
        --bs 1 --fp16
"""

import torch
import torch.nn as nn
import cube
import math
import numpy as np

import argparse
from cube.runtime.device import DeviceGroup
from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank

from cube.runtime.adapter.distnn import AllGatherSplit, IdentityAllreduce



parser = argparse.ArgumentParser(description='sparse_attention')

parser.add_argument('--hidden-size', type=int, default=4096,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=32,
                    help='number of heads')
parser.add_argument('--seqlen', type=int, default=4096,
                    help='sequence length')
parser.add_argument('--blk-size', type=int, default=64,
                    help='sequence length')
# training config
parser.add_argument('--bs', type=int, default=256,
                    help='num of micro batch')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--sparse', action='store_true', default=False)
args = parser.parse_args()

print(args)
cube.init()

tp_ranks = list(range(DeviceGroup().world_size))
_tp_group = -1
_tp_size = len(tp_ranks)
_tp_rank = DeviceGroup().rank
if len(tp_ranks) > 1:
    print_each_rank(f'initializing tp ranks: {tp_ranks}')
    _tp_group = DeviceGroup().get_group(tp_ranks)


class Config:

    num_attention_heads = args.heads
    hidden_size = args.hidden_size
    all_head_sie = hidden_size
    seqlen = args.seqlen # seqlen
    num_random_blocks = 2
    block_size=args.blk_size
    use_bias = True

config = Config()


def bmm(tensor1: torch.Tensor, tensor2: torch.Tensor, ndim: int, out=None):
    # print(f'bmm: {tensor1.size()} {tensor2.size()}')
    return torch.bmm(
        tensor1.reshape((-1,) + tensor1.shape[-2:]),
        tensor2.reshape((-1,) + tensor2.shape[-2:]),
        out=out
    ).view(tensor1.shape[: ndim - 2] + (tensor1.shape[ndim - 2], tensor2.shape[ndim-1]))


def stride_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              h: int, block_size: int, out=None):
    CudaTimer().start('stride_q@k')
    # print('start stride qk')
    # q, k, v: (N h) L d
    num_head = h
    L, N = q.size(1), q.size(0) // h
    dim_head = q.size(2)
    assert L % block_size == 0

    q = q.view(N * num_head, L // block_size, block_size, dim_head) # (N h) nblock blksize d
    k = k.view(N * num_head, L // block_size, block_size, dim_head) # (N h) nblock blksize d

    # stride diagnal [2:]
    middle_q = q[:, 2:] # (N h) (nblock-2) blksize d
    # (N h) nblock-3 (3 blksize) d
    sliding_keys = torch.cat((k[:, 1:-2], k[:, 2:-1], k[:, 3:]), dim=2)
    # (N h) 1 blksize d
    pad_zero = torch.zeros_like(k[:,-3:-2])
    # (N h) 1 (3 blksize) d
    sliding_bottom_keys = torch.cat((pad_zero, k[:,-2:-1], k[:,-1:]), dim=2)
    # (N h) (nblock-2) (3 blksize) d
    sliding_keys = torch.cat((sliding_keys, sliding_bottom_keys), dim=1)
    # (N h) (nblock-2) d (3 blksize)
    sliding_keys = sliding_keys.transpose(2, 3)

    # (N h) (nblock-2) blksize (3 blksize)
    out = bmm(middle_q, sliding_keys, ndim=4, out=out)
    # (N h) ((nblock-2) blksize) (3 blksize)
    qk = out.view(N * h, -1, block_size * 3)
    CudaTimer().stop('stride_q@k')
    return qk


def parallel_stride_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       h: int, block_size: int, start: int, end: int):
    CudaTimer().start('parallel_stride_q@k')
    # print('start stride qk')
    # q, k, v: (N h) L d
    num_head = h
    L, N = q.size(1), q.size(0) // h
    dim_head = q.size(2)
    assert L % block_size == 0

    q = q.view(N * num_head, L // block_size, block_size, dim_head) # (N h) nblock blksize d
    k = k.view(N * num_head, L // block_size, block_size, dim_head) # (N h) nblock blksize d

    # (N h) 1 blksize d
    pad_zero = torch.zeros_like(k[:,-1:])
    # (N h) (end-start) blksize d
    middle_q = q[:,2+start:2+end]
    if end + 2 == L // block_size:
        # (N h) (end-start) blksize d
        k_right = torch.cat((k[:,start+3:], pad_zero), dim=1)
    else:
        k_right = k[:,start+3:end+3]
    # (N h) (end-start) (3 blksize) d
    sliding_keys = torch.cat((k[:, start+1:end+1], k[:, start+2:end+2], k_right), dim=2)
    # (N h) (end-start) blksize (3 blksize)
    qk = bmm(middle_q, sliding_keys.transpose(2, 3), ndim=4)
    # (N h) (nblock-2) blksize (3 blksize)
    qk = torch.nn.functional.pad(
        qk, (0,0,0,0,start,L//block_size-2-end), 'constant', 0)
    # (N h) ((nblock-2) blksize) (3 blksize)
    qk = qk.view(N * h, -1, block_size * 3)
    CudaTimer().stop('parallel_stride_q@k')
    return qk


def stride_v(v: torch.Tensor, h: int, block_size: int):
    L, N, dim_head = v.size(1), v.size(0) // h, v.size(2)
    assert L % block_size == 0
    v = v.view(N * h, L // block_size, block_size, dim_head)
    # (N h) 1 blksize d
    pad_zero = torch.zeros_like(v[:,-3:-2])
    # (N h) (nblock-3) (3 blksize) d
    stride_vals = torch.cat((v[:, 1:-2], v[:, 2:-1], v[:, 3:]), dim=2)
    # (N h) 1 (3 blksize) d
    stride_bottom_vals = torch.cat((v[:,-2:-1], v[:,-1:], pad_zero), dim=2)
    # (N h) (nblock-2) (3 blksize) d
    stride_vals = torch.cat((stride_vals, stride_bottom_vals), dim=1)
    return stride_vals


def global_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              h: int, block_size: int):
    CudaTimer().start('global_q@k')
    # print('start global qk')
    # q, k, v: (N h) L d
    num_head = h
    L, N = q.size(1), q.size(0) // h
    dim_head = q.size(2)
    assert L % block_size == 0

    # first two row
    head_q = q[:, :2 * block_size] # (N h) (2 blocksize) d
    head_k = k.transpose(1, 2) # (N h) d L
    head = bmm(head_q, head_k, ndim=3) # (N h) (2 blocksize) L
    # (N h) L d
    head_v = v

    # remain first two column
    col_q = q[:, 2 * block_size:] # (N h) ((nblock-2) blocksize) d
    col_k = k[:, :2 * block_size].transpose(1, 2) # (N h) d (2 blocksize)
    # (N h) (2 blksize) d
    col_v = v[:, :2 * block_size]
    col = bmm(col_q, col_k, ndim=3) # (N h) ((nblock-2) blocksize) (2 blocksize)
    CudaTimer().stop('global_q@k')
    return head, head_v, col, col_v


def randn_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             h: int, block_size: int, rand_num: int = 2, out=None):
    CudaTimer().start('rand_q@k')
    torch.manual_seed(0)
    # q, k, v: (N h) L d
    # rand_num = 2
    num_head = h
    L, N = q.size(1), q.size(0) // h
    dim_head = q.size(2)
    # nblock-2 2
    indices = torch.randint(
        2, L // block_size, (L // block_size-2, rand_num),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    # (N h) nblock blksize d
    k = k.view(N * num_head, L // block_size, block_size, dim_head)

    # Optimize: remove for loop, use direct index can greatly speedup
    # (N h) nblock-2 (randnum blksize) d
    keys = tuple(k[:,indices[:,idx]] for idx in range(rand_num))
    gathered_k = torch.cat(keys, dim=2)

    # (N h) nblock blksize d
    q = q.view(N * num_head, L // block_size, block_size, dim_head)
    # (N h) nblock-2 blksize d
    q = q[:,2:]
    # (N h) nblock-2 blksize (randnum blksize)
    out = bmm(q, gathered_k.transpose(2, 3), ndim=4, out=out)
    # (N h) ((nblock-2) blksize) (randnum blksize)
    qk = out.view(N * h, -1, rand_num * block_size)
    CudaTimer().stop('rand_q@k')
    return qk


def randn_v(v: torch.Tensor, h: int, block_size: int, rand_num: int = 2):
    # v: (N h) L d
    # CudaTimer().start('rand_v')
    torch.manual_seed(0)
    L, N, dim_head = v.size(1), v.size(0) // h, v.size(2)
    # nblock-2 2
    indices = torch.randint(
        2, L // block_size, (L // block_size-2, rand_num),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    v = v.view(N * h, L // block_size, block_size, dim_head)
    vals = tuple(v[:,indices[:,idx]] for idx in range(rand_num))
    gathered_v = torch.cat(vals, dim=2)
    # CudaTimer().stop('rand_v')
    return gathered_v


def sparse_attn(query: torch.Tensor,
                q_proj: torch.Tensor, q_bias: torch.Tensor,
                k_proj: torch.Tensor, k_bias: torch.Tensor,
                v_proj: torch.Tensor, v_bias: torch.Tensor,
                out_proj: torch.Tensor, out_bias: torch.Tensor,
                h: int, scale: float, dropout_p: float, block_size: int):
    rand_num = 2
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = q_proj.size(0) // num_head
    nblocks = L // block_size

    CudaTimer().start('to_qkv')
    q = torch.nn.functional.linear(query, q_proj, q_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(query, k_proj, k_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(query, v_proj, v_bias)   # L N E, (h d) E -> L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    q = q.transpose(0, 1)  # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q * scale          # (N h) L d, 1 -> (N h) L d
    CudaTimer().stop('to_qkv')

    # sqk = torch.empty(
    #     N * h, (nblocks - 2) * block_size, 3 * block_size,
    #     dtype=torch.float16 if args.fp16 else torch.float32,
    #     device=torch.cuda.current_device()
    # )
    # 
    # rqk = torch.empty(
    #     N * h, (nblocks - 2) * block_size, 2 * block_size,
    #     dtype=torch.float16 if args.fp16 else torch.float32,
    #     device=torch.cuda.current_device()
    # )
    # we don't need pre-allocation as memory are sufficient
    sqk = rqk = None

    CudaTimer().start('q@k')
    # sqk:   (N h) ((nblock-2) blksize) (3 blksize)
    sqk = stride_qk(q, k, v, h, block_size=block_size, out=sqk)
    # head: (N h) (2 blocksize) L
    # head_v: (N h) L d
    # col: (N h) ((nblock-2) blocksize) (2 blocksize)
    # col_v: (N h) (2 blksize) d
    head, head_v, col, col_v = global_qk(q, k, v, h, block_size=block_size)
    # rqk: (N h) ((nblock-2) blksize) (2 blksize)
    rqk = randn_qk(q, k, v, h, block_size=block_size, out=rqk)
    # (N h) ((nblock-2) blksize) (7 blksize)
    middle_attn = torch.cat((col, sqk, rqk), dim=-1)
    CudaTimer().stop('q@k')

    CudaTimer().start('all_softmax')
    # (N h) ((nblock-2) blksize) L
    head_attn = torch.nn.functional.softmax(head, dim=-1)
    head_attn = torch.nn.functional.dropout(head_attn, dropout_p, True, False)
    # (N h) ((nblock-2) blksize) (7 blksize)
    middle_attn = torch.nn.functional.softmax(middle_attn, dim=-1)
    middle_attn = torch.nn.functional.dropout(middle_attn, dropout_p, True, False)
    CudaTimer().stop('all_softmax')

    CudaTimer().start('qk@v')
    # sqk_v: (N h) (nblock-2) (3 blksize) d
    sqk_v = stride_v(v, h, block_size)
    # rqk_v: (N h) (nblock-2) (2 blksize) d
    rqk_v = randn_v(v, h, block_size)

    CudaTimer().start('global_qk@v')
    # (N h) (2 blocksize) L, (N h) L d -> (N h) (2 blksize) d
    head_output = bmm(head_attn, v, ndim=3)

    # global col v:  (N h) ((nblock-2) blksize) (2 blksize), (N h) (2 blksize) d
    #             :-> (N h) (L-(2 blksize)) d
    middle_output = bmm(middle_attn[:,:,:2 * block_size], col_v, ndim=3)
    CudaTimer().stop('global_qk@v')

    CudaTimer().start('stride_qk@v')
    middle_stride = middle_attn[:,:,2*block_size:5*block_size].view(
        N * h, L // block_size - 2, block_size, 3 * block_size
    )
    # stide v: (N h) (nblock-2) blksize (3 blksize), (N h) (nblock-2) (3 blksize) d
    #        : -> (N h) (nblock-2) blksize d
    middle_stride_output = bmm(middle_stride, sqk_v, ndim=4)
    middle_output += middle_stride_output.view(N * h, -1, sqk_v.size(-1))
    CudaTimer().stop('stride_qk@v')

    # (N h) (nblock-2) blksize (randnum blksize)
    CudaTimer().start('rand_qk@v')
    middle_rand = middle_attn[:,:,5*block_size:].view(
        N * h, L // block_size - 2, block_size, rand_num * block_size
    )
    # rand v: (N h) (nblock-2) blksize (2 blksize), (N h) (nblock-2) (2 blksize) d
    #         -> (N h) (nblock-2) blksize d
    middle_rand_output = bmm(middle_rand, rqk_v, ndim=4)
    middle_output += middle_rand_output.view(N * h, -1, rqk_v.size(-1))
    CudaTimer().stop('rand_qk@v')

    # (N h) (2 blksize) d, (N h) ((nblock-2) blksize) d -> (N h) L d
    output = torch.cat((head_output, middle_output), dim=1)
    CudaTimer().stop('qk@v')

    CudaTimer().start('out_proj')
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj, out_bias) # L N (h d), E E  -> L N E
    CudaTimer().stop('out_proj')
    return output


def dense_attn(query: torch.Tensor,
               q_proj: torch.Tensor, q_bias: torch.Tensor,
               k_proj: torch.Tensor, k_bias: torch.Tensor,
               v_proj: torch.Tensor, v_bias: torch.Tensor,
               out_proj: torch.Tensor, out_bias: torch.Tensor,
               h: int, scale: float, dropout_p: float, block_size: int):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = q_proj.size(0) // num_head

    CudaTimer().start('to_qkv')
    q = torch.nn.functional.linear(query, q_proj, q_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(query, k_proj, k_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(query, v_proj, v_bias)   # L N E, (h d) E -> L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q.transpose(0, 1).contiguous()  # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q * scale          # (N h) L d, 1 -> (N h) L d
    CudaTimer().stop('to_qkv')
    # k = k.transpose(1, 2).contiguous()  # (N h) L d -> (N h) d L

    CudaTimer().start('allocation')
    attn = torch.empty(
        (N * h), L, L, dtype=torch.float16 if args.fp16 else args.fp32,
        device=torch.cuda.current_device()
    )
    CudaTimer().stop('allocation')

    CudaTimer().start('q@k')
    attn = torch.bmm(q, k.transpose(1, 2), out=attn) # (N h) L d, (N h) d L -> (N h) L L
    CudaTimer().stop('q@k')

    # attention mask
    attention_mask = torch.ones(((N, L)), device=torch.cuda.current_device())
    attention_mask = attention_mask.view(N, L // block_size, block_size)
    exp_blocked_to_pad = torch.cat(
        [attention_mask[:, 1:-3], attention_mask[:, 2:-2], attention_mask[:, 3:-1]], dim=2
    )
    band_mask = torch.einsum("blq,blk->blqk", attention_mask[:, 2:-2], exp_blocked_to_pad)
    band_mask.unsqueeze_(1)
    band_mask = band_mask < 0.5
    # attn.masked_fill_(band_mask, -1000.0)

    CudaTimer().start('all_softmax')
    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    CudaTimer().stop('all_softmax')

    CudaTimer().start('qk@v')
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    CudaTimer().stop('qk@v')

    CudaTimer().start('out_proj')
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj, out_bias) # L N (h d), E E  -> L N E
    CudaTimer().stop('out_proj')
    return output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.kdim = config.hidden_size
        self.vdim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = 0.0
        self.block_size = config.block_size
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.q_bias = torch.nn.Parameter(torch.empty(config.hidden_size))
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(config.hidden_size, self.kdim))
        self.k_bias = torch.nn.Parameter(torch.empty(config.hidden_size))
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(config.hidden_size, self.vdim))
        self.v_bias = torch.nn.Parameter(torch.empty(config.hidden_size))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.out_bias = torch.nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, query):
        if args.sparse:
            if _tp_size > 1:
                return parallel_sparse_attn(
                    query,
                    self.q_proj, self.q_bias,
                    self.k_proj, self.k_bias,
                    self.v_proj, self.v_bias,
                    self.out_proj, self.out_bias,
                    self.num_heads, self.scaling, self.dropout_p, self.block_size
                )
            else:
                return sparse_attn(
                    query,
                    self.q_proj, self.q_bias,
                    self.k_proj, self.k_bias,
                    self.v_proj, self.v_bias,
                    self.out_proj, self.out_bias,
                    self.num_heads, self.scaling, self.dropout_p, self.block_size
                )
        else:
            return dense_attn(
                query,
                self.q_proj, self.q_bias,
                self.k_proj, self.k_bias,
                self.v_proj, self.v_bias,
                self.out_proj, self.out_bias,
                self.num_heads, self.scaling, self.dropout_p, self.block_size
            )


class AttnDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):
        self.bs = batch_size
        super().__init__(
            shapes=(
                [config.seqlen, args.bs, config.hidden_size],
            ),
            dtypes=(torch.float16 if args.fp16 else torch.float,),
            batch_dims=(0,)
        )
        self.samples = [self.random_sample()]

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]

    def random_sample(self):
        hidden_state = torch.randn(
            config.seqlen, self.bs, config.hidden_size,
            dtype=torch.float16 if args.fp16 else torch.float,
            device=torch.cuda.current_device()
        )
        return hidden_state



if __name__ == '__main__':

    model = MultiHeadSelfAttention()
    nparams = sum([param.numel() for param in model.parameters()])
    print_each_rank(f'model params (M): {nparams / 1e6}. Launching model...')
    model = model.half().cuda() if args.fp16 else model.cuda()
    model.eval()

    dataloader = AttnDataLoader(args.bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    print_each_rank('model weight consumpition:')
    memory_summary()

    CudaTimer(enable=False)
    torch.distributed.barrier()
    iter_num =32
    for step in range(iter_num):
        if step >= 8:
            CudaTimer(enable=True).start('e2e')

        # train 1 step
        # num_microbatch = 1
        with torch.no_grad():
            data = next(dataloader)
            out = model(data)
        # loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step >= 8:
            CudaTimer().stop('e2e')

        torch.cuda.empty_cache()
        torch.distributed.barrier()

        if step == 0:
            print_each_rank('memory after optimizer:', rank_only=0)
            memory_summary()

        if (step + 1) % 8 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-8, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-8)
    memory_summary()