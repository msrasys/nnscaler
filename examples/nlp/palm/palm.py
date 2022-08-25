"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASMegatron
"""

import torch
import torch.nn.functional as F
from torch import nn, einsum

from math import log2, floor

from einops import rearrange, repeat

from cube.graph import IRGraph

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.ir.operator import IRDataOperation, IRFwOperation

import examples.mlp.policy.spmd as spmd
import examples.mlp.policy.mpmd as mpmd

import argparse

cube.init()

# =================== Semantic Model Description ====================

# normalization


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def exists(val):
    return val is not None


# AliBi


class AlibiPositionalBias(nn.Module):

    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, 'j -> 1 1 j') -
            rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):

        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2**floor(log2(heads))
        return get_slopes_power_of_2(
            closest_power_of_2) + get_slopes_power_of_2(
                2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return bias


class SwiGLU(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        # x, gate = x.chunk(2, dim=-1)
        u, gate = x[:, :, 0:self.dim // 2], x[:, :, self.dim // 2:]
        return F.silu(gate) * u


class PaLMLayer(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()

        self.dim, self.dim_head, self.heads, self.scale = dim, dim_head, heads, dim_head**-0.5

        # TODO
        # self.alibi_pos_biases = AlibiPositionalBias(heads=self.heads)
        # self.norm = RMSNorm(dim)
        self.norm = torch.nn.LayerNorm(self.dim)

        self.qkv_proj = nn.Linear(dim, dim + dim_head, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.ff = nn.Sequential(nn.Linear(dim, 2 * ff_mult * dim, bias=False),
                                SwiGLU(2 * ff_mult * dim),
                                nn.Linear(ff_mult * dim, dim, bias=False))

        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool),
                          1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, in_x):
        bs, n, device = in_x.shape[0], in_x.shape[1], in_x.device

        # pre layernorm
        x = self.norm(in_x)

        ff_out = self.ff(x)

        # q, kv = self.qkv_proj(x).split((self.dim, self.dim_head), dim=-1)
        proj = self.qkv_proj(x)
        q, kv = proj[:, :, 0:self.dim], proj[:, :, self.dim:]

        # q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q.view(bs, n, self.heads, self.dim_head).transpose(1, 2)

        # scale
        q = q * self.scale

        # similarity
        # sim = einsum('b h i d, b j d -> b h i j', q, kv)
        q = q.reshape(bs, self.heads * n, self.dim_head)
        trans_kv = kv.transpose(1, 2)
        sim = torch.bmm(q, trans_kv).view(bs, self.heads, n, n)

        # TODO
        # sim = sim + self.alibi_pos_biases(sim)

        # TODO
        # causal mask

        # causal_mask = self.get_mask(n, device)
        # sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        # attn = sim.softmax(dim=-1)
        attn = torch.nn.functional.softmax(sim, dim=-1)

        # out = einsum("b h i j, b j d -> b h i d", attn, kv)
        attn = attn.view(bs, self.heads * n, n)
        out = torch.bmm(attn, kv).view(bs, self.heads, n, self.dim_head)

        # merge heads

        # out = rearrange(out, "b h n d -> b n (h d)")
        out = torch.transpose(out, 1, 2).reshape(bs, n, self.dim)

        merge_heads = self.attn_out(out) + ff_out
        return in_x + merge_heads


class PaLM(nn.Module):

    def __init__(self,
                 dim,
                 num_tokens,
                 depth,
                 dim_head=64,
                 heads=8,
                 ff_mult=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Embedding(num_tokens, dim),
            *[PaLMLayer(dim, dim_head, heads, ff_mult) for _ in range(depth)],
            # TODO: RMSNorm(dim),
            torch.nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False),
        )

        self.net[-1].weight = self.net[0].weight
        nn.init.normal_(self.net[0].weight, std=0.02)

    def forward(self, x):
        return self.net(x).mean()


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1

    for node in graph.nodes():
        if isinstance(node, (IRDataOperation, IRFwOperation)):
            graph.assign(node, 0)

    return graph


def train():
    bs, n, dim = 8, 1024, 512
    num_tokens, depth, heads, dim_head = 20000, 1, 8, 64

    model = PaLM(dim, num_tokens, depth, heads=heads, dim_head=dim_head)

    # for debug
    # tokens = torch.randint(0, num_tokens, (bs, n))
    # print(model(tokens))
    # return

    model = cube.SemanticModel(
        model,
        input_shapes=([bs, n], ),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(shapes=([bs, n], ),
                                                    dtypes=(torch.int32, ),
                                                    batch_dims=(0, ))

    @cube.compile(model, dataloader, PAS=PASSingle)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()

    model = model.get_gen_module()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer(enable=False).warmup()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    iter_num = 64
    warmup = 20
    for step in range(iter_num):
        if step >= warmup:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= warmup:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num - warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num - warmup)


train()
