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

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank

import examples.mlp.policy.spmd as spmd
import examples.mlp.policy.mpmd as mpmd

import argparse

def exists(val):
    return val is not None

# parser = argparse.ArgumentParser(description='comm primitive')
# parser.add_argument('--policy', type=str, help='PAS policy choice, starting with "PAS"')
# args = parser.parse_args()
#
# cube.init()
#
# # set up policy
# PAS = None
# policies = list(spmd.__dict__.keys()) + list(mpmd.__dict__.keys())
# if args.policy in spmd.__dict__:
#     PAS = spmd.__dict__[args.policy]
#     print_each_rank(f'using policy from spmd.{args.policy}')
# elif args.policy in mpmd.__dict__:
#     PAS = mpmd.__dict__[args.policy]
#     print_each_rank(f'using policy from mpmd.{args.policy}')
# else:
#     raise ValueError(f"policy {args.policy} not found. Candidates: {policies}")


# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)
        # self.linear5 = nn.Linear(dim, dim * mult)
        # self.linear6 = nn.Linear(dim * mult, dim)
        # self.linear7 = nn.Linear(dim, dim * mult)
        # self.linear8 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        # output = self.linear5(output)
        # output = self.linear6(output)
        # output = self.linear7(output)
        # output = self.linear8(output)
        loss = torch.sum(output)
        return loss

# normalization

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# AliBi

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

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
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class PaLMLayer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()

        self.dim, self.dim_head, self.heads, self.scale = dim, dim_head, heads, dim_head**-0.5

        self.alibi_pos_biases = AlibiPositionalBias(heads=self.heads)

        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim + dim_head, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.ff = nn.Sequential(
                nn.Linear(dim, 2 * ff_mult * dim, bias=False),
                SwiGLU(),
                nn.Linear(ff_mult * dim, dim, bias=False)
                )

        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), 1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, x):
        n, device = x.shape[1], x.device

        # pre layernorm
        x = self.norm(x)

        ff_out = self.ff(x)

        q, kv = self.qkv_proj(x).split((self.dim, self.dim_head), dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale
        q = q * self.scale

        # similarity
        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        sim = sim + self.alibi_pos_biases(sim)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, kv)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        merge_heads = self.attn_out(out) + ff_out
        return merge_heads

def train():
    bs, n, d = 8, 1024, 512

    model = PaLMLayer(d)

    x = torch.randn(bs, n, d)

    y = model(x)

# def train():
#     batch_size = 256
#     dim = 8192
#
#     model = MLP(dim=dim)
#     model = cube.SemanticModel(
#         model, input_shapes=([batch_size, dim],),
#     )
#
#     dataloader = cube.runtime.syndata.SynDataLoader(
#         shapes=([batch_size, dim],),
#         dtypes=(torch.float32,),
#         batch_dims=(0,)
#     )
#
#     @cube.compile(model, dataloader, PAS=PAS)
#     def train_iter(model, dataloader):
#         data = next(dataloader)
#         loss = model(data)
#         loss.backward()
#     model = model.get_gen_module()
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#     CudaTimer(enable=False).warmup()
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()
#     iter_num = 64
#     warmup = 20
#     for step in range(iter_num):
#         if step >= warmup:
#             CudaTimer(enable=True).start('e2e')
#         train_iter(model, dataloader)
#         optimizer.step()
#         optimizer.zero_grad()
#         if step >= warmup:
#             CudaTimer().stop('e2e')
#         if (step + 1) % 20 == 0:
#             print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
#
#     print_each_rank('e2e time (ms) per iteration: {} ms'.format(
#           CudaTimer().duration(iter_num-warmup, field_name='e2e')))
#     CudaTimer().print_all(times=iter_num-warmup)


train()
