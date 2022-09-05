"""
2 way branch:
    OMP_NUM_THREADS=2 torchrun --nproc_per_node=2 --nnodes=1 palm.py
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


@cube.graph.parser.register('N L^ E^, E^ F^, E^ E^ -> N L^ E^',
                            name='multi_head_attention')
def multi_head_attention(x: torch.Tensor, qkv_proj: torch.Tensor,
                         out_proj: torch.Tensor, heads: int, scale: float):
    '''
    x: [bs, len, dim]
    qkv_proj: [dim, dim + dim_head]
    out_proj: [dim, dim]
    '''
    bs, n, dim = x.size()
    dim_head = dim // heads

    q, kv = torch.matmul(x, qkv_proj).split((dim, dim_head), dim=-1)
    q = q.view(bs, n, heads, dim_head).transpose(1, 2) * scale
    q = q.reshape(bs, heads * n, dim_head)
    trans_kv = kv.transpose(1, 2)
    sim = torch.bmm(q, trans_kv).view(bs, heads, n, n)
    attn = torch.nn.functional.softmax(sim, dim=-1)
    attn = attn.view(bs, heads * n, n)
    out = torch.bmm(attn, kv).view(bs, heads, n, dim_head)
    out = torch.transpose(out, 1, 2).reshape(bs, n, dim)
    out = torch.matmul(out, out_proj)
    return out


@cube.graph.parser.register('N L^ E^, E^ F^, G^ H^ -> N L^ H^',
                            name='feedforward')
def feedforward(x: torch.Tensor, proj1: torch.Tensor, proj2: torch.Tensor):
    '''
    x: [bs, len, dim]
    proj1: [dim, 2 * ff_mult * dim]
    proj2: [ff_mult * dim, dim]
    '''
    x = torch.matmul(x, proj1)
    x, gate = x.chunk(2, dim=-1)
    x = torch.nn.functional.silu(gate) * x
    x = torch.matmul(x, proj2)
    return x


class PaLMLayer(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()

        self.dim, self.dim_head, self.heads, self.scale = dim, dim_head, heads, dim_head**-0.5

        # TODO
        # self.alibi_pos_biases = AlibiPositionalBias(heads=self.heads)
        # self.norm = RMSNorm(dim)
        self.norm = torch.nn.LayerNorm(self.dim)

        self.qkv_proj = torch.nn.Parameter(torch.randn(dim, dim + dim_head))
        self.attn_out_proj = torch.nn.Parameter(torch.randn(dim, dim))

        self.ff_proj1 = torch.nn.Parameter(torch.randn(dim, 2 * ff_mult * dim))
        self.ff_proj2 = torch.nn.Parameter(torch.randn(ff_mult * dim, dim))

        # self.register_buffer("mask", None, persistent=False)

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

        attn_out = multi_head_attention(x, self.qkv_proj, self.attn_out_proj,
                                        self.heads, self.scale)

        ff_out = feedforward(x, self.ff_proj1, self.ff_proj2)

        return in_x + attn_out + ff_out


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


def PASData(graph: IRGraph, resource):
    '''
    2 way Data Parallel
    '''
    assert resource.ngpus == 2

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node,
                                        algo,
                                        idx=0,
                                        dim=batch_dim,
                                        num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
    return graph


def PASBranch(graph: IRGraph, resource):
    '''
    2 way brach
    '''
    assert resource.ngpus == 2

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            print(node)
            if node.name == 'embedding' or node.name == 'linear':
                # data parallel
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node,
                                            algo,
                                            idx=0,
                                            dim=batch_dim,
                                            num=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif node.name == 'layernorm' or node.name == 'multiref' or node.name == 'add' or node.name == 'mean':
                # replicate
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif node.name == 'feedforward':
                graph.assign(node, 0)
            elif node.name == 'multi_head_attention':
                graph.assign(node, 1)
            else:
                assert False, node.name

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
                                                    dtypes=(torch.float32, ),
                                                    batch_dims=(0, ))

    # @cube.compile(model, dataloader, PAS=PASSingle)
    # @cube.compile(model, dataloader, PAS=PASData)
    @cube.compile(model, dataloader, PAS=PASBranch)
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
