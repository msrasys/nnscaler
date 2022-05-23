"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/dummy/transformers.py
"""
import torch
from torch.utils import checkpoint

import cube
from cube.profiler.memory import memory_summary, model_summary
from cube.profiler.timer import CudaTimer, print_each_rank
import argparse


parser = argparse.ArgumentParser(description='transformer')
# model arch
parser.add_argument('--layers', type=int, default=4,
                    help='number encoder/decoder of layers')
parser.add_argument('--hidden-size', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=16,
                    help='number of heads')
parser.add_argument('--bs', type=int, default=8,
                    help='number of heads')
# parallelism
parser.add_argument('--seq', type=int, default=1,
                    help='sharding sequential execution')
args = parser.parse_args()
print(args)


cube.init()


class Config:

    layers    = args.layers
    embed_dim = args.hidden_size
    num_heads = args.heads
    ffn_dim   = embed_dim * 4

    seqlen = 1024


def self_attention(query: torch.Tensor,
                   q_proj: torch.Tensor, q_bias: torch.Tensor,
                   k_proj: torch.Tensor, k_bias: torch.Tensor,
                   v_proj: torch.Tensor, v_bias: torch.Tensor,
                   out_proj: torch.Tensor, out_bias: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask=True):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = q_proj.size(0) // num_head

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
    k = k.transpose(1, 2)  # (N h) L d -> (N h) d L
    attn = torch.bmm(q, k) # (N h) L d, (N h) d L -> (N h) L L

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        mask = torch.tril(ones)
        mask = mask.view(N, 1, L, L)
        mask = (mask < 0.5)
        attn = attn.masked_fill_(mask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj, out_bias) # L N (h d), E E  -> L N E
    return output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0, bias=True):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim)) if bias else None

    def forward(self, query):
        # x = self_attention(
        #     query,
        #     self.q_proj, self.q_bias,
        #     self.k_proj, self.k_bias,
        #     self.v_proj, self.v_bias,
        #     self.out_proj, self.out_bias,
        #     self.num_heads, self.scaling, self.dropout_p, mask=True
        # )
        x = checkpoint.checkpoint(
            self_attention,
            query,
            self.q_proj, self.q_bias,
            self.k_proj, self.k_bias,
            self.v_proj, self.v_bias,
            self.out_proj, self.out_bias,
            self.num_heads, self.scaling, self.dropout_p, True
        )
        return x


class SeqMHSA(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 inner_dim: int, dropout: float = 0.0,
                 bias=True, shard_num=4):
        super().__init__()
        assert num_heads % shard_num == 0
        print_each_rank(f'using sequence MHSA: sharding size: {shard_num}')
        self.layers = torch.nn.ModuleList(
            MultiHeadSelfAttention(
                embed_dim,
                num_heads // shard_num,
                inner_dim // shard_num,
                dropout,
                bias
            ) for _ in range(shard_num)
        )

    def forward(self, x):
        out_sum = None
        for layer in self.layers:
            out = layer(x)
            out_sum = out if out_sum is None else out_sum + out
        return out_sum


def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor, proj2_bias: torch.Tensor,
                dropout: float) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, True, False)
    x = torch.nn.functional.linear(x, proj2, proj2_bias)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout: float, bias=True):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = checkpoint.checkpoint(
            feedforward,
            x, self.proj1, self.proj1_bias, self.proj2, self.proj2_bias, self.dropout
        )
        # x = feedforward(x,
        #                 self.proj1, self.proj1_bias,
        #                 self.proj2, self.proj2_bias,
        #                 self.dropout)
        return x


class SeqMLP(torch.nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout: float,
                 bias=True, shard_num = 4):
        super().__init__()
        print_each_rank(f'using sequence MLP: sharding size: {shard_num}')
        assert hidden_dim % shard_num == 0
        self.layers = torch.nn.ModuleList(
            [MLP(embed_dim, hidden_dim // shard_num, dropout, bias) for _ in range(shard_num)]
        )

    def forward(self, x: torch.Tensor):
        out_sum = None
        for layer in self.layers:
            out = layer(x)
            out_sum = out if out_sum is None else out_sum + out
        return out_sum


class TransformerLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, attn_inner_dim: int, ffn_embed_dim: int,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()

        if args.seq > 1:
            self.self_attn = SeqMHSA(embed_dim, num_heads, attn_inner_dim, atten_dropout, shard_num=args.seq)
        else:
            self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_inner_dim, atten_dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

        if args.seq > 1:
            self.mlp = SeqMLP(embed_dim, ffn_embed_dim, activation_dropout, shard_num=args.seq)
        else:
            self.mlp = MLP(embed_dim, ffn_embed_dim, activation_dropout)

        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.layers = torch.nn.ModuleList(
            [TransformerLayer(
                self.cfg.embed_dim,
                self.cfg.num_heads,
                self.cfg.embed_dim,
                self.cfg.ffn_dim
            ) for _ in range(self.cfg.layers)]
        )

    def forward(self, x: torch.Tensor):  # L N E

        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


class DataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.cfg = Config()
        super().__init__(
            shapes=([self.cfg.seqlen, batch_size, self.cfg.embed_dim],),
            dtypes=(torch.float,),
            batch_dims=(1,)
        )
        self.samples = [self.random_sample()]

    def random_sample(self):
        inputs = torch.randn(
            *(self.cfg.seqlen, self.bs, self.cfg.embed_dim),
            dtype=torch.float,
            device=torch.cuda.current_device(),
            requires_grad=True
        )
        return (inputs,)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]



if __name__ == '__main__':

    dataloader = DataLoader(batch_size=args.bs)
    model = Model().cuda()
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-05)

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    iter_num = 10
    for step in range(iter_num):
        dataloader = iter(dataloader)
        if step >= 4:
            CudaTimer(enable=True).start('e2e')
        if step == 0:
            model_summary(model, next(dataloader))
        loss = model(*next(dataloader))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step >= 4:
            CudaTimer().stop('e2e')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-4, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-4)
    memory_summary()