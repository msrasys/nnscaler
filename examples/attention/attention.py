"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/attention/attention.py

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/attention/attention.py
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank

from examples.attention.policy.naive import PAS


@cube.graph.parser.register('L^ N E^, (3 h d^) E^ -> L^ N (h d^)')
def attnfc1(x: torch.Tensor, wqkv: torch.Tensor, h: int,
            scale: float, dropout: float, training: bool):
    """
    L: sequence length
    N: batch size
    E: embedding size
    x: hidden state: [L, N, E]
    wqkv: qkv weight: [3 * (num_head * dim_head), E]
    dropout: float
    h: int: number of heads
    """
    num_head = h
    L, N = x.shape[0], x.shape[1]
    dim_head = wqkv.shape[0] // 3 // num_head
    # L N E, (3 h d) E -> L N (3 h d)
    qkv = torch.nn.functional.linear(x, wqkv, None)
    # L N (3 h d) -> L N (h d), L N (h d), L N (h d)
    q, k, v = qkv.chunk(3, dim=-1)
    # L N (h d) -> L (N h) d
    q = q.contiguous().view(L, (N * num_head), dim_head)
    # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head)
    # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head)
    # L (N h) d -> (N h) L d
    q = q.transpose(0, 1)
    # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)
    # L (N h) d -> (N h) L d
    v = v.transpose(0, 1)
    # (N h) L d, 1 -> (N h) L d
    q = q * scale
    # (N h) L d -> (N h) d L
    k = k.transpose(-2, -1)
    # (N h) L d, (N h) d L -> (N h) L L
    attn = torch.bmm(q, k)

    # attention mask
    # (N h) L L -> (N h) L L
    attn = attn.view(N, num_head, L, L)
    ones = torch.ones((N, L, L), device=attn.device)
    mask = torch.tril(ones)
    mask = mask.view(N, 1, L, L)
    mask = (mask < 0.5)
    attn = attn.masked_fill_(mask, -10000.0)
    attn = attn.view((N * num_head), L, L)

    # (N h) L L -> (N h) L L
    attn = torch.nn.functional.softmax(attn, dim=-1)
    # (N h) L L -> (N h) L L
    if training:
        attn = torch.nn.functional.dropout(attn, dropout, True, False)
    # (N h) L L, (N h) L d -> (N h) L d
    output = torch.bmm(attn, v)
    # (N h) L d -> L (N h) d
    output = output.transpose(0, 1).contiguous()
    # L (N h) d -> L N (h d)
    output = output.view(L, N, num_head * dim_head)
    return output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, seq_len, embed_dim, heads, dropout: float):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_head = heads
        self.dim_head = embed_dim // heads
        self.scale = self.dim_head ** -0.5

        self.wqkv = torch.nn.Parameter(torch.empty(
            3 * embed_dim, embed_dim
        ))
        self.wout = torch.nn.Parameter(torch.empty(
            embed_dim, embed_dim
        ))
        self.dropout = dropout

    def forward(self, x):
        """
        x: [L, N, E]: seq_len, batch_size, embedding dimension
        output: [L, N, E]
        """
        # L N E, (3 h d) E -> L N (h d)
        output = attnfc1(x, self.wqkv, self.num_head,
                         self.scale, self.dropout, self.training)
        # L N (h d), E (h d) -> L N E
        output = torch.nn.functional.linear(output, self.wout)

        loss = torch.sum(output)
        return loss


def train():
    L = 512  # seq len
    N = 32   # batch size
    # configs: [hidden size, num_head]
    # E, num_head = [1536, 16]  # 1.2B model
    # E, num_head = [1920, 20]  # 2.5B model
    # E, num_head = [2304, 24]  # 4.2B model
    E, num_head = [3072, 32]  # 8.7B model


    model = MultiHeadSelfAttention(
        seq_len=L, embed_dim=E, heads=num_head, dropout=0.5
    )
    model = cube.SemanticModel(
        model, input_shapes=([L, N, E],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([L, N, E],),
        dtypes=(torch.float32,),
        batch_dims=(1,)
    )

    @cube.compile(model, dataloader, PAS=PAS)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer().warmup()
    torch.distributed.barrier()
    iter_num = 128
    for step in range(iter_num):
        if step >= 40:
            CudaTimer().start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= 40:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))


if __name__ == '__main__':

    cube.init()
    train()
