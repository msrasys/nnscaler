"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/attention.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import cube


from examples.attention.policy.tensor_parallel import transform_policy
from examples.attention.policy.tensor_parallel import schedule_policy

from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, seq_len, embed_dim, heads, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_head = heads
        self.dim_head = embed_dim // heads
        self.scale = self.dim_head ** -0.5

        self.weight_qkv = torch.nn.Parameter(torch.empty(
            3 * embed_dim, embed_dim
        ))
        self.weight_out = torch.nn.Parameter(torch.empty(
            embed_dim, embed_dim
        ))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [L, N, E]: seq_len, batch_size, embedding dimension
        output: [L, N, E]
        """
        # [L, N, E] -> 3 x [L, (N * num_head), dim_head]
        q, k, v = cube.runtime.function.toqkv(
            x, self.weight_qkv, self.num_head
        )

        # [L, (N * num_head), dim_head] -> [(N * num_head), L, dim_head]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # [(N * num_head), L, dim_head] -> [(N * num_head), L, dim_head]
        q = q * self.scale
        # [(N * num_head), L, dim_head] -> [(N * num_head), dim_head, L]
        k = k.transpose(-2, -1)
        # [(N * num_head), L, dim_head] * [(N * num_head), dim_head, L]
        # -> [(N * num_head), L, L]
        attn = torch.bmm(q, k)

        # [(N * num_head), L, L] -> [(N * num_head), L, L]
        attn = cube.runtime.function.tril_mask(attn, self.num_head)

        # [(N * num_head), L, L] -> [(N * num_head), L, L]
        attn = F.softmax(attn, dim=-1)

        # [(N * num_head), L, L] -> [(N * num_head), L, L]
        attn = self.dropout(attn)
        # [(N * num_head), L, L] * [(N * num_head), L, dim_head]
        # -> [(N * num_head), L, dim_head]
        output = torch.bmm(attn, v)

        # [(N * num_head), L, dim_head] -> [L, N, num_head * dim_head]
        output = cube.runtime.function.attn_view(output, self.num_head)

        # [L, (N * num_head), dim_head] * []
        output = F.linear(output, self.weight_out)

        loss = torch.sum(output)
        return loss


def train():
    L = 128  # seq len
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
        # TODO: data parallel batch dim
        model, input_shapes=([L, N, E],),
    )
    
    # TODO: data parallel batch dim
    dataloader = cube.runtime.syndata.SynDataLoader(1280, [1], [L, N, E])

    @cube.compile(model, dataloader, policy=(transform_policy, schedule_policy))
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
