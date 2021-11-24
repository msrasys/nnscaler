"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/transformer/transformers.py
"""

import torch
from torch import nn
import cube

from examples.transformer.policy.megatron_parallel import transform_policy
from examples.transformer.policy.megatron_parallel import schedule_policy

from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, heads, dropout):
        super().__init__()

        self.num_head = heads
        self.dim_head = embed_dim // heads
        self.dropout = dropout

        self.weight_qkv = torch.nn.Parameter(torch.empty(
            3 * embed_dim, embed_dim
        ))
        self.weight_out = torch.nn.Parameter(torch.empty(
            embed_dim, embed_dim
        ))

    def forward(self, x):
        """
        Multi-Head Self-Attention.

        L: sequence length
        N: batch size
        E: embedding size

        Inputs:
            hidden_state: [L, N, E]
            w_qkv       : [3 * num_head * dim_head, E]
            w_out       : [E, E]

        Outputs:
            hidden_state: [L, N, E]
        """
        
        hidden_state = cube.runtime.function.complex.self_attn(
            x, self.weight_qkv, self.weight_out,
            self.num_head, self.dim_head, self.dropout
        )
        return hidden_state


class FFN(torch.nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj1_weight = torch.nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size)
        )
        self.proj1_bias = torch.nn.Parameter(
            torch.empty(4 * hidden_size)
        )
        self.proj2_weight = torch.nn.Parameter(
            torch.empty(hidden_size, 4 * hidden_size)
        )
        self.proj2_bias = torch.nn.Parameter(
            torch.empty(hidden_size)
        )

    def forward(self, hidden_states):
        hidden_states = cube.runtime.function.complex.feedforward(
            hidden_states,
            self.proj1_weight, self.proj1_bias,
            self.proj2_weight, self.proj2_bias
        )
        return hidden_states


class TransformerLayer(torch.nn.Module):

    def __init__(self, hidden_size, head_num, dropout):
        super().__init__()
        # layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=0.00001)

        self.attention = MultiHeadSelfAttention(hidden_size, head_num, dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.ffn_layernorm = torch.nn.LayerNorm(hidden_size, eps=0.00001)
        self.ffn = FFN(hidden_size)
        self.ffn_dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states):
        # Attention
        in_attn_norm = self.input_layernorm(hidden_states)
        attn_out = self.attention(in_attn_norm)
        # residual
        attn_out = self.attn_dropout(attn_out)
        residual = attn_out + hidden_states
        # ffn
        in_ffn_norm = self.ffn_layernorm(residual)
        ffn_out = self.ffn(in_ffn_norm)
        # residual
        ffn_out = self.ffn_dropout(ffn_out)
        ffn_out = ffn_out + residual
        return ffn_out


class Transformers(torch.nn.Module):

    def __init__(self, hidden_size, head_num):
        super().__init__()

        self.transformer1 = TransformerLayer(hidden_size, head_num, 0.5)
        self.transformer2 = TransformerLayer(hidden_size, head_num, 0.5)
        self.transformer3 = TransformerLayer(hidden_size, head_num, 0.5)
        self.transformer4 = TransformerLayer(hidden_size, head_num, 0.5)

    def forward(self, hidden_states):

        hidden_states = self.transformer1(hidden_states)
        hidden_states = self.transformer2(hidden_states)
        hidden_states = self.transformer3(hidden_states)
        hidden_states = self.transformer4(hidden_states)
        loss = torch.sum(hidden_states)
        return loss


def train():
    L = 512  # seq len
    N = 8   # batch size
    # configs: [hidden size, num_head]
    # E, num_head = [2304, 24, 24]  # 1.7B model
    E, num_head, layers = [3072, 32, 30]  # 3.6B model
    # E, num_head, layers = [4096, 32, 36]  # 7.5B model


    model = Transformers(
        hidden_size=E, head_num=num_head
    )
    model = cube.SemanticModel(
        model, input_shapes=([L, N, E],),
    )

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

    iter_time = CudaTimer().duration(iter_num-40, field_name='e2e')
    throughput = N / iter_time * 1000
    print_each_rank('e2e time {:.2f} ms/iter. Throughput: {:.2f} samples/sec'.format(
          iter_time, throughput)
    )
    memory_summary()

if __name__ == '__main__':

    cube.init()
    train()