"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/transformer/transformer.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import cube


from examples.gpt.policy.megatron_parallel import transform_policy
from examples.gpt.policy.megatron_parallel import schedule_policy

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

    def __init__(self, hidden_size, num_head, dropout):
        super().__init__()
        # layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=0.00001)

        self.attention = MultiHeadSelfAttention(hidden_size, num_head, dropout)
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
        # residual = attn_out + hidden_states
        residual = attn_out * 2
        # ffn
        in_ffn_norm = self.ffn_layernorm(residual)
        ffn_out = self.ffn(in_ffn_norm)
        # residual
        ffn_out = self.ffn_dropout(ffn_out)
        # ffn_out = ffn_out + residual
        ffn_out = ffn_out * 2
        return ffn_out
        

class GPT(torch.nn.Module):

    def __init__(self, hidden_size, vocab_size, seqlen_size,
                 bs, seqlen, num_head, num_layers: int):
        super().__init__()

        self.num_layers = num_layers
        self.bs = bs
        self.seqlen = seqlen
        self.ntoken = 1.0 / self.bs * self.seqlen

        # embeddings
        self.vocab_size = vocab_size
        self.vocab_embed_weight = torch.nn.Parameter(
            torch.empty(vocab_size, hidden_size)
        )
        self.seqlen_size = seqlen_size
        self.pos_embed_weight = torch.nn.Parameter(
            torch.empty(seqlen_size, hidden_size)
        )
        
        self.embed_dropout = torch.nn.Dropout(0.5)

        # transformer layers
        # self.layers = torch.nn.ModuleList(
        #     [TransformerLayer(seqlen, hidden_size, num_head, 0.5) for _ in range(num_layers)]
        # )
        self.transform1 = TransformerLayer(hidden_size, num_head, 0.5)
        self.transform2 = TransformerLayer(hidden_size, num_head, 0.5)
        self.transform3 = TransformerLayer(hidden_size, num_head, 0.5)
        self.transform4 = TransformerLayer(hidden_size, num_head, 0.5)

        # final linear
        self.final_layernorm = torch.nn.LayerNorm(
            hidden_size, 1e-5
        )

    def forward(self, input_ids, position_ids):
        """
        input_ids:
            [bs, seqlen]
        position_ids:
            [bs, seqlen]
        """

        # preprocess: embedding
        # [bs, seqlen] -> [bs, seqlen, hidden size]
        words_embeddings = cube.runtime.function.embedding(
            input_ids, self.vocab_embed_weight, 0, self.vocab_size
        )
        # [bs, seqlen] -> [bs, seqlen, hidden size]
        position_embeddings = cube.runtime.function.embedding(
            position_ids, self.pos_embed_weight, 0, self.seqlen_size
        )
        embeddings = words_embeddings + position_embeddings
        encoder_input = self.embed_dropout(embeddings)

        # [bs, seqlen, hidden size] -> [seqlen, bs, hidden size]
        hidden_states = encoder_input.transpose(0, 1) #.contiguous()

        # transformer
        # [seqlen, bs, hidden size] -> [seqlen, bs, hidden size]
        # for layer in self.layers:
        #     hidden_states = layer(hidden_states)
        hidden_states = self.transform1(hidden_states)
        hidden_states = self.transform2(hidden_states)
        hidden_states = self.transform3(hidden_states)
        hidden_states = self.transform4(hidden_states)
        
        hidden_states = self.final_layernorm(hidden_states)

        # post process
        # [seqlen, bs, hidden size] -> [bs, seqlen, hidden size]
        hidden_states = hidden_states.transpose(0, 1) # .contiguous()
        # [bs, seqlen, hidden size] * [self.vocab_size, hidden size]
        # => [bs, seqlen, self.vocab_size]
        logits = F.linear(hidden_states, self.vocab_embed_weight)

        # loss # for verification, the mask is ommitted
        # [bs, seqlen, self.vocab_size] -> [1]
        loss = torch.sum(logits) 
        # loss = loss * self.ntoken

        return loss


def train():
    L = 512  # seq len
    N = 4   # batch size
    # configs: [hidden size, num_head]
    # E, num_head = [1536, 16]  # 1.2B model
    # E, num_head = [1920, 20]  # 2.5B model
    # E, num_head = [2304, 24]  # 4.2B model
    E, num_head = [3072, 32]  # 8.7B model
    layers = 4


    model = GPT(
        hidden_size=E, vocab_size=50304, seqlen_size=L,
        bs=N, seqlen=L, num_head=num_head, num_layers=layers
    )
    model = cube.SemanticModel(
        model, input_shapes=([N, L], [N, L],),
    )

    dataloader = cube.runtime.syndata.SynTextDataLoader(1280, [0, 0], [N, L], [N, L])

    @cube.compile(model, dataloader, policy=(transform_policy, schedule_policy))
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
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
    
    memory_summary()
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))


if __name__ == '__main__':

    cube.init()
    train()
