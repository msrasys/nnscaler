"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    benchmark/megatron/gpt.py
"""

import torch
import torch.nn.functional as F
import cube
from benchmark.megatron.layers import ColumnOutputAdapter, ShardEmbedding
from benchmark.megatron.transformer import TransformerLayer


from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank


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
        self.vocab_embedding = ShardEmbedding(self.vocab_size, hidden_size)
        self.seqlen_size = seqlen_size
        self.pos_embed_weight = torch.nn.Parameter(
            torch.empty(seqlen_size, hidden_size)
        )
        
        self.embed_dropout = torch.nn.Dropout(0.5)

        self.transform1 = TransformerLayer(seqlen, hidden_size, num_head, 0.5)
        self.transform2 = TransformerLayer(seqlen, hidden_size, num_head, 0.5)
        self.transform3 = TransformerLayer(seqlen, hidden_size, num_head, 0.5)
        self.transform4 = TransformerLayer(seqlen, hidden_size, num_head, 0.5)

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
        words_embeddings = self.vocab_embedding(input_ids)
        
        # [bs, seqlen] -> [bs, seqlen, hidden size]
        position_embeddings = cube.runtime.function.embedding(
            position_ids, self.pos_embed_weight, 0, self.seqlen_size
        )
        embeddings = words_embeddings + position_embeddings
        encoder_input = self.embed_dropout(embeddings)

        # [bs, seqlen, hidden size] -> [seqlen, bs, hidden size]
        hidden_states = encoder_input.transpose(0, 1)

        hidden_states = self.transform1(hidden_states)
        hidden_states = self.transform2(hidden_states)
        hidden_states = self.transform3(hidden_states)
        hidden_states = self.transform4(hidden_states)
        
        hidden_states = self.final_layernorm(hidden_states)

        # post process
        hidden_states = hidden_states.transpose(0, 1) # .contiguous()
        logits = F.linear(hidden_states, self.vocab_embedding.weight)
        # all gather
        logits = ColumnOutputAdapter.apply(logits)

        # loss # for verification, the mask is ommitted
        # [bs, seqlen, self.vocab_size] -> [1]
        loss = torch.sum(logits) 
        # loss = loss * self.ntoken
        return loss

def train():
    L = 512  # seq len
    N = 8   # batch size
    # configs: [hidden size, num_head]
    # E, num_head = [2304, 24, 24]  # 1.7B model
    E, num_head, layers = [3072, 32, 30]  # 3.6B model
    # E, num_head, layers = [4096, 32, 36]  # 7.5B model

    print_each_rank('config: L={}, N={}, E={}, num-head={}'.format(
        L, N, E, num_head
    ))


    model = GPT(
        hidden_size=E, vocab_size=50304, seqlen_size=L,
        bs=N, seqlen=L, num_head=num_head, num_layers=layers
    ).cuda()

    dataloader = cube.runtime.syndata.SynTextDataLoader(1280, [0, 0], [N, L], [N, L])

    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        torch.distributed.broadcast(input_ids, 0)
        torch.distributed.broadcast(position_ids, 0)
        torch.cuda.synchronize()
        loss = model(input_ids, position_ids)
        loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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