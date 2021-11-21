"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/ffn/ffn.py
"""

import torch
import torch.nn.functional as F
import cube

from examples.ffn.policy.data_parallel import transform_policy
from examples.ffn.policy.data_parallel import schedule_policy

from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


class FFN(torch.nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(
            hidden_size, 4 * hidden_size
        )
        self.dense_4h_to_h = torch.nn.Linear(
            4 * hidden_size, hidden_size
        )

    def forward(self, hidden_states):
        # [L, N, E] * [E, 4E] -> [L, N, 4E]
        out = self.dense_h_to_4h(hidden_states)
        # [L, N, 4E] -> [L, N, 4E]
        out = F.gelu(out)
        # [L, N, 4E] * [4E, E] -> [L, N, E]
        out = self.dense_4h_to_h(out)

        loss = torch.sum(out)
        return loss


def train():
    L = 512  # seq len
    N = 32   # batch size
    # configs: [hidden size, num_head]
    # E, num_head = [1536, 16]  # 1.2B model
    # E, num_head = [1920, 20]  # 2.5B model
    # E, num_head = [2304, 24]  # 4.2B model
    E, num_head = [3072, 32]  # 8.7B model


    model = FFN(hidden_size=E)
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
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))


if __name__ == '__main__':

    cube.init()
    train()
