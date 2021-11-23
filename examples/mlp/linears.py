"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/mlp/linears.py
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from examples.mlp.policy.pipe1f1b_parallel import transform_policy
from examples.mlp.policy.pipe1f1b_parallel import schedule_policy

# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)
        self.linear5 = nn.Linear(dim, dim * mult)
        self.linear6 = nn.Linear(dim * mult, dim)
        self.linear7 = nn.Linear(dim, dim * mult)
        self.linear8 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        output = self.linear5(output)
        output = self.linear6(output)
        output = self.linear7(output)
        output = self.linear8(output)
        loss = torch.sum(output)
        return loss


def train():
    batch_size = 128
    dim = 1024

    model = MLP(dim=dim)
    model = cube.SemanticModel(
        model, input_shapes=([batch_size, dim],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(1280, [0], [batch_size, dim])

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