"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/linears.py
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from examples.policy.hybrid_parallel import transform_policy
from examples.policy.hybrid_parallel import schedule_policy

# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        loss = torch.sum(output)
        return loss


def train():
    batch_size = 128
    dim = 1024

    model = MLP(dim=dim)
    model = cube.SemanticModel(
        model, input_shapes=([batch_size, dim],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(1280, [batch_size, dim])

    @cube.compile(model, dataloader, policy=(transform_policy, schedule_policy))
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    iter_num = 128
    for step in range(iter_num):
        if step >= 10:
            CudaTimer().start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= 10:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print(f'iter [{step + 1}/{iter_num}]')

    print('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-10, field_name='e2e')))


if __name__ == '__main__':

    cube.init()
    train()