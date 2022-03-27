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

OMP_NUM_THREADS=4 torchrun --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --rdzv_id=888 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=worker0:8004 \
    examples/mlp/linears.py
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from examples.atmosphere.policy.split import PAS
import torch.nn.functional as F


# from examples.mlp.policy.col_parallel import P, A, S
# PAS = (P, A, S)

# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)


    def forward(self, data):
        a = self.linear1(data)
        paded = F.pad(a, (1, 1), "constant", 8.8)
        output = paded + 0
        # loss = torch.sum(output)
        # return loss
        return output


def train():
    batch_size = 4
    dim = 4

    model = MLP(dim=dim)
    model = cube.SemanticModel(
        model, input_shapes=([batch_size, dim],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([batch_size, dim],),
        dtypes=(torch.float32,),
        batch_dims=(0,)
    )

    @cube.compile(model, dataloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        data = next(dataloader)
        # loss = model(data)
        # loss.backward()
        output = model(data)
        return output

    model = model.get_gen_module()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    iter_num = 1
    for step in range(iter_num):
        # if step >= 40:
            # CudaTimer(enable=True).start('e2e')
        output = train_iter(model, dataloader)
        # optimizer.step()
        # optimizer.zero_grad()
        # if step >= 40:
        #     CudaTimer().stop('e2e')
        # if (step + 1) % 20 == 0:
        #     print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
        print(f'output = {output}')

    # print_each_rank('e2e time (ms) per iteration: {} ms'.format(
    #     CudaTimer().duration(iter_num - 40, field_name='e2e')))
    # CudaTimer().print_all(times=iter_num - 40)


if __name__ == '__main__':
    cube.init()
    train()