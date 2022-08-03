"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/train.py
"""


import torch

from examples.nlp.gpt.model import GPT
from examples.nlp.gpt.model import GPTDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary

from examples.nlp.gpt.policy.mpmd import PASRoundRobin as PAS


def train():

    batch_size = 1

    model = GPT()
    dataloader = GPTDataLoader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    model = cube.SemanticModel(model, dataloader.shapes)
    @cube.compile(model, dataloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        loss.backward()
    model = model.get_gen_module()

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num = 40
    warmup = 8
    for step in range(iter_num):
        # if step == 0:
        #     model_summary(model, next(dataloader))

        if step >= warmup:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= warmup:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()