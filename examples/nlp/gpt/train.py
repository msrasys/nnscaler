"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --policy PASMegatron --fp16
"""


import torch
import time

from examples.nlp.gpt.model import GPT
from examples.nlp.gpt.model import GPTDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary

from examples.nlp.gpt.policy.mpmd import PASMegatron as PAS
import examples.nlp.gpt.policy.spmd as spmd
import examples.nlp.gpt.policy.mpmd as mpmd

import argparse

parser = argparse.ArgumentParser(description='GPT Train')
parser.add_argument('--policy', type=str, help='PAS policy choice, starting with PAS')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
args = parser.parse_args()

cube.init()

PAS = None
policies = list(spmd.__dict__.keys()) + list(mpmd.__dict__.keys())
policies = [policy for policy in policies if policy.startswith('PAS')]
if args.policy in spmd.__dict__:
    PAS = spmd.__dict__[args.policy]
    print_each_rank(f'using policy from spmd.{args.policy}')
elif args.policy in mpmd.__dict__:
    PAS = mpmd.__dict__[args.policy]
    print_each_rank(f'using policy from mpmd.{args.policy}')
else:
    raise ValueError(f"policy {args.policy} not found. Candidates: {policies}")



def train():

    batch_size = 2

    model = GPT()
    model = model if not args.fp16 else model.half()
    dataloader = GPTDataLoader(batch_size)

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=PAS, override=True, load_content=True)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        loss.backward()
    model = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')

        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)

    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()