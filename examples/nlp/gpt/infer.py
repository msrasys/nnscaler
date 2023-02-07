"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/nlp/gpt/infer.py --policy PASMeshShard --fp16

PYTHONPATH=.:..:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1  examples/nlp/gpt/infer.py --policy PASSingle --fp16

PYTHONPATH=.:..:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=2  examples/nlp/gpt/infer.py --policy PASMegatronInferTP --fp16
"""


import torch

from examples.nlp.gpt.model import GPTInfer, GPTInferDataLoader
from examples.nlp.gpt.model import GPTDataLoader
from examples.nlp.gpt.model import build_gpt_config

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
parser.add_argument('--local_rank', type=int, default=0)
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

def inter():
    print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

    batch_size = 8

    model = GPTInfer(batch_size=batch_size, cfg=build_gpt_config('350M'))
    model = model if not args.fp16 else model.half()
    # model = model.cuda() #only for PyTorch run
    model.eval()
    dataloader = GPTInferDataLoader(batch_size)

    ################## SuperScaler run
    model = cube.SemanticModel(model, dataloader.shapes)
    @cube.compile(model, dataloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        # return loss
    model = model.get_gen_module()

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num = 4
    warmup = 2
    for step in range(iter_num):
        # if step == 0:
        #     model_summary(model, next(dataloader))

        if step >= warmup:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        if step >= warmup:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num - warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num - warmup)
    memory_summary()

    # iter_num = 2
    # for step in range(iter_num):
    #     output = train_iter(model, dataloader)
    #     print(f'output = {output}')

    ################## PyTorch run
    # output = None
    # for i in range(10):
    #     input_ids, position_ids = next(dataloader)
    #     print(f'input_ids = {input_ids} [{input_ids.size()}], position_ids = {position_ids} [{position_ids.size()}]')
    #     output = model(input_ids, position_ids)
    #     print(f'output = {output}')


if __name__ == '__main__':

    cube.init()
    inter()