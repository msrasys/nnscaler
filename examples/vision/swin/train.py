"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/vision/swin/train.py --policy PASMeshShard --fp16
"""

import torch
from examples.vision.swin.model import Config, SwinTransformer, ImageDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary

import examples.vision.swin.policy.spmd as spmd
import examples.vision.swin.policy.mpmd as mpmd

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

    batch_size = 4

    cfg = Config()
    model = SwinTransformer()
    model = model.half() if args.fp16 else model

    dtype = torch.float16 if args.fp16 else torch.float32
    dataloader = ImageDataLoader(batch_size, cfg.img_size, cfg.num_classes, dtype=dtype)

    model = cube.SemanticModel(model, dataloader.shapes)
    @cube.compile(model, dataloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        imgs = next(dataloader)
        loss = model(imgs)
        loss.backward()
        # return loss
    model: torch.nn.Module = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'model parameter: {nparams}')

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 10, 2
    for step in range(iter_num):

        if step >= warmup:
            CudaTimer(enable=True).start('e2e')

        # training
        loss = train_iter(model, dataloader)
        # print(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step >= warmup:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 4 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()
