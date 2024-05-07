"""
example:

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/vision/swin/train.py --policy PASMegatronTP --fp16
"""

import math
import torch
from functools import partial
from examples.vision.swin.blocks.attention import init_relative_position_index
from examples.vision.swin.model import Config, SwinTransformer, dummy_data

import nnscaler
from nnscaler.compiler import compile
from nnscaler.profiler.timer import CudaTimer, print_each_rank
from nnscaler.profiler.memory import memory_summary
from nnscaler.runtime.utils import microbatches

import examples.vision.swin.policy.gallery as gallery
from examples.utils import get_policy

import argparse

parser = argparse.ArgumentParser(description='GPT Train')
parser.add_argument('--policy', type=str, help='PAS policy choice, starting with PAS')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--dp', type=int, default=1,
                    help='data parallel size, only for megatron')
parser.add_argument('--tp', type=int, default=1,
                    help='tensor parallel size, only for megatron')
# training
parser.add_argument('--gbs', type=int, default=4, help='global batch size')
parser.add_argument('--mbs', type=int, default=4, help='micro batch size')

args = parser.parse_args()
nnscaler.init()


# get policy
policy = get_policy([gallery], args.policy)
policy = partial(policy,
    nmicros=args.gbs//args.mbs,
    dp_size=args.dp,
    tp_size=args.tp
)


def train():

    batch_size = args.mbs
    load_content: bool = False

    cfg = Config()
    model = SwinTransformer()
    model = model.half() if args.fp16 else model

    dtype = torch.float16 if args.fp16 else torch.float32


    gen_data = partial(dummy_data, args.mbs, torch.float16, cfg)
    dataloader = microbatches((gen_data(),))

    @compile(model, dataloader, PAS=policy, load_content=load_content)
    def train_iter(model, dataloader):
        imgs = next(dataloader)
        loss = model(imgs)
        loss.backward()
    model = nnscaler.utils.load_model()

    if not load_content:
        for name, buffer in model.named_buffers():
            if 'rp_index' in name:
                window_size = int(math.sqrt(buffer.size(0)))
                buffer.copy_(init_relative_position_index(window_size).cuda())

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'model parameter: {nparams}')

    CudaTimer().warmup()
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')

        # collect data
        samples = [gen_data() for _ in range(args.gbs // args.mbs)]
        dataloader = microbatches(samples, dtype=dtype)
        # train iteration
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

    train()
