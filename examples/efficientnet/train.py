"""
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/efficientnet/train.py --bs 1
"""

import torch
from torch import nn
from examples.efficientnet.efficientnet import EfficientNet
import time
import argparse

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup
from cube.runtime.reducer import Reducer



def train(args):

    N = args.bs

    # L2 config
    # C, H, W = [3, 800, 800]
    # model = EfficientNet.from_name('efficientnet-l2')
    
    # B8 config
    C, H, W = [3, 672, 672]
    model = EfficientNet.from_name('efficientnet-b8')

    model = model.cuda()

    if N % args.bs != 0:
        raise RuntimeError("global bs is not divisible by DP")
    dataloader = cube.runtime.syndata.SynDataLoader(
        1280, [0], [N // args.dp, C, H, W])


    def train_iter(model, dataloader):
        img = next(dataloader)
        loss = model(img)
        loss = torch.sum(loss)
        loss.backward()

    optimizer = torch.optim.RMSprop(model.parameters())

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    span = 0
    iter_num = 128
    for step in range(iter_num):
        if step >= 40:
            torch.cuda.synchronize()
            start = time.time()
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step == 1:
            print('> passed on 1st iteration')
            memory_summary()
        if step >= 40:
            torch.cuda.synchronize()
            stop = time.time()
            span += (stop - start) * 1000
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)


if __name__ == '__main__':

    cube.init()
    
    # resource allocation
    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--bs', type=int, default=1,
                        help='bs')
    parser.add_argument('--dp', type=int, default=1,
                        help='data parallel')
    parser.add_argument('--fp16', action='store_true', dest='fp16')
    args = parser.parse_args()

    train(args)
