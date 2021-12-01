"""
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/efficientnet/train.py \
        --pp 8 --gbs 8 --mbs 1
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
from examples.efficientnet.schedule import is_last_stage, scheduling_1f1b


def model_partition(model, in_size):
    # pipeline stage
    pp_rank = torch.distributed.get_rank()
    pp_size = torch.distributed.get_world_size()

    layers = model._blocks

    chunk = len(layers) // pp_size
    if len(layers) % pp_size != 0:
        remain = len(layers) % pp_size
        if pp_rank < remain:
            start = pp_rank * (chunk+1)
            chunk = chunk + 1
        else:
            start = remain * (chunk + 1) + (pp_rank - remain) * chunk
    else:
        start = pp_rank * chunk
    stop = start + chunk

    print_each_rank(f'layer start -> end: {start} -> {stop}')
    layers = layers[start:stop]
    model._blocks = layers

    if pp_rank == 0:
        model.preprocess = True
        model.in_size = in_size
    else:
        model.preprocess = False
        model.in_size = layers[0].in_size

    if is_last_stage():
        model.postprocess = True
        model.out_size = [1,]
    else:
        model.postprocess = False
        model.out_size = layers[-1].out_size

    return model


def train(args):

    N = args.gbs

    # L2 config
    C, H, W = [3, 800, 800]
    model = EfficientNet.from_name('efficientnet-l2')
    
    # B8 config
    # C, H, W = [3, 672, 672]
    # model = EfficientNet.from_name('efficientnet-b8')

    model = model_partition(model, [C, H, W])
    if args.fp16:
        model == model.half()
    model = model.cuda()

    nparams_million = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    print_each_rank('model has {:.2f} million parameters'.format(nparams_million))
    memory_summary()

    if N % args.gbs != 0:
        raise RuntimeError("global bs is not divisible by DP")
    dataloader = cube.runtime.syndata.SynDataLoader(
        1280, [0], [N // args.dp, C, H, W])
    
    if args.fp16:
        data_buff = [[e.half() for e in data] for data in dataloader.datas]
        dataloader.datas = data_buff

    def train_iter(model, dataloader):
        img = next(dataloader)
        scheduling_1f1b(model, [img], args.gbs, args.mbs, dtype=torch.float)

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

    iter_time = CudaTimer().duration(iter_num-10, field_name='e2e')
    throughput = N / iter_time * 1000
    print_each_rank('e2e time {:.2f} ms/iter. Throughput: {:.2f} samples/sec'.format(
          iter_time, throughput)
    )

    CudaTimer().print_all(times=iter_num-10)
    memory_summary()


if __name__ == '__main__':

    cube.init()
    
    # resource allocation
    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--tp', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--dp', type=int, default=1,
                        help='data parallel size')
    parser.add_argument('--pp', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--gbs', type=int, default=-1)
    parser.add_argument('--mbs', type=int, default=-1)
    parser.add_argument('--fp16', action='store_true', dest='fp16')
    args = parser.parse_args()


    resource = cube.runtime.resource.EnvResource()
    ndevs = resource.ngpus

    tp_size, tp_group_nums = args.tp, ndevs // args.tp
    dp_size, dp_group_nums = args.dp, ndevs // args.dp
    pp_size, pp_group_nums = args.pp, ndevs // args.pp
    
    if not pp_size * dp_size * tp_size == ndevs:
        raise RuntimeError("Expected all devices are used")

    devs = cube.runtime.device.DeviceGroup()

    myrank = torch.distributed.get_rank()

    # initialize data parallel group
    all_data_parallel_group_ranks = list()
    for i in range(pp_size):
        start_rank = i * pp_group_nums
        end_rank = (i + 1) * pp_group_nums
        for j in range(tp_size):
            ranks = list(range(start_rank + j, end_rank, tp_size))
            all_data_parallel_group_ranks.append(ranks)
            # initialize groups
            group = devs.get_group(ranks)
            if myrank in ranks:
                dp_ranks = ranks
                resource.dp_group = group
                resource.reducer = cube.runtime.reducer.Reducer(ranks)
    print_each_rank(f'initialzed data parallel group: {dp_ranks}', rank_only=myrank)

    # initialize pipelne parallel groups
    for i in range(dp_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = devs.get_group(ranks)
        if myrank in ranks:
            pp_ranks = ranks
            resource.pp_group = group
    print_each_rank(f'initialzed pipeline parallel group: {pp_ranks}', rank_only=myrank)

    # initialize tensor parallel groups
    for i in range(tp_group_nums):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = devs.get_group(ranks)
        if myrank in ranks:
            tp_ranks = ranks
            resource.tp_group = group
    print_each_rank(f'initialzed tensor parallel group: {tp_ranks}', rank_only=myrank)

    train(args)
