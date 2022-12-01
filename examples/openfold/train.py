"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/openfold/train.py --fp16 --tp 4 --dp 1
"""


import torch
from examples.openfold.model import AlphaFold, Config

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from examples.openfold.policy.mpmd import PASDAP

import argparse
from functools import partial


cube.init()

parser = argparse.ArgumentParser(description='AlphaFold Train')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--layers', type=int, default=4,
                    help='evoformer layer number')
parser.add_argument('--msa-hidden', type=int, default=256,
                    help='cm value')
parser.add_argument('--pair-hidden', type=int, default=128,
                    help='cz value')
parser.add_argument('--head-dim', type=int, default=32,
                    help='c value')
parser.add_argument('--mbs', type=int, default=1,
                    help='micro batch size')
parser.add_argument('--gbs', type=int, default=1,
                    help='global batch size')
parser.add_argument('--tp', type=int, default=1,
                    help='tensor parallelism size')
parser.add_argument('--dp', type=int, default=1,
                    help='data parallelism size')

args = parser.parse_args()
assert args.gbs % args.mbs == 0
assert args.mbs % args.dp == 0
assert args.msa_hidden % args.head_dim == 0
assert args.pair_hidden % args.head_dim == 0


PASDAP = partial(PASDAP, tp=args.tp, dp=args.dp)


def nparams(model) -> int:
    cnt = 0
    for param in model.parameters():
        cnt += param.nelement()
    return cnt


def train():

    cfg = Config(evoformer_cm=args.msa_hidden, evoformer_cz=args.pair_hidden,
                 evoformer_c=args.head_dim, evoformer_nlayers=args.layers,
                 bs=args.mbs)
    print_each_rank(cfg, rank_only=0)

    model = AlphaFold(cfg)
    if args.fp16:
        model = model.half()

    dtype = torch.float16 if args.fp16 else torch.float32
    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([cfg.bs, cfg.evoformer_s, cfg.evoformer_r, cfg.evoformer_cm], 
                [cfg.bs, cfg.evoformer_r, cfg.evoformer_r, cfg.evoformer_cz]),
        dtypes=(dtype, dtype),
        batch_dims=(0, 0)
    )

    print_each_rank(f'before partitioned model parameter: {nparams(model)}')

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=PASDAP, override=True, load_content=True)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        loss.backward()
    model = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    print_each_rank(f'after partitioned model parameter: {nparams(model)}')

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True, predefined=True).start('e2e')
        for _ in range(args.gbs // args.mbs):
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

train()