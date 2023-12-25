"""
example:

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/nlp/mbart/train.py --policy PASMegatronTP --fp16
"""


import torch
import logging
import argparse
import math
from functools import partial

from examples.nlp.mbart.model import MBartForSentenceClassification, Config
from examples.nlp.mbart.model import dummy_data

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.utils import microbatches

import examples.nlp.mbart.policy.gallery as gallery

from examples.utils import get_policy

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
# arch
parser.add_argument('--vocab', type=int, default=2500,
                    help='used vocabulary size')
parser.add_argument('--layers', type=int, default=8,
                    help='layer number of each encoder and decoder')
parser.add_argument('--heads', type=int, default=16,
                    help='head number')
parser.add_argument('--hidden', type=int, default=2048,
                    help='head number')
parser.add_argument('--seqlen', type=int, default=1024,
                    help='sequence length')

args = parser.parse_args()

cube.init()
print(args)


cube.init()
cube.set_logger_level(logging.WARN)
logging.getLogger('cube.compiler').setLevel(logging.INFO)

# get policy
policy = get_policy([gallery], args.policy)
policy = partial(policy, 
    nmicros=args.gbs//args.mbs, 
    dp_size=args.dp,
    tp_size=args.tp
)


def trunc_normal_(tensor: torch.Tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        # tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def train():

    batch_size = args.mbs

    config = Config(
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        seqlen=args.seqlen,
        ffn_hidden_dim=args.hidden * 4,
        vocab=args.vocab,
    )
    print_each_rank(config)

    model = MBartForSentenceClassification(batch_size, config)
    torch.manual_seed(0)
    for param in model.parameters():
        trunc_normal_(param)
    model = model.half() if args.fp16 else model

    gen_data = partial(dummy_data, batch_size, config)
    dataloader = microbatches((gen_data(),), cycle=True)

    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        input_ids, decoder_input_ids = next(dataloader)
        loss = model(input_ids, decoder_input_ids)
        loss.backward()
    model = cube.load_model()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    CudaTimer().warmup()
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')
        # prepare input data
        samples = [gen_data() for _ in range(args.gbs // args.mbs)]
        dataloader = microbatches(samples)

        # training
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