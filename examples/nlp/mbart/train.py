"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    examples/nlp/mbart/train.py --policy PASMixPipe
"""


import torch

from examples.nlp.mbart.model import MBartForSentenceClassification, Config
from examples.nlp.mbart.model import MBartSyntheticDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary
import examples.nlp.mbart.policy.mpmd as mpmd

import argparse
import math

parser = argparse.ArgumentParser(description='GPT Train')
parser.add_argument('--policy', type=str, help='PAS policy choice, starting with PAS')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
# training
parser.add_argument('--gbs', type=int, default=1, help='global batch size')
parser.add_argument('--mbs', type=int, default=2, help='micro batch size')
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

PAS = None
policies = list(mpmd.__dict__.keys())
policies = [policy for policy in policies if policy.startswith('PAS')]
if args.policy in mpmd.__dict__:
    PAS = mpmd.__dict__[args.policy]
    print_each_rank(f'using policy from mpmd.{args.policy}')
else:
    raise ValueError(f"policy {args.policy} not found. Candidates: {policies}")


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
    Config.num_embeddings = args.vocab
    Config.layers = args.layers
    Config.hidden = args.hidden
    Config.heads = args.heads
    Config.seqlen = args.seqlen

    if cube.runtime.device.DeviceGroup().local_rank == 0:
        model = MBartForSentenceClassification(batch_size)
        model = model.half() if args.fp16 else model
    else:
        model = None
    dataloader = MBartSyntheticDataLoader(batch_size)

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=PAS, override=True, load_content=False)
    def train_iter(model, dataloader):
        input_ids = next(dataloader)
        loss = model(input_ids)
        loss.backward()
    model = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    for name, buffer in model.named_buffers():
        torch.manual_seed(0)
        if name.startswith('decoder_input_ids'):
            inputs = torch.randint(
                0, args.vocab, buffer.size(),
                dtype=torch.int64, device=torch.cuda.current_device(),
            )
            buffer.copy_(inputs)
    
    torch.manual_seed(0)
    for param in model.parameters():
        trunc_normal_(param)

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 5, 2
    for step in range(iter_num):

        if step == warmup:
            CudaTimer(enable=True).start('e2e')

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