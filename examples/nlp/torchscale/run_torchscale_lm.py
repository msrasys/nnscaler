# USE_TORCHFX=1 SINGLE_DEV_MODE=1 PYTHONPATH=.:$PYTHONPATH:torchscaletest/torchscale python examples/nlp/torchscale/run_torchscale_lm.py  examples/nlp/torchscale/lm_input  --activation-fn gelu --share-decoder-input-output-embed --validate-interval-updates 1000 --save-interval-updates 1000 --no-epoch-checkpoints --memory-efficient-fp16 --fp16-init-scale 4 --arch lm_base --task language_modeling --sample-break-mode none --tokens-per-sample 128 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --clip-norm 0.0 --lr 5e-4 --lr-scheduler polynomial_decay --warmup-updates 750 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size 4 --update-freq 1 --required-batch-size-multiple 1 --total-num-update 50000 --max-update 50000 --seed 1 --ddp-backend=c10d --subln --xpos-rel-pos --fp16 --policy PASData

import torch
import pickle
from fairseq import (
    tasks,
    options,
    checkpoint_utils
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.trainer import Trainer
from fairseq.data import iterators

import sys

import os
print(f'os.getcwd() = {os.getcwd()}')


# https://github.com/microsoft/torchscale/tree/main/examples/fairseq
# sys.path.append('/home/v-junliang/torchscaletest/torchscale/examples/fairseq')
# sys.path.append('./torchscaletest/torchscale/examples/fairseq')
sys.path.append('examples/nlp/torchscale/torchscaletest/torchscale/examples/fairseq')
sys.path.append('examples/nlp/torchscale/torchscaletest/torchscale')
print(f'sys.path = {sys.path}')
import models

#:torchscaletest/torchscale
import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
sys.path.append('.')
from policy import mpmd, spmd
# import examples.nlp.torchscale.policy.spmd as spmd

# import argparse

# parser = argparse.ArgumentParser(description='comm primitive')
# parser.add_argument('--policy', type=str, help='PAS policy choice, starting with "PAS"')
# parser.add_argument('--local_rank', type=int, default=0)
# args = parser.parse_args()

# build model
parser = options.get_training_parser()
parser.add_argument('--policy', type=str, help='PAS policy choice, starting with "PAS"')
# parser.add_argument('--local_rank', type=int, default=0)

args = options.parse_args_and_arch(parser)

cube.init()
# set up policy
PAS = None
policies = list(spmd.__dict__.keys()) + list(mpmd.__dict__.keys())
if args.policy in spmd.__dict__:
    PAS = spmd.__dict__[args.policy]
    print_each_rank(f'using policy from spmd.{args.policy}')
elif args.policy in mpmd.__dict__:
    PAS = mpmd.__dict__[args.policy]
    print_each_rank(f'using policy from mpmd.{args.policy}')
else:
    raise ValueError(f"policy {args.policy} not found. Candidates: {policies}")

cfg = convert_namespace_to_omegaconf(args)
task = tasks.setup_task(cfg.task)
model = task.build_model(cfg.model)
model.eval()
print("building model succeed: ", type(model))

# create dummy input
with open('examples/nlp/torchscale/input_lm', 'rb') as f:
    dummy_input = pickle.load(f)
device = next(model.parameters()).device
print(f'device = {device}')
for key in dummy_input.keys():
    dummy_input[key] = dummy_input[key].to(device)
print(f'dummy_input <{type(dummy_input)}> = {dummy_input}')

# create input as list of tensors/objects
dummy_input_list = [val for key, val in dict(dummy_input).items()]
# print(f'dummy_input_list = {dummy_input_list}, len = {len(dummy_input_list)}')

with torch.no_grad():
    output_origin = model(**dummy_input)
    # output_origin = model(*dummy_input_list)
    # print(f'output_origin = {output_origin}')


input_shapes = [list(dummy_input[input].size()) for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)]
input_dtypes = [dummy_input[input].dtype for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)]
input_names = tuple([input for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)])

# input_shapes += [[None], [None]]
# input_dtypes += [bool, bool]

print(f'input_shapes = {input_shapes}')
print(f'input_dtypes = {input_dtypes}')

dataloader = cube.runtime.syndata.SynDataLoader(
    # names=('src_tokens',),
    shapes=(input_shapes),
    dtypes=input_dtypes,
    batch_dims=(0, 0),
)

sample_input = next(dataloader)
print(f'next(dataloader) = {sample_input}')
if isinstance(sample_input, tuple):
    sample_input_cpu = tuple([val.to(device) if isinstance(val, torch.Tensor) else val for val in sample_input])
elif isinstance(sample_input, dict):
    sample_input_cpu = sample_input
    for key in sample_input_cpu.keys():
        sample_input_cpu[key] = sample_input_cpu[key].to(device)
else:
    raise RuntimeError(f'To fix sample_input with type{type(sample_input)}')


model = cube.SemanticModel(
    model, dummy_input=dummy_input,
)

@cube.compile(model, dataloader, PAS=PAS, load_content=False, override=True)
def train_iter(model, dataloader):
    data = next(dataloader)
    loss = model(*data)
    # loss.backward()
    return loss

model = model.get_gen_module()

iter_ret = train_iter(model, dataloader)
print(f'iter_ret = {iter_ret}')

import sys
sys.exit(0)