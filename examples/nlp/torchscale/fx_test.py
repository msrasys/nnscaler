# USE_TORCHFX=1 SINGLE_DEV_MODE=1 PYTHONPATH=.:$PYTHONPATH:torchscaletest/torchscale python examples/nlp/torchscale/fx_test.py    examples/nlp/torchscale/input  --arch mt_base --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --max-tokens 4096 --fp16 --policy PASData

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
with open('examples/nlp/torchscale/input_tl', 'rb') as f:
    dummy_input = pickle.load(f)
device = next(model.parameters()).device
print(f'device = {device}')
for key in dummy_input.keys():
    dummy_input[key] = dummy_input[key].to(device)
print("creating dummy input succeed")
dummy_input['features_only'] = False
dummy_input['return_all_hiddens'] = False
print(f'dummy_input = {dummy_input}, {type(dummy_input)}')

# create input as list of tensors/objects
dummy_input_list = [val for key, val in dict(dummy_input).items()]
print(f'dummy_input_list = {dummy_input_list}')

with torch.no_grad():
    # output_origin = model(**dummy_input)
    output_origin = model(*dummy_input_list)
    # print(f'output_origin = {output_origin}')


input_shapes = [list(dummy_input[input].size()) for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)]
input_dtypes = [dummy_input[input].dtype for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)]
input_names = tuple([input for input in dummy_input if isinstance(dummy_input[input], torch.Tensor)])

input_shapes += [[None], [None]]
input_dtypes += [bool, bool]

print(f'input_shapes = {input_shapes}')
print(f'input_dtypes = {input_dtypes}')

dataloader = cube.runtime.syndata.SynDataLoader(
    shapes=(input_shapes),
    dtypes=input_dtypes,
    batch_dims=(0,0,0, None, None),
)
sample_input = next(dataloader)
print(f'next(dataloader) = {sample_input}')
sample_input_cpu = tuple([val.to(device) if isinstance(val, torch.Tensor) else val for val in sample_input])

model = cube.SemanticModel(
     model, dummy_input=sample_input_cpu,
)

@cube.compile(model, dataloader, PAS=PAS, load_content=False)
def train_iter(model, dataloader):
    data = next(dataloader)
    loss = model(*data)
    loss.backward()

train_iter(model, dataloader)

# Conduct concrete trace below
# sys.path.append('/home/v-junliang/torchscaletest/nni')
# sys.path.append('./torchscaletest/nni')
# from nni.common.concrete_trace_utils import concrete_trace
# from concrete_trace_utils import concrete_trace
from examples.nlp.torchscale.concrete_trace_utils import concrete_trace
import examples.nlp.torchscale.torchscaletest.torchscale


def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return a == b


print("start tracing...")
traced_model, _ = concrete_trace(
    model,
    dummy_input,
    use_operator_patch=True,
    autowrap_leaf_class={
        torch.finfo: ((), False),
        type(output_origin): ((), False),
    },
)
print("trace succeed")
print("checking equal...")
with torch.no_grad():
    output_traced = traced_model(**dummy_input)
assert check_equal(output_origin, output_traced), "check equal failed"
print("checked")

# check graph
traced_model.graph.print_tabular()

# with open('input_tl', 'wb') as f:
#     pickle.dump(dummy_input, f)

# try to save traced model with pickle
# from concrete_trace_utils.concrete_tracer import MagicMethodPatcher
# from pickle import _Pickler, _Unpickler

# with open("save/through_nn_Module/tl_traced_v2.model", "wb") as f:
#     # pickle.dump(traced_model, f)
#     with MagicMethodPatcher():
#         _Pickler(f).dump(traced_model)

# with open("save/through_nn_Module/tl_traced.model", "rb") as f:
#     with MagicMethodPatcher():
#         reload_model = _Unpickler(f).load()


# with torch.no_grad():
#     output_reload = reload_model(**dummy_input)
# assert check_equal(output_origin, output_reload), "reload check equal failed"
# print("reload is good!")

# with open("save/through_nn_Module/tl_origin_v2.model", "wb") as f:
#     with MagicMethodPatcher():
#         _Pickler(f).dump(model)

# with open("save/through_nn_Module/tl_input_v2.pkl", "wb") as f:
#     with MagicMethodPatcher():
#         _Pickler(f).dump(dummy_input)

