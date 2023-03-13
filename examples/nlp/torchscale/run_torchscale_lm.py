# USE_TORCHFX=1 SINGLE_DEV_MODE=1 PYTHONPATH=.:/home/quzha/MagicCube/examples/nlp/torchscale/torchscaletest/torchscale:$PYTHONPATH python3 run_torchscale_lm.py /home/quzha/MagicCube/examples/nlp/torchscale/input --activation-fn gelu --share-decoder-input-output-embed --validate-interval-updates 1000 --save-interval-updates 1000 --no-epoch-checkpoints --memory-efficient-fp16 --fp16-init-scale 4 --arch lm_base --task language_modeling --sample-break-mode none --tokens-per-sample 128 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --clip-norm 0.0 --lr 5e-4 --lr-scheduler polynomial_decay --warmup-updates 750 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size 4 --update-freq 1 --required-batch-size-multiple 1 --total-num-update 50000 --max-update 50000 --seed 1 --ddp-backend=c10d --subln --xpos-rel-pos --fp16 --policy PASData


# USE_TORCHFX=1 SINGLE_DEV_MODE=1 PYTHONPATH=.:$PYTHONPATH:torchscaletest/torchscale python examples/nlp/torchscale/run_torchscale_lm.py  examples/nlp/torchscale/lm_input  --activation-fn gelu --share-decoder-input-output-embed --validate-interval-updates 1000 --save-interval-updates 1000 --no-epoch-checkpoints --memory-efficient-fp16 --fp16-init-scale 4 --arch lm_base --task language_modeling --sample-break-mode none --tokens-per-sample 128 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --clip-norm 0.0 --lr 5e-4 --lr-scheduler polynomial_decay --warmup-updates 750 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size 4 --update-freq 1 --required-batch-size-multiple 1 --total-num-update 50000 --max-update 50000 --seed 1 --ddp-backend=c10d --subln --xpos-rel-pos --fp16 --policy PASData

from pathlib import Path
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
sys.path.append('/home/quzha/MagicCube/examples/nlp/torchscale/torchscaletest/torchscale/examples/fairseq')
sys.path.append('/home/quzha/MagicCube/examples/nlp/torchscale/torchscaletest/torchscale')
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
with open('/home/quzha/MagicCube/examples/nlp/torchscale/input_lm', 'rb') as f:
# with open('examples/nlp/torchscale/lm_input_v2.pkl', 'rb') as f:
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


# model = cube.SemanticModel(
#      #TODO fix me model, dummy_input=sample_input_cpu,
#     # model, dummy_input=dummy_input_list,
#     model, dummy_input=dummy_input,
# )

# @cube.compile(model, dataloader, PAS=PAS, load_content=False)
# def train_iter(model, dataloader):
#     data = next(dataloader)
#     loss = model(*data)
#     # loss.backward()
#     return loss

# model = model.get_gen_module()

# iter_ret = train_iter(model, dataloader)
# print(f'iter_ret = {iter_ret}')

# Conduct concrete trace below
# sys.path.append('/home/v-junliang/torchscaletest/nni')
# sys.path.append('./torchscaletest/nni')
# from nni.common.concrete_trace_utils import concrete_trace
# from concrete_trace_utils import concrete_trace
# from examples.nlp.torchscale.concrete_trace_utils import concrete_trace
# import examples.nlp.torchscale.torchscaletest.torchscale
from cube.graph.parser.concrete_trace_utils.concrete_tracer import concrete_trace

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
traced_graph = concrete_trace(
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
    output_traced = traced_graph(**dummy_input)
assert check_equal(output_origin, output_traced), "check equal failed"
print("checked")

# check graph
traced_graph.graph.print_tabular()

print("parsing fx graph to cube graph...")
from cube.graph.parser import FxModuleParser
inputs, nodes, outputs = FxModuleParser.parse(traced_graph, dummy_inputs=dummy_input)
print("parsing done.")
from cube.graph import IRGraph
module_name = model.__class__.__name__
cube_graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
print("generating cube ir graph done.")

# AutoDist
# # profile communication cost
# import os
# comm_gpu_num = (2, 4)
# for gpu_num in comm_gpu_num:
#     os.system(f'torchrun --nproc_per_node={gpu_num} /home/quzha/AutoDist/comm_profile.py --connect_type=NV')
# profile computation cost
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
config = dotdict({'profile_dir': str(Path.home())+'/.autodist/', 'task_name': 'torchscale'})
config.autodist_config = dotdict({'ngpus': 2})
# NOTE add SINGLE_DEV_MODE=1 before the running command
from autodist.cost_model.cost_database import CostDatabase
cost_database = CostDatabase(cube_graph, config)
# find the best partition plan
from autodist.task_config import TaskConfig
class TorchscaleTaskConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = 'Bloom'
        # self.Bloom_setting = kwargs['Bloom_setting']
        # self.fine_grained_Bloom = kwargs['fine_grained_Bloom']
        # self.bloom_config = build_bloom_config(self.Bloom_setting)
        self.task_name = f'torchscale-{self.autodist_config.ngpus}gpu-'\
                        f'{self.autodist_config.micro_batch_size}batch_size'
        self.estimated_fname, self.backup_fname, self.runtime_fname = self._build_file_name(
            self.task_name)
        self.allow_recom_ops = []
        self.del_dim = []
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Torchscale benchmark')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='use fp16 for the training')
    parser.add_argument('--fine_grained_GPT',
                        action='store_true',
                        help='model = GPTFineGrained')
    parser.add_argument('--GPT_setting',
                        type=str,
                        default='6.7B',
                        help='set GPT model type')
    parser.add_argument('--save_folder',
                        type=str,
                        default='exp_data',
                        help='set the save folder for experiment data')
    parser.add_argument('--micro_batch_size',
                        type=int,
                        default=8,
                        help='set micro batch size')
    parser.add_argument('--global_batch_size',
                        type=int,
                        default=8,
                        help='set the global batch size')
    parser.add_argument('--iter_num',
                        type=int,
                        default=2,
                        help='set the number of all iterations')
    parser.add_argument('--warm_num',
                        type=int,
                        default=1,
                        help='set the number of warmup iterations')
    parser.add_argument('--recompute',
                        action='store_true',
                        help='set recompute flag')
    parser.add_argument('--memory_constraint',
                        type=float,
                        default=32,
                        help='memory constraint for program')
    parser.add_argument('--memory_granularity',
                        type=int,
                        default=1,
                        help='memory granularity in byte')
    parser.add_argument('--profile_dir',
                        type=str,
                        default=str(Path.home()) + '/.autodist',
                        help='profile dir')
    parser.add_argument('--connect_type',
                        type=str,
                        default='NV2',
                        help='connect type from nvidia-smi topo -m')
    parser.add_argument('--use_prev_plan',
                        action='store_true',
                        help='run from previous plan')
    parser.add_argument('--is_train',
                        action='store_true',
                        help='True: train, False: inference')
    parser.add_argument('--topk',
                        type=int,
                        default=20,
                        help='generate multiple plans for robustness')
    parser.add_argument('--mesh_row', type=int, default=1, help='node num')
    parser.add_argument('--mesh_col',
                        type=int,
                        default=2,
                        help='dev num in a node')
    parser.add_argument('--compile',
                        action='store_true',
                        help='compile stage: true, runtime stage: false')
    parser.add_argument('--pipeline',
                        action='store_true',
                        help='pipeline: true, tensor parallel: false')
    parser.add_argument('--nproc',
                        type=int,
                        default=12,
                        help='multiprocess deg in pipeline')
    parser.add_argument('--adaptive_recom',
                        action='store_true',
                        help='allow adaptive recompute')
    parser.add_argument('--plan_idx',
                        type=int,
                        default=0,
                        help='runtime plan idx')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--ignore_small_tensor_threshold',
                        type=int,
                        default=0,
                        help='set the tensor size threshold to ignore')
    parser.add_argument('--parse_plan',
                        action='store_true',
                        help='parse plan to user-friendly format')
    parser.add_argument('--alphafold',
                        action='store_true',
                        help='use alphafold2')
    parser.add_argument('--alphafold_setting',
                        type=int,
                        default=1,
                        help='1: bs, s, r = 1, 128, 256'\
                             '2: bs, s, r = 1, 512, 256'\
                             '3: bs, s, r = 1, 512, 384')
    args = parser.parse_args()

    # if args.compile:
    #     assert args.ignore_small_tensor_threshold >= 64, 'suggest ignore_small_tensor_threshold >= 64'

    task_config = TorchscaleTaskConfig(**vars(args))
    from autodist.apis import calc_parallel_plan
    topk_plans = calc_parallel_plan(cube_graph, cost_database, task_config)
