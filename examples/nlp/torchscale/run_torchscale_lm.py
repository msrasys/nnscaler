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

# https://github.com/microsoft/torchscale/tree/main/examples/fairseq
# sys.path.append('/home/quzha/quzha/MagicCube/examples/nlp/torchscale/torchscaletest/torchscale/examples/fairseq')
# sys.path.append('/home/quzha/quzha/MagicCube/examples/nlp/torchscale/torchscaletest/torchscale')
sys.path.append('/home/quzha/quzha/torchscale/examples/fairseq')
print(f'sys.path = {sys.path}')
import models

import cube

bs, seql, ngpu = 2, 2048, 4

# # build model
parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
cfg = convert_namespace_to_omegaconf(args)
task = tasks.setup_task(cfg.task)
model = task.build_model(cfg.model)
model.eval()
print("building model succeed: ", type(model))

# create dummy input
# with open('/home/quzha/quzha/MagicCube/examples/nlp/torchscale/input_lm', 'rb') as f:
with open(f'/home/quzha/torchscale_{bs}_{seql}.pkl', 'rb') as f:
    dummy_input = pickle.load(f)
dummy_input = dummy_input['net_input']
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

# move simple type inputs to kwargs
# for node in cube_graph.nodes

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

config = dotdict({'profile_dir': str(Path.home())+'/.autodist/', 'task_name': f'torchscale_{bs}_{seql}_{ngpu}'})
config.autodist_config = dotdict({'ngpus': ngpu})
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

kwargs = {'consider_mem': False, 'save_folder': 'exp_data', 'micro_batch_size': bs, 'global_batch_size': bs, 'iter_num': 2,
          'warm_num': 1, 'recompute': False, 'memory_constraint': 40, 'memory_granularity': 1,
          'profile_dir': str(Path.home())+'/.autodist/', 'connect_type': 'NV2', 'use_prev_plan': False,
          'is_train': True, 'topk': 20, 'mesh_row': 1, 'mesh_col': ngpu, 'compile': True, 'pipeline': False,
          'nproc': 12, 'adaptive_recom': False, 'plan_idx': 0, 'verbose': True, 'ignore_small_tensor_threshold': 0,
          'parse_plan': True}

task_config = TorchscaleTaskConfig(**kwargs)
from autodist.apis import calc_parallel_plan
topk_plans = calc_parallel_plan(cube_graph, cost_database, task_config)

best_plan = topk_plans[0][0].partition_descs

from cube.ir.operator import IRDataOperation, IRFwOperation, IRBpOperation

for node in cube_graph.select(ntype=IRFwOperation):
    if node.cid in best_plan:
        print(f'{node}, {node.anno}, autodist ret: {best_plan[node.cid]}')
    else:
        print(f'{node}, switch to default replica')