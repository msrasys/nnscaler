from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import json

def convert_mem_into_GB(mem):
    if type(mem) in [int, float]:
        return mem / 1024 / 1024 / 1024
    else:
        return [x / 1024 / 1024 / 1024 for x in mem]

model_name = "bigscience/bloom-560m"
model_path = "/home/quzha/bloom560m"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, cache_dir=model_path)
print(type(model), '; is nn.Module? ', isinstance(model, nn.Module))
print("Model's generation config which does not list default values: ", model.generation_config)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("Loading Done!")
prompt = "If I want to travel to a new city, I should plan my trip as follows:"
#input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
inputs = tokenizer(prompt, return_tensors="pt")

# Cube
# from cube.graph import parser
# ir_graph = parser.convert_model(model, input_shapes=[1, 17], save_content=False)

print("concrete tracing model...")
import sys
nni_path = "/home/quzha/yizhu1/yizhu1_autodist/nni/"
sys.path.append(nni_path)

# from concrete_trace_utils import concrete_trace
from nni.common.concrete_trace_utils import concrete_trace
# from cube.graph.parser.concrete_trace_utils import concrete_trace

traced_graph = concrete_trace(model, inputs, use_operator_patch=True,
        autowrap_leaf_class={torch.finfo: ((), False)})
print("tracing model done.")
# print(traced_graph)

print("parsing fx graph to cube graph...")
from cube.graph.parser import FxModuleParser
inputs, nodes, outputs = FxModuleParser.parse(traced_graph, dummy_inputs=inputs)
print("parsing done.")
from cube.graph import IRGraph
module_name = model.__class__.__name__
cube_graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)

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
config = dotdict({'profile_dir': str(Path.home())+'/.autodist/', 'task_name': 'bloom'})
config.autodist_config = dotdict({'ngpus': 2})
# NOTE add SINGLE_DEV_MODE=1 before the running command
from autodist.cost_model.cost_database import CostDatabase
cost_database = CostDatabase(cube_graph, config)
# find the best partition plan
from autodist.task_config import TaskConfig
class BloomTaskConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = 'Bloom'
        # self.Bloom_setting = kwargs['Bloom_setting']
        # self.fine_grained_Bloom = kwargs['fine_grained_Bloom']
        # self.bloom_config = build_bloom_config(self.Bloom_setting)
        self.task_name = f'bloom-{self.autodist_config.ngpus}gpu-'\
                        f'{self.autodist_config.micro_batch_size}batch_size'
        self.estimated_fname, self.backup_fname, self.runtime_fname = self._build_file_name(
            self.task_name)
        self.allow_recom_ops = []
        self.del_dim = []
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bloom benchmark')
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

    task_config = BloomTaskConfig(**vars(args))
    from autodist.apis import calc_parallel_plan
    compile_start_time = time.time()
    topk_plans = calc_parallel_plan(cube_graph, cost_database, task_config)
    compile_cost_time = time.time() - compile_start_time
    plan_info = []
    for plan in topk_plans:
        cur_info = {}
        if task_config.pipeline:
            cur_spmd_descs, cur_time, cur_mems, cur_devs, cur_times = plan
            cur_info['plan'] = []
            for item in cur_spmd_descs:
                cur_info['plan'].append(item.to_json_object())
            cur_info['estimated time'] = cur_time
            cur_info['estimated memory'] = convert_mem_into_GB(cur_mems)
            cur_info['estimated time list'] = cur_times
            cur_info['compile time'] = compile_cost_time
            plan_info.append(cur_info)
        else:
            cur_spmd_desc, cur_mem, cur_time = plan
            cur_info['plan'] = cur_spmd_desc.to_json_object()
            cur_info['estimated time'] = cur_time
            cur_info['estimated memory'] = convert_mem_into_GB(cur_mem)
            cur_info['compile time'] = compile_cost_time
            plan_info.append(cur_info)

    with open(task_config.backup_fname, 'w') as f:
        json.dump(plan_info, f)
