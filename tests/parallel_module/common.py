from datetime import datetime
import math
import random
import shutil
from typing import List, Optional
import contextlib

import torch
from torch import nn
import numpy as np

from cube.parallel import ComputeConfig
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    sub_nodes = graph.partition(
        node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _replica(graph: IRGraph, node, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def PASRandomSPMD(graph: IRGraph, env_resource: ComputeConfig):
    """
    Random SPMD policy
    """
    ngpus = env_resource.plan_ngpus
    # get the current random state
    state = random.getstate()

    seed = 1
    # print(f'> set random SPDM policy seed to {seed}')
    random.seed(seed)
    devs = list(range(ngpus))

    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0:
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                 key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0: continue
                    if node.algorithms('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                        # print(f'> partition node {node.name} ({node.cid}) with config idx={idx}, dim={dim}')
                        _tp(graph, node, devs, idx, dim)
                        break
                else:
                    _replica(graph, node, devs)
        else:
            _replica(graph, node, devs)

    # restore the random state
    random.setstate(state)
    # print(graph.extra_repr())
    return graph


def PASData(graph: IRGraph, env_resource: ComputeConfig):
    """
    Data Parallel
    """
    ngpus = env_resource.plan_ngpus
    # auto multi-ref
    for ftensor in graph.full_tensors():
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

    batch_dim = 0
    for dl in graph.select(ntype=IRDataOperation):
        _replica(dl, list(range(ngpus)))

    graph_inputs = IRSegment.get_objects_from_complex(graph.inputs())
    graph_outputs = IRSegment.get_objects_from_complex(graph.outputs())
    for node in graph.nodes():
        # print(node)
        if isinstance(node, IRFwOperation):
            # Currently cube only support replicate if node's input or input is part of the graph output
            # workaround for now
            # will fix later.
            if any(output in graph_outputs for output in node.outputs()) \
                or any(input in graph_outputs for input in node.inputs()):
                # or any(input in graph_inputs for input in node.inputs()):
                sub_nodes = graph.replicate(node, ngpus)
            else:
                try:
                    algo = node.algorithms('dim')
                    idx = 0
                    sub_nodes = graph.partition(
                        node, algo, idx=idx, dim=batch_dim, num=ngpus)
                # except AssertionError:
                except:
                    # print(f'WARNING: {node} cannot find dim algo, using replicate instead')
                    sub_nodes = graph.replicate(node, ngpus)

            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    # print(graph.extra_repr())
    return graph


class CubeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.fc(x)
        if self.bias is not None:
            x = x + self.bias
        return x


def init_distributed():
    torch.distributed.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    torch.set_default_device(f'cuda:{rank}')


def init_random():
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)



@contextlib.contextmanager
def clear_dir_on_rank0(tempdir):
    if torch.distributed.get_rank() == 0 and tempdir.exists():
        shutil.rmtree(tempdir)
    yield tempdir
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0 and tempdir.exists():
        shutil.rmtree(tempdir)
