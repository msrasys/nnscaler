"""
pytest unit_tests/compiler/test_compile.py
"""
import torch
import logging
from functools import partial

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation
from cube.flags import CompileFlag
from ..launch_torchrun import torchrun
from ..utils import init_parameter, assert_parity


class MLP(torch.nn.Module):
    def __init__(self, dim=512, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def get_dummy_data(batch_size: int = 512):
    torch.random.manual_seed(0)
    return torch.randn(
        [128, 512], dtype=torch.float32, 
        device=torch.cuda.current_device()).repeat([batch_size // 128, 1])


def baseline():

    model = MLP()
    init_parameter(model)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = model(x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0  # scale for comparison
        losses.append(loss)

    return losses


def scale(ngpus_per_unit: int):

    model = MLP()
    init_parameter(model)
    
    def policy(graph: IRGraph, resource):

        ngpus = min(ngpus_per_unit, resource.ngpus)

        def tensor_parallelism(node, idx, dim, num):
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=idx, dim=dim, num=num)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            return sub_nodes

        l1, l2, l3, l4 = graph.select(name='linear')

        # l1 tensor parallelism
        tensor_parallelism(l1, idx=1, dim=0, num=ngpus)
        # l2 data parallelism
        tensor_parallelism(l2, idx=0, dim=0, num=ngpus)
        # l3 tensor parallelism
        tensor_parallelism(l3, idx=1, dim=1, num=ngpus)
        # l4 replicate

        for node in graph.select(ntype=IRFwOperation):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, times=ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
        return graph
    
    ngpus_per_unit = min(ngpus_per_unit, torch.distributed.get_world_size())
    nreplicas = torch.distributed.get_world_size() // ngpus_per_unit
    batch_size = 512 // nreplicas
    print('>> set batch size to', batch_size)
    x = get_dummy_data(batch_size=batch_size)

    @cube.compile(model, x, PAS=policy, scale=True)
    def train_iter(model, x):
        loss = model(x)
        loss.backward()
        return loss
    
    model = cube.load_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for _ in range(3):
        x = get_dummy_data(batch_size=batch_size)
        loss = train_iter(model, x)
        loss = loss * nreplicas
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0  # scale for comparison
        losses.append(loss)

    return losses


def scale_test():
    cube.init()
    CompileFlag.disable_code_line_info = True  # speedup parse
    assert_parity(baseline, partial(scale, 2))


def scale_test_dp():
    cube.init()
    CompileFlag.disable_code_line_info = True  # speedup parse
    assert_parity(baseline, partial(scale, 1))


test_scale_2gpu = partial(torchrun, 2, scale_test)
test_scale_2gpu_dp = partial(torchrun, 2, scale_test_dp)
test_scale_4gpu = partial(torchrun, 4, scale_test)
