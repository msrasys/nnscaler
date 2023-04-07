"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/graph/test_infer_grad.py
USE_TORCHFX=1 torchrun --nproc_per_node=2 tests/graph/test_infer_grad.py
"""
from typing import List
import torch

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation

cube.init()


def _param(size, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(size, dtype=dtype))

def _rand(size, dtype=torch.float32):
    return torch.rand(size, dtype=dtype, device=torch.cuda.current_device())


class TestOpModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = _param([256, 512])
        self.param2 = _param([256, 512])
        self.param3 = _param([256, 512])

    def forward(self, x: torch.Tensor):
        x1 = x * self.param1
        x2 = x1 * self.param2  # no grad

        cube.runtime.function.anchor('residual')
        x3 = x1 + 2
        x4 = x3 * self.param3

        loss  = torch.sum(x4)
        return {'intermediate': [x3, x2], 'loss': loss}, loss.data


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self) -> None:
        self.sample = _rand([256, 512])
        super().__init__(256, (0,))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


def policy_test_single_device(graph: IRGraph, resource):
    print(graph.extra_repr())
    for idx, node in enumerate(graph.select(name='mul')):
        if idx == 1:
            assert node.mirror is None
            for t in node.inputs() + node.outputs():
                assert t.grad is None
        elif idx == 2:
            assert node.mirror is not None
            for t in node.inputs() + node.outputs():
                assert t.grad is not None
    for node in graph.select(ntype=(IRDataOperation, IRFwOperation)):
        graph.assign(node, 0)
    return graph


def policy_test_multi_device(graph: IRGraph, resource):
    # multiref
    for ftensor in graph.full_tensors():
        if ftensor.is_attr(): continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

    print(graph.extra_repr())
    assert resource.ngpus == 2
    for idx, node in enumerate(graph.select(ntype=(IRFwOperation, IRDataOperation))):
        devid = 0 if idx < 4 else 1
        graph.assign(node, devid)
    print(graph.extra_repr())
    return graph


def test_single_no_backward_ops():

    model = TestOpModule()
    dataloader = TestDataLoader()

    @cube.compile(model, dataloader, PAS=policy_test_single_device, load_content=False,
                  model_dummy_inputs={'x': next(dataloader)})
    def train_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out[0]['loss'].backward()
        return out

    model = cube.load_model(load_content=False)

    for idx in range(3):
        train_iter(model, dataloader)
        print(f"single device: iter {idx}/3")


def test_multidev_residual():

    model = TestOpModule()
    dataloader = TestDataLoader()

    @cube.compile(model, dataloader, PAS=policy_test_multi_device, load_content=False,
                  model_dummy_inputs={'x': next(dataloader)})
    def train_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out[0]['loss'].backward()
        return out

    model = cube.load_model(load_content=False)

    for idx in range(3):
        train_iter(model, dataloader)
        print(f"multi device: iter {idx}/3")


if __name__ == '__main__':
    if torch.distributed.get_world_size() == 1:
        test_single_no_backward_ops()
    if torch.distributed.get_world_size() == 2:
        test_multidev_residual()
