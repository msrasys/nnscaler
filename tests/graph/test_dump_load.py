"""
USE_TORCHFX=1 torchrun --nproc_per_node=2 tests/graph/test_dump_load.py
"""
from typing import List
import torch
from cube.ir.cten import IRObject

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops


cube.init()


def _param(size, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(size, dtype=dtype))


class TestOpModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = _param([512, 256])
        self.param2 = _param([512, 256])
        self.ints = [1, 2, 3]

    def forward(self, x: torch.Tensor):
        # matmul: [bs, 512], [512, 256] -> [bs, 256]
        x1 = torch.matmul(x, self.param1)
        # [bs, 256] -> [bs, 256]
        x1 = x1 + x1.size(0) + x1.size()[0]
        # [bs, 256] -> [bs, 128], [bs, 128]
        x2 = torch.chunk(x1, 2, dim=1)[0]
        # [bs, 128] -> [bs, 128]
        x3 = x2 + x2.size(0)
        x4 = x3 + self.ints[0]
        # [bs, 128] -> [1]
        loss = torch.sum(x4)
        return {'x': x4, 'loss': loss} # , [x3,]


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int = 256) -> None:
        self.sample = torch.rand(
            [batch_size, 512],
            dtype=torch.float32,
            device=torch.cuda.current_device()
        )
        super().__init__(batch_size, (0,))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


def test_graph_dump_load_single():

    model = TestOpModule()
    dataloader = TestDataLoader()
    
    def policy(graph: IRGraph, resource):
        print('================ original one:')
        print(graph.extra_repr())
        
        graph.dump('graph.pickle')
        new_graph = IRGraph.load('graph.pickle')

        print('================ loaded from pickled one:')
        print(graph.extra_repr())

        for node in graph.nodes():
            for t in node.inputs():
                if isinstance(t, IRObject):
                    assert t.cell is not None

        assert graph.extra_repr() == new_graph.extra_repr()

        assert resource.ngpus == 1
        for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
            if node.name == 'add':
                sub_nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=resource.ngpus)
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
        return graph


    @cube.compile(model, dataloader, PAS=policy, load_content=False,
                  model_dummy_inputs={'x': next(dataloader)})
    def train_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out['loss'].backward()

    model = cube.load_model(load_content=False)

    for idx in range(3):
        train_iter(model, dataloader)
        print(f"iter {idx}/3")


def test_graph_dump_load_with_transform():

    model = TestOpModule()
    dataloader = TestDataLoader()
    
    def policy(graph: IRGraph, resource):
        print('================ original one:')
        print(graph.extra_repr())
        old_repr = graph.extra_repr()
        
        graph.dump('graph.pickle')
        graph = IRGraph.load('graph.pickle')

        print('================ loaded from pickled one:')
        print(graph.extra_repr())
        new_repr = graph.extra_repr()

        for node in graph.nodes():
            for t in node.inputs():
                if isinstance(t, IRObject):
                    assert t.cell is not None

        assert new_repr == old_repr

        assert resource.ngpus == 2
        for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
            graph.assign(node, 0)
        return graph


    @cube.compile(model, dataloader, PAS=policy, load_content=False,
                  model_dummy_inputs={'x': next(dataloader)})
    def train_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out['loss'].backward()

    model = cube.load_model(load_content=False)

    for idx in range(3):
        train_iter(model, dataloader)
        print(f"iter {idx}/3")


if __name__ == '__main__':
    # test_graph_dump_load_single()
    test_graph_dump_load_with_transform()
