"""
torchrun --nproc_per_node=2 tests/graph/test_multiref.py
"""
import torch

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops

cube.init()


def _param(shape, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(shape, dtype=dtype))


class TestOpModule(torch.nn.Module):

    def __init__(self, shape=[256, 512]):
        super().__init__()
        self.param = _param(shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x * self.param
        x = torch.sum(x)

        y = y * self.param
        y = torch.sum(y)

        loss = x + y
        return loss


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int = 256) -> None:
        self.sample = (
            torch.rand([batch_size, 512], dtype=torch.float32, device=torch.cuda.current_device()),
            torch.rand([batch_size, 512], dtype=torch.float32, device=torch.cuda.current_device()),
        )
        super().__init__(batch_size, (0, 0))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


def _tp(graph, node, devs, idx, dim):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    for node, devid in zip(sub_nodes, devs):
        graph.assign(node, devid)
    return sub_nodes


def _replica(graph, node, devs):
    rnodes = graph.replicate(node, times=len(devs))
    for rnode, devid in zip(rnodes, devs):
        graph.assign(rnode, devid)
    return rnodes


def test_multiref_param():

    cube.init()

    model = TestOpModule()
    dataloader = TestDataLoader()
    
    def policy(graph: IRGraph, resource):

        # multiref
        for t in graph.full_tensors():
            if len(graph.consumers(t)) > 1:
                graph.multiref(t)
        
        devs = list(range(resource.ngpus))

        muls = graph.select(name='mul')
        _tp(graph, muls[0], devs, idx=1, dim=0)
        _tp(graph, muls[1], devs, idx=1, dim=1)

        for node in graph.select(ntype=(IRDataOperation, IRFwOperation)):
            if node.name == 'multiref': continue
            if node.name == 'mul': continue
            _replica(graph, node, devs)

        return graph
    
    sample_x, sample_y = next(dataloader)

    @cube.compile(model, dataloader, PAS=policy, load_content=True,
                  model_dummy_inputs={'x': sample_x, 'y': sample_y})
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(*data)
        loss.backward()

    model = cube.load_model()

    for idx in range(3):
        train_iter(model, dataloader)
        print(f"iter {idx}/3")
    print('Done')


if __name__ == '__main__':
    test_multiref_param()
    exit(0)
