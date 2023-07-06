"""
PYTHONPATH=.:$PYTHONPATH torchrun --nproc_per_node=1 \
    tests/graph/test_segment.py
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

    def forward(self, x: torch.Tensor, y: int):
        x = x * self.param
        x = x + y
        loss = torch.sum(x)
        return loss


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int = 256) -> None:
        self.sample = (
            torch.rand([batch_size, 512], dtype=torch.float32, device=torch.cuda.current_device()),
            4,
        )
        super().__init__(batch_size, (0, None))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


def test_segment_creation():

    cube.init()

    model = TestOpModule()
    dataloader = TestDataLoader()
    
    def policy(graph: IRGraph, resource):
        assert resource.ngpus == 1
        fwops = graph.select(ntype=IRFwOperation)
        graph.staging([fwops[0]])
        print(graph.extra_repr())
        for node in fwops:
            graph.assign(node, 0)
        for dl in graph.select(ntype=IRDataOperation):
            graph.assign(dl, 0)
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
    test_segment_creation()
