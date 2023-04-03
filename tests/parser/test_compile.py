"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/parser/test_compile.py
"""
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.ir.tensor import IRFullTensor
from cube.graph.function.dimops import IRDimops


cube.init()


class TestOpModule(torch.nn.Module):

    def __init__(self, shape=[256, 512]):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        x = x + cache
        # [256, 512], [256, 512] -> [256, 512]
        x = x * self.param
        # [256, 512] -> [512]
        x1 = x.select(0, 6)
        # [256, 512], [512] -> [256, 512]
        x2 = x.select_scatter(x1, 0, 7)
        # [256, 512] -> [512, 512]
        x3 = x2.repeat(2, 1)
        # [512, 512] -> [256, 512]: this will be parsed to 2 slice operations
        x4 = x3[:256,:]
        loss = x4.sum()
        return loss


class TestDataLoader1(cube.runtime.syndata.CubeDataLoader):

    def __init__(self) -> None:
        self.sample = (
            torch.rand([256, 512], dtype=torch.float32, device=torch.cuda.current_device()),
            torch.rand([256, 512], dtype=torch.float32, device=torch.cuda.current_device()),
        )
        batch_size = self.sample[0][0]
        super().__init__(batch_size, (0, 0))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


class TestDataLoader2(cube.runtime.syndata.CubeDataLoader):

    def __init__(self) -> None:
        self.sample = torch.rand(
            [256, 512], dtype=torch.float32, device=torch.cuda.current_device())
        batch_size = self.sample[0]
        super().__init__(batch_size, (0,))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        return True


model = TestOpModule()
dataloader1 = TestDataLoader1()
dataloader2 = TestDataLoader2()


def graph_check(graph):
    for t in graph.inputs():
        assert not isinstance(t, IRFullTensor)
    for node in graph.nodes():
        for t in node.inputs() + node.outputs():
            assert not isinstance(t, IRFullTensor)
    for t in graph.outputs():
        assert not isinstance(t, IRFullTensor)


def policy(graph, resource):
    graph_check(graph)
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph


def test_compile_with_dataloader():
    global model

    sample, cache = next(dataloader1)

    @cube.compile(model, dataloader1, PAS=policy,
                  model_dummy_inputs={'x': sample, 'cache': cache})
    def train_step(model, dataloader):
        data = next(dataloader)
        print(data)
        loss = model(*data)
        loss.backward()

    gmodel = cube.load_model()
    
    for step in range(4):
        train_step(gmodel, dataloader1)
        print(f'step [{step}/4]')


def test_compile_without_dataloader():
    global model

    dummy_args = next(dataloader1)
    sample, cache = dummy_args

    @cube.compile(model, sample, cache, PAS=policy,
                  model_dummy_inputs={'x': sample, 'cache': cache})
    def train_step(model, x, cache):
        loss = model(x, cache)
        loss.backward()

    gmodel = cube.load_model()

    for step in range(4):
        x, cache = next(dataloader1)
        train_step(gmodel, x, cache)
        print(f'step [{step}/4]')



def test_compile_with_complex():
    global model

    sample = next(dataloader2)
    cache = torch.rand([256, 512], dtype=torch.float32, device=torch.cuda.current_device())
    
    # @cube.compile(model, dataloader2, cache, PAS=policy)
    # print(sample.size(), cache.size())
    
    @cube.compile(model, dataloader2, cache, PAS=policy,
                  model_dummy_inputs={'x': sample, 'cache': cache})
    def train_step(model, dataloader, cache):
        sample = next(dataloader)
        loss = model(sample, cache)
        loss.backward()

    gmodel = cube.load_model()

    for step in range(4):
        train_step(gmodel, dataloader2, step)
        print(f'step [{step}/4]')


    
if __name__ == '__main__':
    test_compile_with_dataloader()
    test_compile_without_dataloader()
    test_compile_with_complex()