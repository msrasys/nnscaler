"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/parser/test_no_grad.py
"""
from typing import List
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops

cube.init()


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(512, 10)
    
    def forward(self, x: torch.Tensor):
        # this no grad will be dce
        with torch.no_grad():
            pass

        # this no grad will not be dce
        with torch.no_grad():
            res = self.fc(x)
        
        return {'res': res, 'loss': res.sum()}


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


def test_no_grad():

    model = TestModel()
    dataloader = TestDataLoader()
    
    def policy(graph, resource):
        print(graph.extra_repr())
        assert resource.ngpus == 1
        for node in graph.nodes():
            if isinstance(node, IRDimops):
                print(f'# {node.anno}')
                print(node)
            elif isinstance(node, (IRFwOperation, IRDataOperation)):
                print(node)
        for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
            graph.assign(node, 0)
        return graph

    model = cube.SemanticModel(model)

    @cube.compile(model, dataloader, PAS=policy, load_content=False,
                  model_dummy_inputs={'x': next(dataloader)})
    def eval_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out['loss'].backward()
        # return out

    model = model.get_gen_module()

    for idx in range(3):
        eval_iter(model, dataloader)
        print(f"iter {idx}/3")


if __name__ == '__main__':
    # consecutive no_grad __enter__ __exit__ sequences will be dce
    test_no_grad()
