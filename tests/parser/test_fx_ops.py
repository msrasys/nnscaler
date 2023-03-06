"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/parser/test_fx_ops.py
"""
from typing import List
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops


class TestOpModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.empty([512, 256], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.empty([512, 256], dtype=torch.float32))
        self.ints = [1, 2, 3]

    def forward(self, x: torch.Tensor):
        # matmul: [256, 512], [512, 256] -> [256, 256]
        x1 = torch.matmul(x, self.param1)
        x1 = torch.matmul(x, self.param1)
        x1 = x1 + x1.size(0) + x1.size()[0]
        x2 = torch.chunk(x, 2, dim=1)
        x3 = x2[0]
        x = x + x.size(0)
        x = x + self.ints[0]
        return {'x': x}, [x3,]


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int = 256) -> None:
        # self.sample = (
        #     torch.rand(
        #         [batch_size, 512],
        #         dtype=torch.float32,
        #         device=torch.cuda.current_device()
        #     ),
        #     [torch.tensor([1], dtype=torch.float32),]
        # )
        # super().__init__(batch_size, (0, None))
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


def test_parse_ops():

    cube.init()

    model = TestOpModule()
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

    @cube.compile(model, dataloader, policy, load_content=False)
    def eval_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        return out

    model = model.get_gen_module()

    for idx in range(3):
        eval_iter(model, dataloader)
        print(f"iter {idx}/3")


if __name__ == '__main__':
    test_parse_ops()

