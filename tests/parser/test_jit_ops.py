"""
torchrun --nproc_per_node=1 tests/parser/test_jit_ops.py
"""
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops


class TestOpModule(torch.nn.Module):

    def __init__(self, shape=[256, 512]):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor, cache: int):
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
        return x4


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self) -> None:
        self.sample = (
            torch.rand([256, 512], dtype=torch.float32, device=torch.cuda.current_device()),
            4
        )
        batch_size = self.sample[0][0]
        super().__init__(batch_size, (0, None))

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
        data1, data2 = next(dataloader)
        out = model(data1, data2)

    model = model.get_gen_module()

    for idx in range(3):
        eval_iter(model, dataloader)
        print(f"iter {idx}/3")


if __name__ == '__main__':
    test_parse_ops()

