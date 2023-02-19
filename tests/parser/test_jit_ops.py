"""
torchrun --nproc_per_node=1 tests/parser/test_torch_ops.py
"""
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops


class TestOpModule(torch.nn.Module):

    def __init__(self, shape=[256, 512]):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
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
    

def test_parse_ops():

    cube.init()

    model = TestOpModule()
    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([256, 512],), dtypes=(torch.float32,), batch_dims=(0,))
    
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
        data = next(dataloader)
        out = model(data)

    model = model.get_gen_module()

    for _ in range(3):
        eval_iter(model, dataloader)


if __name__ == '__main__':
    test_parse_ops()

