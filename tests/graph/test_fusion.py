"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/graph/test_fusion.py
"""
from typing import List
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops

cube.init()


def _param(size, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(size, dtype=dtype))
    

class TestModuleForFusedOp(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = _param([512, 256])
        self.param2 = _param([512, 256])
        self.ints = [1, 2, 3]

    def forward(self, x: torch.Tensor):
        # matmul: [bs, 512], [512, 256] -> [bs, 256]
        x1 = torch.matmul(x, self.param1)
        # [bs, 256] -> [bs, 256]
        x2 = x1.clone()
        x3 = x2 + 1
        loss = torch.sum(x3)
        return {'x': x3, 'loss': loss} # , [x3,]


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



def test_fused_op():

    model = TestModuleForFusedOp()
    dataloader = TestDataLoader()
    
    def policy(graph, resource):
        assert resource.ngpus == 1
        print(graph.extra_repr())

        clone = graph.select(name='clone')[0]
        idx = graph.index(clone)
        clonse_add = [clone, graph.node(idx+1)]
        graph.fuse(clonse_add)

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
    test_fused_op()
