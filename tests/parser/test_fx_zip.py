"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/parser/test_fx_zip.py
"""
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops

cube.init()

class TestModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fcs = torch.nn.Sequential(torch.nn.Linear(512, 10, bias=False), torch.nn.Linear(512, 10, bias=False))
    
    def forward(self, x: torch.Tensor):
        result = []
        xs = x.chunk(2, dim=1)
        for x, fc in zip(xs, self.fcs):
            result.append(fc(x))
        res = torch.cat(result)
        return {'result': res, 'loss': torch.sum(res)}


class TestDataLoader(cube.runtime.syndata.CubeDataLoader):
    def __init__(self, batch_size: int = 256) -> None:
        self.sample = torch.rand(
            [batch_size, 1024],
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


def test_zip():
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

    @cube.compile(model, dataloader, PAS=policy, load_content=False, model_dummy_inputs={'x': next(dataloader)})
    def eval_iter(model, dataloader):
        data = next(dataloader)
        out = model(data)
        out['loss'].backward()

    model = model.get_gen_module()

    for idx in range(3):
        eval_iter(model, dataloader)
        print(f"iter {idx}/3")


if __name__ == '__main__':
    # zip should not appear in graph
    test_zip()
