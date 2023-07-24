"""
USE_TORCHFX=1 torchrun --nproc_per_node=1 tests/parser/test_cus_autograd.py
"""
import torch

import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.dimops import IRDimops
from cube.graph.parser import register

cube.init()


class GeLU(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input: torch.Tensor, bias: torch.Tensor):
        ctx.save_for_backward(input, bias)
        return GeLU.bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = GeLU.bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

    @staticmethod
    def bias_gelu(bias, y):
        x = bias + y
        return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @staticmethod
    def bias_gelu_back(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff*g


class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 10)
        self.bias = torch.nn.Parameter(torch.rand(10))

    def forward(self, x: torch.Tensor):
        res = GeLU.apply(self.fc(x), self.bias)
        loss = res.sum()
        return {'res': res, 'loss': loss}


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


def test_cus_autograd():
    register('* h, h -> * h')(GeLU.apply)

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
    test_cus_autograd()
