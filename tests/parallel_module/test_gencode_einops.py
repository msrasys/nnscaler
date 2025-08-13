#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import functools
from einops import rearrange
import torch

from nnscaler import parallelize, ComputeConfig
from nnscaler.graph import parser
from nnscaler.graph.tracer import ConcreteTracer

from tests.utils import replace_all_device_with
from .test_gencode import _gencode_contains, print_gencode


class RearrangeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        return self.linear(x) + rearrange(y, '(h w) -> h w', h=3, w=3) + f(3)


def log_f(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function '{func.__name__}' called")
        return func(*args, **kwargs)
    return wrapper


@log_f
@functools.cache
def f(x: int) -> int:
    return x * 2


@replace_all_device_with('cpu')
def test_trace_rearrange():
    import gc
    def _convert():
        model = RearrangeModule()
        parser.to_fx_graph(model, {'x': torch.randn(3, 3), 'y': torch.randn(9)})
        gc.collect()

    _convert()
    for obj in gc.get_objects():
        # einops is using functools.cache
        # will leak memory if not properly handle it.
        assert not isinstance(obj, ConcreteTracer)


@replace_all_device_with('cpu')
def test_codegen_rearrange():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            RearrangeModule(),
            {'x': torch.randn(3, 3), 'y': torch.randn(9)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # parallelize will succeed.
        assert True
