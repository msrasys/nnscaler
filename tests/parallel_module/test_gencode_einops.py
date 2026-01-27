#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import functools
from einops import rearrange
import torch
import pytest

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


class RearrangeModule2(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape
        self.weight = torch.nn.Parameter(torch.ones(self.shape))

    def forward(self, x):
        bsz = x.size(0)
        x = rearrange(x, 'n l d -> (n l) d', n=bsz)
        return x + self.weight


@replace_all_device_with('cpu')
@pytest.mark.parametrize("constant_folding", [True, False])
def test_rearrange2(tmp_path, constant_folding):
    parallelize(
        RearrangeModule2(4),
        {'x': torch.randn(4, 4, 4)},
        'dp',
        ComputeConfig(1, 1, constant_folding=constant_folding),
        gen_savedir=tmp_path,
        load_module=False
    )
    # parallelize will succeed.
    assert True

    # code will look like this when constant_folding=True
    # def segment22(self, x_25):
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/_backends.py", line 93, in reshape,  return x.reshape(shape)
    #     reshape_27 = torch.Tensor.reshape(x_25, shape=(16, 4))
    #     del x_25
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_einops.py", line 80, in forward,  return x + self.weight
    #     add_26 = torch.add(reshape_27, self.weight_28, alpha=1)
    #     del reshape_27
    #     return add_26


    # code will look like this when constant_folding=False
    # def segment25(self, x_28):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_einops.py", line 78, in forward,  bsz = x.size(0)
    #     size_21 = torch.Tensor.size(x_28, dim=0)
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/_backends.py", line 90, in shape,  return x.shape
    #     im_output_36 = builtins.getattr(x_28, 'shape')
    #     getattr_3_22 = im_output_36[0]
    #     getattr_3_23 = im_output_36[1]
    #     getattr_3_24 = im_output_36[2]
    #     del im_output_36
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/einops.py", line 33, in _product,  result *= element
    #     mul_1_25 = _operator.mul(1, size_21)
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/einops.py", line 33, in _product,  result *= element
    #     imul_26 = _operator.imul(mul_1_25, getattr_3_23)
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/einops.py", line 33, in _product,  result *= element
    #     mul_2_27 = _operator.mul(1, getattr_3_24)
    #     # File "/data/weijiangxu/uvenv/kosmos3.10/lib/python3.10/site-packages/einops/_backends.py", line 93, in reshape,  return x.reshape(shape)
    #     reshape_30 = torch.Tensor.reshape(x_28, shape=(imul_26, mul_2_27))
    #     del x_28
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_einops.py", line 80, in forward,  return x + self.weight
    #     add_29 = torch.add(reshape_30, self.weight_31, alpha=1)
    #     del reshape_30
    #     return add_29

