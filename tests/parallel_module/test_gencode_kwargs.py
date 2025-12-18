import nnscaler
from nnscaler import parallelize, ComputeConfig

import torch

from .test_gencode import _gencode_contains, replace_all_device_with, print_gencode


# note: annotation is wrong. test only
@nnscaler.register_op('*, * -> 1', name='kw_operator')
def kw_operator(x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    a = kwargs.get('a', 1)
    b = kwargs.get('b', 2)
    x = (x * a + b) @ y
    return torch.sum(x)

# note: annotation is wrong. test only
@nnscaler.register_op('*, * -> 1', name='kw_operator2')
def kw_operator2(x: torch.Tensor, y: torch.Tensor, kwargs) -> torch.Tensor:
    a = kwargs.get('a', 1)
    b = kwargs.get('b', 2)
    x = (x * a + b) @ y
    return torch.sum(x)


class KwargsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x, **kwargs):
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 2)
        c = kwargs['c']
        return kw_operator(x, self.scale, **kwargs) \
            + kw_operator2(x, self.scale, kwargs) +  a + b + c


@replace_all_device_with('cpu')
def test_kwargs(tmp_path):
    m = KwargsModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4), 'a': 3, 'c': 4, 'd': 5},
        'dp',
        ComputeConfig(1, 1, constant_folding=False),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert _gencode_contains(tmp_path, KwargsModule, 0,
        r'_operator\.getitem\(kwargs_.*, \'a\'\)')
    assert not _gencode_contains(tmp_path, KwargsModule, 0,
        r'_operator\.getitem\(kwargs_.*, \'b\'\)')
    assert _gencode_contains(tmp_path, KwargsModule, 0,
        r'_operator\.getitem\(kwargs_.*, \'c\'\)')
    assert _gencode_contains(tmp_path, KwargsModule, 0,
        r'_operator\.getitem\(kwargs_.*, \'d\'\)')
    assert _gencode_contains(tmp_path, KwargsModule, 0,
        r'tests.parallel_module.test_gencode_kwargs.kw_operator\(x_\d+, self.scale_\d+, a=getitem_[\d_]+, c=getitem_[\d_]+, d=getitem_[\d_]+\)')
    assert _gencode_contains(tmp_path, KwargsModule, 0,
        r"tests.parallel_module.test_gencode_kwargs.kw_operator2\(x_\d+, self.scale_\d+, kwargs=\{'a': getitem_[\d_]+, 'c': getitem_[\d_]+, 'd': getitem_[\d_]+\}\)")
    # code looks like:
    # def segment49(self, x_31, **kwargs_6):
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2338, in test_kwargs,  parallelize(
    #     getitem_28 = _operator.getitem(kwargs_6, 'a')
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2338, in test_kwargs,  parallelize(
    #     getitem_1_29 = _operator.getitem(kwargs_6, 'c')
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2338, in test_kwargs,  parallelize(
    #     getitem_2_30 = _operator.getitem(kwargs_6, 'd')
    #     # created at IRAdapterGener:local_consumer_multiref
    #     x_52, x_56 = nnscaler.runtime.function.multiref(x_31, times=2)
    #     del x_31
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     kw_operator_34 = tests.parallel_module.test_gencode_kwargs.kw_operator(x_52, self.scale_33, a=getitem_28, c=getitem_1_29, d=getitem_2_30)
    #     del x_52
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     kw_operator2_35 = tests.parallel_module.test_gencode_kwargs.kw_operator2(x_56, self.scale_33, kwargs={'a': getitem_28, 'c': getitem_1_29, 'd': getitem_2_30})
    #     del x_56
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     add_36 = torch.add(kw_operator_34, kw_operator2_35, alpha=1)
    #     del kw_operator_34, kw_operator2_35
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     add_1_37 = torch.add(add_36, getitem_28, alpha=1)
    #     del add_36
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     add_2_38 = torch.add(add_1_37, 2, alpha=1)
    #     del add_1_37
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2330, in forward,  return kw_operator(x, self.scale, **kwargs) \
    #     add_3_32 = torch.add(add_2_38, getitem_1_29, alpha=1)
    #     del add_2_38
    #     return add_3_32

    # def _forward_impl(self, x, **kwargs):
    #     add_3_32 = self.segment49(x, **kwargs)
    #     return add_3_32
    assert True


# note: annotation is wrong. test only
@nnscaler.register_op('*, * -> 1', name='dict_operator')
def dict_operator(x: torch.Tensor, y: torch.Tensor, kwargs: dict) -> torch.Tensor:
    a = kwargs.get('a', 1)
    b = kwargs.get('b', 2)
    x = (x * a + b) @ y
    return torch.sum(x)


class DictargsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x, kwargs: dict):
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 2)
        c = kwargs['c']
        return dict_operator(x, self.scale, kwargs) \
            + kw_operator(x, self.scale, **kwargs) +  a + b + c


@replace_all_device_with('cpu')
def test_dictargs(tmp_path):
    m = DictargsModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4), 'kwargs': {'a': 3, 'c': 4, 'd': 5}},
        'dp',
        ComputeConfig(1, 1, constant_folding=False),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert _gencode_contains(tmp_path, DictargsModule, 0,
        r"builtins.dict.get\(kwargs_.*, 'a', 1\)")
    assert _gencode_contains(tmp_path, DictargsModule, 0,
        r"builtins.dict.get\(kwargs_.*, 'b', 2\)")
    assert len(_gencode_contains(tmp_path, DictargsModule, 0,
        r"_operator\.getitem\(kwargs_.*, 'a'\)")) == 1
    assert len(_gencode_contains(tmp_path, DictargsModule, 0,
        r"_operator\.getitem\(kwargs_.*, 'c'\)")) == 2
    assert _gencode_contains(tmp_path, DictargsModule, 0,
        r'_operator\.getitem\(kwargs_.*, \'d\'\)')
    assert _gencode_contains(tmp_path, DictargsModule, 0,
        r'tests.parallel_module.test_gencode_kwargs.kw_operator\(x_\d+, self.scale_\d+, a=getitem_[\d_]+, c=getitem_[\d_]+, d=getitem_[\d_]+\)')
    assert _gencode_contains(tmp_path, DictargsModule, 0,
        r"tests.parallel_module.test_gencode_kwargs.dict_operator\(x_\d+, self.scale_\d+, kwargs=kwargs_\d+\)")
    # code looks like:
    # def segment52(self, x_35, kwargs_6):
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2411, in forward,  a = kwargs.get('a', 1)
    #     get_7 = builtins.dict.get(kwargs_6, 'a', 1)
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2412, in forward,  b = kwargs.get('b', 2)
    #     get_1_8 = builtins.dict.get(kwargs_6, 'b', 2)
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2413, in forward,  c = kwargs['c']
    #     getitem_31 = _operator.getitem(kwargs_6, 'c')
    #     # created at IRAdapterGener:local_consumer_multiref
    #     x_56, x_60 = nnscaler.runtime.function.multiref(x_35, times=2)
    #     del x_35
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     dict_operator_38 = tests.parallel_module.test_gencode_kwargs.dict_operator(x_56, self.scale_37, kwargs=kwargs_6)
    #     del x_56
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     getitem_1_32 = _operator.getitem(kwargs_6, 'a')
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     getitem_2_33 = _operator.getitem(kwargs_6, 'c')
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     getitem_3_34 = _operator.getitem(kwargs_6, 'd')
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     kw_operator_39 = tests.parallel_module.test_gencode_kwargs.kw_operator(x_60, self.scale_37, a=getitem_1_32, c=getitem_2_33, d=getitem_3_34)
    #     del x_60
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     add_40 = torch.add(dict_operator_38, kw_operator_39, alpha=1)
    #     del dict_operator_38, kw_operator_39
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     add_1_41 = torch.add(add_40, get_7, alpha=1)
    #     del add_40
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     add_2_42 = torch.add(add_1_41, get_1_8, alpha=1)
    #     del add_1_41
    #     # File "/data/weijiangxu/MagicCube/tests/parallel_module/test_gencode_kwargs.py", line 2414, in forward,  return dict_operator(x, self.scale, kwargs) \
    #     add_3_36 = torch.add(add_2_42, getitem_31, alpha=1)
    #     del add_2_42
    #     return add_3_36

    # def _forward_impl(self, x, kwargs):
    #     add_3_36 = self.segment52(x, kwargs)
    #     return add_3_36
