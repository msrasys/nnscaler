from contextlib import contextmanager
from typing import Callable
import torch
import math
import random


def init_parameter(model: torch.nn.Module, seed: int = 0):
    """
    Initialize a model's parameters with truncated normal distribution.
    """
    def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
        with torch.no_grad():
            l = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
            u = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
        return tensor

    torch.random.manual_seed(seed)
    random.seed(seed)

    for param in list(model.parameters()) + list(model.buffers()):
        if len(param.size()) > 1:
            trunc_normal_(param, std=.02)
        else:
            torch.nn.init.constant_(param, 0)


def assert_parity(baseline_fn: Callable, compile_fn: Callable, atol: float=1e-4) -> bool:
    """Compare the output of baseline_fn and compile_fn

    Error will raise if the output of two functions are not the same.

    Args:
        baseline_fn (Callable): a function that returns the output of baseline
        compile_fn (Callable): a function that returns the output of compile (cube)
        atol (Callable): absolute tolerance when comparing two torch tensors

    Returns:
        result (bool): True if the output of two functions are the same else raise Error
    """
    baseline_outputs = baseline_fn()
    compile_outputs = compile_fn()

    print(f'comparing\nGT:\t{baseline_outputs}\nOUT:\t{compile_outputs}')

    def assert_same_complex(gt, out):
        if isinstance(gt, tuple):
            assert isinstance(out, tuple)
            for ele_gt, ele_out in zip(gt, out):
                assert_same_complex(ele_gt, ele_out)
        elif isinstance(gt, list):
            assert isinstance(out, list)
            for ele_gt, ele_out in zip(gt, out):
                assert_same_complex(ele_gt, ele_out)
        elif isinstance(gt, dict):
            assert isinstance(out, dict)
            assert set(gt.keys()) == set(out.keys())
            for key in gt:
                assert_same_complex(gt[key], out[key])
        elif isinstance(gt, torch.Tensor):
            assert isinstance(out, torch.Tensor)
            assert torch.allclose(gt, out, atol=atol), f'mismatched: {gt} != {out}'
        elif isinstance(gt, float):
            assert isinstance(out, float)
            assert math.isclose(gt, out, abs_tol=atol), f'mismatched: {gt} != {out}'
        else:
            assert gt == out, f'mismatched: {gt} != {out}'
    assert_same_complex(baseline_outputs, compile_outputs)
    return None


@contextmanager
def replace_all_device_with(device='cpu'):
    from cube.graph.parser.fx.concrete_trace_utils.concrete_tracer import ConcreteTracer

    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda
    orig_cpu = torch.Tensor.cpu

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] =device
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        return wrapper
    # these constructors are enough for most cases
    patched_tensor_constructors = [
        'empty', 'zeros', 'ones', 'full', 'eye',
        'linspace', 'logspace', 'arange',
        'rand', 'randn', 'randint', 'randperm',
        'randn_like', 'rand_like', 'randint_like',
        'tensor'
    ]
    old_tensor_constructors = {
        tf_name: getattr(torch, tf_name)
        for tf_name in patched_tensor_constructors
    }
    patched_tensor_constructors = {
        tf_name: patch_tensor_constructor(fn)
        for tf_name, fn in old_tensor_constructors.items()
    }

    def patched_to(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (torch.device, str)):
            args[0] = device
            return orig_to(self, *args, **kwargs)
        if 'device' in kwargs:
            kwargs['device'] = device
            return orig_to(self, *args, **kwargs)
        return orig_to(self, *args, **kwargs)

    def patched_cuda(self, *args, **kwargs):
        return orig_to(self, device)

    def patched_cpu(self, *args, **kwargs):
        return orig_to(self, device)

    try:
        torch.Tensor.to = patched_to
        torch.Tensor.cuda = patched_cuda
        torch.Tensor.cpu = patched_cpu
        # patch tensor constructors
        for tf_name, fn in old_tensor_constructors.items():
            setattr(torch, tf_name, patched_tensor_constructors[tf_name])

        # patch concrete tracer's autowrap leaf function
        for tf_name, fn in old_tensor_constructors.items():
            leaf_info = ConcreteTracer.default_autowrap_leaf_function.pop(fn, None)
            if leaf_info:
                ConcreteTracer.default_autowrap_leaf_function[
                    patched_tensor_constructors[tf_name]
                ] = leaf_info
        yield
    finally:
        for tf_name, fn in patched_tensor_constructors.items():
            leaf_info = ConcreteTracer.default_autowrap_leaf_function.pop(fn, None)
            if leaf_info:
                ConcreteTracer.default_autowrap_leaf_function[
                    old_tensor_constructors[tf_name]
                ] = leaf_info
        for tf_name, fn in old_tensor_constructors.items():
            setattr(torch, tf_name, fn)
        torch.Tensor.to = orig_to
        torch.Tensor.cuda = orig_cuda
        torch.Tensor.cpu = orig_cpu
