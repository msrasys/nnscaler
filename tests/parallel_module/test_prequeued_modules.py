import copy
from contextlib import contextmanager

import pytest
import torch
import nnscaler
from nnscaler import ModuleStreamRule, ComputeConfig
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.deferred import PrequeuedLaunchPoint
from .test_module_streams import CPUDeferredStreams, dispatch_anchor, combine_anchor
from ..utils import replace_all_device_with, new_empty


class Side(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(4, 8)
        self.down = torch.nn.Linear(8, 4)
    def forward(self, x):
        return self.down(self.up(x).sin())


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.side = Side()
    def forward(self, x):
        return combine_anchor(dispatch_anchor(x.sin()).cos()) + self.side(x)


class CPUStreams(CPUDeferredStreams):
    def schedule_native(self, key, stream, function, parent, inputs, attrs, *,
                        kernel_graphs=False, prequeue=False, preserve_accumulators=False):
        super().schedule_native(key, stream, function, parent, inputs, attrs, kernel_graphs=kernel_graphs)
        assert prequeue
        self.native_results[key] = function(parent, *(inputs() if callable(inputs) else inputs))

    @contextmanager
    def launch_point(self, keys, *, native=False, prequeued=False, backward_keys=(), callbacks=()):
        assert native and prequeued
        def release(key):
            self.native.pop(key)
        dependencies = lambda: tuple(v for key in backward_keys for v in self.native_results[key])
        with PrequeuedLaunchPoint(tuple(lambda key=key: release(key) for key in keys) + tuple(callbacks),
                                  lambda ready: None, dependencies):
            yield


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('split', ['none', 'same', 'separate'])
def test_generated_prequeued_regions_preserve_math(tmp_path, monkeypatch, workers, split):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUStreams)
    model = Model()
    reference = copy.deepcopy(model)
    kwargs = {} if split == 'none' else dict(split_before='down',
        resume_on='*.dispatch_anchor' if split == 'same' else '*.combine_anchor')
    rule = ModuleStreamRule('side', stream='prequeued', launch_on='*.dispatch_anchor',
                            launch_phase='prequeued', overlap_backward=True, **kwargs)
    name = f'prequeued_{workers}_{split}'
    nnscaler.parallelize(model, {'x': torch.randn(4, 4, requires_grad=True)}, 'dp',
        ComputeConfig(1, 1, module_streams=[rule]), gen_savedir=tmp_path,
        instance_name=name, codegen_workers=workers, load_module=False)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'prequeue=True' in source and 'prequeued=True' in source
    cls = _load_parallel_module_class(Model, gen_savedir=tmp_path, instance_name=name, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    y, expected = generated(x), reference(rx)
    y.square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
    for attr, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[attr].orig_name).grad, rtol=0, atol=0)


def test_prequeued_requires_explicit_backend_boundary():
    with pytest.raises(ValueError, match='launch_on'):
        ModuleStreamRule('side', launch_before='*.anchor', launch_phase='prequeued')
