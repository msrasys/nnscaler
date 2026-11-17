import copy

import pytest
import torch

import nnscaler
from nnscaler import ComputeConfig, ModuleStreamRule
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.weight_backward import WeightBackwardRegion
from ..utils import replace_all_device_with, new_empty
from ..runtime.test_weight_backward import Anchor, Linear
from .test_module_streams import SerialStreams


@nnscaler.register_op('a b -> a b')
def weight_anchor(x):
    return Anchor.apply(nnscaler.prepare_async_input(x), [])


@nnscaler.register_op('a k, k n -> a n')
def weight_linear(x, w):
    binding = nnscaler.prepare_weight_backward((w,))
    return Linear.apply(x, w if binding is None else binding.parameters[0], binding, [])


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 6))

    def forward(self, x):
        return weight_linear(x, self.weight)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = Expert()

    def forward(self, x):
        return self.expert(weight_anchor(x))


class Streams(SerialStreams):
    def __enter__(self):
        super().__enter__()
        self.weights = []
        return self

    def weight_region(self, key, name, parameters):
        region = WeightBackwardRegion(name, parameters)
        self.weights.append(region)
        return region

    def __exit__(self, *exc):
        for region in self.weights:
            region.close_forward(failed=exc[0] is not None)
        return super().__exit__(*exc)


def policy(graph, config):
    from nnscaler.policies import pas_dp
    graph.recompute([node for node in graph.nodes() if 'expert' in (node.module_stack or {})])
    return pas_dp(graph, config)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('workers', [1, 2])
def test_generated_weight_split_preserves_outputs_gradients(tmp_path, monkeypatch, recompute, workers):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', Streams)
    model = Model()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('expert', stream='weights', weight_backward_on='*.weight_anchor')
    instance = f'weights_{recompute}_{workers}'
    nnscaler.parallelize(model, {'x': torch.randn(8, 4, requires_grad=True)},
                         policy if recompute else 'dp', ComputeConfig(1, 1, module_streams=(rule,)),
                         gen_savedir=tmp_path, instance_name=instance, load_module=False,
                         codegen_workers=workers)
    cls = _load_parallel_module_class(Model, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(8, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    assert '.anchor():' in source and '.compute():' in source and '.weight_region(' in source


class RepeatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Model()

    def forward(self, x):
        return self.block(x) + self.block(x * 3)


@replace_all_device_with('cpu', force=True)
def test_repeated_module_invocations_keep_separate_weight_regions(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', Streams)
    model = RepeatedModel()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('block.expert', stream='weights', weight_backward_on='*.weight_anchor')
    nnscaler.parallelize(model, {'x': torch.randn(8, 4, requires_grad=True)}, 'dp',
                         ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                         instance_name='repeated_weights', load_module=False)
    cls = _load_parallel_module_class(RepeatedModel, gen_savedir=tmp_path,
                                     instance_name='repeated_weights', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(8, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    assert next(tmp_path.rglob('gencode0.py')).read_text().count('.weight_region(') == 2


@nnscaler.register_op('a b -> a b')
def combine_anchor(x):
    return Anchor.apply(nnscaler.prepare_async_input(x), [])


class SharedModel(Model):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 6, bias=False)

    def forward(self, x):
        return combine_anchor(self.expert(weight_anchor(x))) + self.shared(x)


@replace_all_device_with('cpu', force=True)
def test_weight_split_composes_with_whole_shared_branch_at_combine(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', Streams)
    model = SharedModel()
    reference = copy.deepcopy(model)
    rules = (ModuleStreamRule('shared', launch_before='*.combine_anchor', overlap_backward=True),
             ModuleStreamRule('expert', stream='weights', weight_backward_on='*.weight_anchor'))
    nnscaler.parallelize(model, {'x': torch.randn(8, 4, requires_grad=True)}, policy,
                         ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
                         instance_name='weights_shared_combine', load_module=False)
    cls = _load_parallel_module_class(SharedModel, gen_savedir=tmp_path,
                                     instance_name='weights_shared_combine', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(8, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
