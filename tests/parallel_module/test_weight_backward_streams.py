# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import copy

import pytest
import torch

from nnscaler import ComputeConfig, ModuleStreamRule, parallelize, register_op
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.policies import pas_dp
from nnscaler.ir.operator import IRFwOperation
from ..runtime.test_weight_accumulator import fragment
from ..utils import new_empty, replace_all_device_with


@register_op('n h^, h^ k^ -> n k^')
def cooperative_linear(x, weight):
    return fragment(x, weight, [])


class CooperativeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return cooperative_linear(x, self.weight)


class CooperativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = CooperativeLayer()

    def forward(self, x):
        return self.expert(x).sin()


RULE = ModuleStreamRule('expert', operators='*.cooperative_linear', stream='expert_dw', weight_backward_only=True)

OBSERVE = False
OBSERVED = []
WEIGHT_DONE = None


class ObserveInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 3

    @staticmethod
    def backward(ctx, grad):
        dx = grad * 3
        if OBSERVE:
            ready = torch.cuda.Event()
            ready.record()
            ready.synchronize()
            OBSERVED.append(not WEIGHT_DONE.query())
        return dx


@register_op('n h -> n h')
def observe_input(x):
    return ObserveInput.apply(x)


class ObserveModel(CooperativeModel):
    def forward(self, x):
        return super().forward(observe_input(x))


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('workers', [1, 2])
def test_weight_only_rule_preserves_checkpoint_and_gradients(tmp_path, checkpointed, workers):
    def policy(graph, config):
        if checkpointed:
            graph.recompute([node for node in graph.select(ntype=IRFwOperation)
                             if node.signature.endswith('.cooperative_linear')])
        return pas_dp(graph, config)

    model = CooperativeModel()
    reference = copy.deepcopy(model)
    instance = f'weight_only_{checkpointed}_{workers}'
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, policy,
                ComputeConfig(1, 1, module_streams=(RULE,)), gen_savedir=tmp_path,
                codegen_workers=workers, load_module=False, instance_name=instance)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'nnscaler.WeightGradientAccumulator(' in code
    assert 'track_stream_inputs = False' in code
    assert '_module_streams.fork' not in code
    assert ('ckpt.checkpoint(' in code) == checkpointed
    tree = ast.parse(code)
    affinities = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id=='module_parameter_streams' for t in n.targets))
    assert len(affinities)==1 and set(affinities.values())=={'expert_dw'}
    cls = _load_parallel_module_class(CooperativeModel, gen_savedir=tmp_path, rank=0, instance_name=instance)
    generated = new_empty(cls, device='cpu', init_params=True)
    inputs = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    actual, expected = [generated(x) for x in inputs], [reference(x) for x in refs]
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
    for a,b in zip(inputs,refs):torch.testing.assert_close(a.grad,b.grad)
    for name,p in generated.named_parameters():
        torch.testing.assert_close(p.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


@pytest.mark.parametrize('kwargs', [dict(), dict(recompute_only=True), dict(launch_before='x'), dict(overlap_backward=True)])
def test_weight_rules_reject_ambiguous_modes(kwargs):
    if kwargs:kwargs=dict(operators='*.cooperative_linear',**kwargs)
    with pytest.raises(ValueError):
        ModuleStreamRule('expert', stream='dw', weight_backward_only=True, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_generated_weight_affinity_cuda_keeps_input_backward_independent(tmp_path, monkeypatch):
    import torch.distributed as dist
    import nnscaler
    from nnscaler.runtime.weight_accumulator import WeightGradientAccumulator
    from nnscaler.runtime.adapter.reducer import Reducer

    def policy(graph, config):
        graph.recompute([node for node in graph.select(ntype=IRFwOperation)
                         if node.signature.endswith('.cooperative_linear')])
        return pas_dp(graph, config)

    global OBSERVE, WEIGHT_DONE
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/pg', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = ObserveModel().cuda()
        reference = copy.deepcopy(model)
        generated = parallelize(model, {'x': torch.randn(4,8,device='cuda',requires_grad=True)}, policy,
                                ComputeConfig(1,1,module_streams=(RULE,)),gen_savedir=tmp_path,
                                instance_name='cuda_weight_scope')
        weight = next(generated.parameters())
        reducer = Reducer([0])
        reducer.add_param(weight)
        reducer.build_buckets()
        reducer.zero_grad()
        warm = generated(torch.randn(4,8,device='cuda',requires_grad=True))
        warm.sum().backward()
        reducer.sync_grads()
        del warm
        torch.cuda.synchronize()
        reducer.zero_grad()
        WEIGHT_DONE = torch.cuda.Event()
        original = WeightGradientAccumulator.accumulate

        def delayed(region, indices, callback, inputs):
            def work(outputs):
                torch.cuda._sleep(500_000_000)
                result = callback(outputs)
                WEIGHT_DONE.record()
                return result
            return original(region, indices, work, inputs)

        monkeypatch.setattr(WeightGradientAccumulator,'accumulate',delayed)
        OBSERVED.clear()
        OBSERVE = True
        x = torch.randn(4,8,device='cuda',requires_grad=True)
        actual = generated(x)
        actual.square().sum().backward()
        reducer.sync_grads()
        assert OBSERVED == [True]
        OBSERVE = False
        ref_x = x.detach().clone().requires_grad_()
        expected = reference(ref_x)
        expected.square().sum().backward()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(x.grad,ref_x.grad)
        for name,p in generated.named_parameters():
            torch.testing.assert_close(p.grad,reference.get_parameter(generated.fullmap[name].orig_name).grad)
    finally:
        OBSERVE = False
        nnscaler.uninit()
        if dist.is_initialized():dist.destroy_process_group()
