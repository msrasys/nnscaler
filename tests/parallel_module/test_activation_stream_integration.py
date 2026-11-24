"""Exercise the joint checkpoint/offload/stream path against eager autograd."""

import copy
from collections import OrderedDict
from itertools import product

import pytest
import torch

import nnscaler
from nnscaler.graph.recompute import merge_recompute_regions
from nnscaler.graph.schedule.module_streams import ModuleStreamRule, recompute_region_key
from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from nnscaler.policies import OpPlan, fn, get_pas_ops
from nnscaler.runtime.cpu_offloading import CPUOffloadConfig
from tests.graph.test_recompute_merge import _chain
from tests.parallel_module.test_module_streams import SerialStreams
from tests.utils import new_empty, replace_all_device_with


class Region(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).tanh()


class CombinedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Region()
        self.second = Region()
        self.side = Region()

    def forward(self, x):
        value = torch.sin(x)
        return self.second(self.first(value)) + self.side(x)


def configured_policy(graph, config):
    plans = []
    for node in get_pas_ops(graph):
        group = next((index for index, name in enumerate(('first', 'second', 'side'))
                      if name in (node.module_stack or {})), -1)
        plans.append(OpPlan(node, stage_id=0, recompute_id=group,
                            offload_id=group if config.cpu_offload is not None else -1))
    return fn(graph, config, lambda *_: plans)


def configuration(merge, offload, streams):
    return ComputeConfig(
        1, 1, recompute_merge_adjacent=merge,
        cpu_offload=CPUOffloadConfig(max_cpu_bytes=32 * 1024**2, min_tensor_bytes=0,
                                    host_reserve_bytes=0, skip_retained_inputs=False)
                    if offload else None,
        module_streams=(ModuleStreamRule('side', stream='side', launch_before='torch.sin'),)
                       if streams else (),
    )


def compare_gradients(generated, reference, x):
    reference_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(reference_x)
    actual.square().sum().backward()
    expected.square().sum().backward()
    if torch.distributed.is_initialized():
        generated.sync_grad()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(
            parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


def checkpoint_roundtrip(generated, path, device):
    optimizer = torch.optim.Adam(generated.parameters(), lr=1e-3)
    optimizer.step()
    torch.save({'model': generated.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    restored = (new_empty(type(generated), device=device, init_params=True)
                if device == 'cpu' else type(generated)(init_params=True))
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    # Let load_state_dict place each optimizer tensor; non-capturable Adam's
    # step counter intentionally stays on CPU even when parameters are on CUDA.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored.load_state_dict(saved['model'])
    restored_optimizer.load_state_dict(saved['optimizer'])
    x = torch.linspace(-1, 1, 32, device=device).reshape(4, 8)
    for model, optim in ((generated, optimizer), (restored, restored_optimizer)):
        optim.zero_grad(set_to_none=True)
        model(x).square().sum().backward()
        if torch.distributed.is_initialized():
            model.sync_grad()
        optim.step()
    for actual, expected in zip(generated.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), restored_optimizer.state_dict(), rtol=0, atol=0)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('merge,offload,streams', list(product((False, True), repeat=3)))
def test_joint_features_cpu_semantics(tmp_path, monkeypatch, merge, offload, streams):
    # Serialize only stream execution on CPU; real codegen and saved-value hooks run.
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = CombinedModel()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, requires_grad=True)
    instance = f'combined_cpu_{int(merge)}{int(offload)}{int(streams)}'
    parallelize(model, {'x': x.detach().requires_grad_()}, configured_policy,
                configuration(merge, offload, streams), gen_savedir=tmp_path,
                instance_name=instance, load_module=False)
    cls = _load_parallel_module_class(CombinedModel, gen_savedir=tmp_path,
                                      instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    compare_gradients(generated, reference, x)
    checkpoint_roundtrip(generated, tmp_path / 'resume.pt', 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('merge,offload,streams', list(product((False, True), repeat=3)))
def test_joint_features_cuda(tmp_path, merge, offload, streams):
    import torch.distributed as dist

    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group',
                            rank=0, world_size=1)
    try:
        nnscaler.init()
        model = CombinedModel().cuda()
        reference = copy.deepcopy(model)
        x = torch.randn(64, 8, device='cuda', requires_grad=True)
        generated = parallelize(
            model, {'x': x.detach().requires_grad_()}, configured_policy,
            configuration(merge, offload, streams), gen_savedir=tmp_path,
            instance_name=f'combined_cuda_{int(merge)}{int(offload)}{int(streams)}')
        compare_gradients(generated, reference, x)
        checkpoint_roundtrip(generated, tmp_path / 'resume.pt', 'cuda')
        torch.cuda.synchronize()
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize('different_call', [False, True])
def test_merge_keeps_distinct_stream_invocations(different_call):
    plans = _chain(3)
    rules = (ModuleStreamRule('side', stream='replay', recompute_only=True),)
    for index, plan in enumerate(plans):
        path = 'side' if different_call or index == 0 else 'main'
        plan.op.module_stack = OrderedDict([(path, Region)])
        plan.op.module_call_stack = {path: int(index > 0) if different_call else 0}
    merged, report = merge_recompute_regions(
        plans, region_key=lambda nodes: recompute_region_key(nodes, rules))
    assert [plan.recompute_id for plan in merged] == [0, 1, 1]
    assert report['barriers']['scheduling'] == 1


def test_combined_configuration_roundtrip_preserves_every_switch(tmp_path, monkeypatch):
    previous = []
    for bits in product((False, True), repeat=3):
        config = configuration(*bits)
        path = tmp_path / (''.join(str(int(bit)) for bit in bits) + '.pt')
        ComputeConfig.safe_dump_to_file(config, path)
        # A caller's current defaults cannot silently enable saved-off features.
        monkeypatch.setenv('NNSCALER_MODULE_STREAMS', '[{"module":"unexpected"}]')
        loaded = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
        assert ComputeConfig.safe_equals(config, loaded)
        assert all(not ComputeConfig.safe_equals(old, loaded) for old in previous)
        previous.append(loaded)
