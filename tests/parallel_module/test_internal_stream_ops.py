# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer, register_op
from nnscaler.policies import OpPlan, get_pas_ops
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0


class _DelayedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        torch.cuda._sleep(100_000)
        return dy @ weight, dy.t() @ x


@register_op('n h^, h^ h^ -> n h^')
def internally_streamed_linear(x, weight):
    if not x.is_cuda:
        return torch.nn.functional.linear(x, weight)
    from nnscaler.runtime.stream import _get_module_stream, input_stream_lifetime
    caller = torch.cuda.current_stream(x.device)
    side = _get_module_stream(x.device, 'internal_linear')
    side.wait_stream(caller)
    with torch.cuda.stream(side), input_stream_lifetime((x, weight), True):
        result = _DelayedLinear.apply(x, weight)
    caller.wait_stream(side)
    result.record_stream(caller)
    return result


class InternalLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8) * .1)
        self.eager = False

    def forward(self, x):
        if self.eager:
            return torch.nn.functional.linear(x, self.weight)
        return internally_streamed_linear(x, self.weight)


class InternalStreamPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([InternalLinear() for _ in range(4)])
        # Only rank 0 owns a declared operator. Rank 1's independent codegen
        # worker must still receive the execution plan's lifetime policy.
        for layer in self.layers[1:]:
            layer.eager = True

    def forward(self, sample):
        x = sample['x']
        for layer in self.layers:
            x = layer(x)
        return x.square().sum()


def internal_stream_policy(graph, config):
    for node in get_pas_ops(graph):
        stage = -1
        if InternalLinear in node.module_class_chain:
            stage = int(node.get_module_fqn(InternalLinear).split('.')[-1])
        yield OpPlan(node, stage_id=stage, partition=None)


def internal_stream_worker():
    nnscaler.init()
    torch.manual_seed(7)
    torch.set_float32_matmul_precision('highest')
    source = InternalStreamPipeline()
    eager = copy.deepcopy(source).cuda()
    for layer in eager.layers:
        layer.eager = True
    expected_params = dict(eager.named_parameters())
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=.01)
    config = ComputeConfig(2, 2, use_end2end=True, use_async_comm=True,
                           multi_stream_ops=('*.internally_streamed_linear',),
                           pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                       'pipeline_scheduler': '1f1b_interleaved'})
    directory = Path(tempfile.gettempdir()) / f'internal_stream_pipeline_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'sample': {'x': torch.ones(8, 8)}}, internal_stream_policy,
                            config, gen_savedir=tempdir, reuse='override', codegen_workers=2).cuda()
        assert model.use_multi_streams
        optimizer = build_optimizer(model, torch.optim.SGD, lr=.01)
        for step in range(3):
            samples = [{'x': torch.arange(64, device='cuda').reshape(8, 8).float() / 64 + micro + step}
                       for micro in range(4)]
            eager_optimizer.zero_grad()
            expected = [eager(sample) for sample in samples]
            torch.stack(expected).sum().backward()
            actual = model.train_step(samples)
            torch.testing.assert_close(actual, expected)
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name).grad,
                                           expected_params[meta.orig_name].grad[meta.slicers])
            optimizer.step()
            eager_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name), expected_params[meta.orig_name][meta.slicers])
            optimizer.zero_grad()
    nnscaler.uninit()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
def test_internal_stream_operator_interleaved_pipeline():
    assert all(launch_torchrun(2, internal_stream_worker).values())
