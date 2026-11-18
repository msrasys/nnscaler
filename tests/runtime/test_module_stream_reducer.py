"""Manual gradients returning None must retain their CUDA write dependencies."""

import pytest
import torch
import torch.distributed as dist

import nnscaler
from nnscaler.runtime.adapter.reducer import Reducer, accumulate_reducer_grad, mark_reducer_grad_ready
from nnscaler.runtime.stream import ModuleStreams, register_parameter_stream


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('two_streams', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_manual_gradient_waits_before_reduction(tmp_path, two_streams, asynchronous):
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        parameter = torch.nn.Parameter(torch.ones(1 << 18, device='cuda'))
        register_parameter_stream(parameter, None)  # mixed caller/side-stream use
        reducer = Reducer([0], async_op=asynchronous)
        reducer.add_param(parameter)
        reducer.build_buckets()
        reducer.zero_grad()

        class Manual(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight, scale):
                ctx.weight, ctx.scale = weight, scale
                return weight * scale

            @staticmethod
            def backward(ctx, gradient):
                torch.cuda._sleep(100_000_000)
                assert accumulate_reducer_grad(ctx.weight, gradient * ctx.scale)
                assert mark_reducer_grad_ready(ctx.weight)
                return None, None

        with ModuleStreams() as streams:
            with streams.fork('first', 'manual_first', (parameter,)):
                first = Manual.apply(parameter, 2.)
            streams.publish((first,))
            if two_streams:
                with streams.fork('second', 'manual_second', (parameter,)):
                    second = Manual.apply(parameter, 3.)
                streams.publish((second,))
                streams.join('second')
            streams.join('first')
            output = first + second if two_streams else first
        output.sum().backward()
        reducer.sync_grads()
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 5. if two_streams else 2.))
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_deferred_weight_reducer_does_not_stall_upstream_input_backward(tmp_path):
    from nnscaler.runtime.weight_backward import WeightBackwardRegion
    from .test_weight_backward import Anchor, Linear
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        x = torch.randn(64, 64, device='cuda', requires_grad=True)
        w = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
        rx, rw = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        register_parameter_stream(w, 'reducer_weight_test')
        reducer = Reducer([0])
        reducer.add_param(w)
        reducer.build_buckets()
        reducer.zero_grad()
        done = torch.cuda.Event()
        observed = []

        class Upstream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value * 3

            @staticmethod
            def backward(ctx, dy):
                dx = dy * 3
                ready = torch.cuda.Event()
                ready.record()
                ready.synchronize()
                observed.append(not done.query())
                return dx

        upstream = Upstream.apply(x)
        region = WeightBackwardRegion('reducer_weight_test', (w,))
        with torch.cuda.stream(region.stream):
            warm = x.detach().t() @ x.detach()
        torch.cuda.synchronize()
        del warm
        original = region.defer

        def defer(indices, callback, inputs):
            def delayed_weight():
                torch.cuda._sleep(1_000_000_000)
                result = callback()
                done.record()
                return result
            original(indices, delayed_weight, inputs)

        region.defer = defer
        with region.anchor():
            routed = Anchor.apply(nnscaler.prepare_async_input(upstream), [])
        with region.compute():
            binding = nnscaler.prepare_weight_backward((w,))
            output = Linear.apply(routed, binding.parameters[0], binding, [])
        region.close_forward()
        output.sum().backward()
        reducer.sync_grads()
        ((rx * 6) @ rw).sum().backward()
        torch.testing.assert_close(w.grad, rw.grad)
        torch.testing.assert_close(x.grad, rx.grad)
        assert observed == [True]
        assert not region.tasks
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()
