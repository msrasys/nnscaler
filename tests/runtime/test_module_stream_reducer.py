"""Manual gradients returning None must retain their CUDA write dependencies."""

import pytest
import torch
import torch.distributed as dist

import nnscaler
from nnscaler.runtime.adapter.reducer import Reducer, accumulate_reducer_grad, mark_reducer_grad_ready
from nnscaler.runtime.stream import ModuleStreams


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('two_streams', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_manual_gradient_waits_before_reduction(tmp_path, two_streams, asynchronous):
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        parameter = torch.nn.Parameter(torch.ones(1 << 18, device='cuda'))
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
