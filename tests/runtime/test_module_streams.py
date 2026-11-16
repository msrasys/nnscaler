# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.stream import ModuleStreams, drain_stream_holds


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('recompute', [False, True])
def test_module_stream_forward_backward_matches_serial(recompute):
    torch.manual_seed(1107)
    layers = torch.nn.ModuleList([torch.nn.Linear(128, 256), torch.nn.Linear(256, 128)]).cuda()
    reference = copy.deepcopy(layers)
    x = torch.randn(64, 128, device='cuda', requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()

    def prefix(value):
        return layers[0](value).relu()

    for _ in range(3):
        with ModuleStreams() as streams:
            with streams.fork('prefix', 'test_shared', (x, *layers[0].parameters())):
                h = checkpoint(prefix, x, use_reentrant=False) if recompute else prefix(x)
            streams.publish((h,))
            routed = x.sin()
            with streams.fork('suffix', 'test_shared', (h, *layers[1].parameters())):
                shared = layers[1](h)
            streams.publish((shared,))
            routed = routed.cos()
            streams.join('suffix')
            actual = routed + shared
        expected = ref_x.sin().cos() + reference[1](reference[0](ref_x).relu())
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, ref_x.grad)
        for a, b in zip(layers.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


def test_module_stream_storage_survives_allocator_pressure():
    # Warm allocation bins on both streams before delaying a consumer, so a
    # cudaMalloc does not accidentally synchronize away a use-after-free.
    size = 1 << 20
    with ModuleStreams() as streams:
        with streams.fork('warm', 'test_lifetime', ()):
            warm = [torch.empty(size, device='cuda') for _ in range(16)]
        streams.publish(warm)
    del warm
    torch.cuda.synchronize()
    for _ in range(4):
        x = torch.full((size,), 3., device='cuda')
        with ModuleStreams() as streams:
            with streams.fork('delayed', 'test_lifetime', (x,)):
                torch.cuda._sleep(20_000_000)
                y = x.square()
            streams.publish((y,))
            del x
            garbage = [torch.full((size,), -17., device='cuda') for _ in range(16)]
            streams.join('delayed')
            result = y + 1
            del y
        torch.testing.assert_close(result, torch.full_like(result, 10.))
        del garbage, result
    drain_stream_holds(wait=True)
