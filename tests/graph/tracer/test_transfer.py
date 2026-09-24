import pytest
import torch

from nnscaler.graph.tracer.trace_strategy import BaseTraceStrategy


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_repeated_inputs_and_output_share_one_transfer_and_gradient():
    x = torch.arange(16, device='cuda', dtype=torch.float32, requires_grad=True)
    output, args, kwargs = BaseTraceStrategy._place_tensors_to(x, (x,), {'x': x}, device='cpu')
    assert output is args[0] is kwargs['x']
    (output.sum() + args[0].sum()).backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_distinct_views_keep_values_layout_and_no_grad_flags():
    x = torch.arange(24, device='cuda', dtype=torch.float32).reshape(4, 6).requires_grad_()
    y = x.t()
    with torch.no_grad():
        first, second, repeated = BaseTraceStrategy._place_tensors_to(x, y, x, device='cpu')
    assert first is repeated and first is not second
    assert first.requires_grad and second.requires_grad
    assert first.stride() == x.stride() and second.stride() == y.stride()
    assert torch.equal(first, x.cpu()) and torch.equal(second, y.cpu())
