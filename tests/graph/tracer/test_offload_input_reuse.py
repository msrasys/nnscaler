import pytest
import torch

from nnscaler.graph.tracer.trace_strategy import CudaRunCpuOffloadStrategy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_unchanged_repeated_cpu_inputs_are_reused():
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.arange(24.).reshape(4, 6)
    result, args, kwargs = strategy.run_call_function(
        lambda value, *, other: value + other, (x,), {'other': x})
    torch.testing.assert_close(result, x * 2)
    assert args[0] is kwargs['other'] is x
    result, args, _ = strategy.run_call_function(lambda value: value, (x,), {})
    assert result is args[0] is x


@pytest.mark.parametrize('method,values', [('add_', (2,)), ('transpose_', (0, 1))])
def test_mutated_inputs_are_copied_back_with_result_identity(method, values):
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.arange(24.).reshape(4, 6)
    expected = x.clone()
    getattr(expected, method)(*values)
    result, args, _ = strategy.run_call_method(method, (x, *values), {})
    torch.testing.assert_close(result, expected)
    assert result is args[0] and result is not x
    torch.testing.assert_close(x, torch.arange(24.).reshape(4, 6))


def test_cpu_mutation_through_closure_prevents_reuse():
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.ones(8)

    def operation(value):
        x.add_(3)
        return value

    result, args, _ = strategy.run_call_function(operation, (x,), {})
    assert result is args[0] and result is not x
    torch.testing.assert_close(result, torch.ones(8))
    torch.testing.assert_close(x, torch.full((8,), 4.))


def test_requires_grad_change_without_version_change_prevents_reuse():
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.ones(8)
    result, args, _ = strategy.run_call_method('requires_grad_', (x, True), {})
    assert result is args[0] and result is not x
    assert result.requires_grad and not x.requires_grad


def test_inference_input_uses_copy_back():
    strategy = CudaRunCpuOffloadStrategy(None)
    with torch.inference_mode():
        x = torch.ones(8)
        result, args, _ = strategy.run_call_function(lambda value: value, (x,), {})
    assert result is args[0] and result is not x
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize('kind', ['strided', 'offset', 'parameter', 'grad_leaf'])
def test_transfer_metadata_changes_keep_original_copy_path(kind):
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.arange(24.)
    if kind == 'strided':
        x = x[::2]
    elif kind == 'offset':
        x = x[1:]
    elif kind == 'parameter':
        x = torch.nn.Parameter(x, requires_grad=False)
    else:
        x.requires_grad_()
    expected = x.to('cuda').to('cpu')
    result, args, _ = strategy.run_call_function(lambda value: value, (x,), {})
    assert result is args[0] and result is not x
    torch.testing.assert_close(result, expected)
    assert type(result) is type(expected)
    assert result.stride() == expected.stride()
    assert result.storage_offset() == expected.storage_offset()
    assert result.is_leaf == expected.is_leaf


def test_reused_nonleaf_input_keeps_real_autograd_gradient():
    strategy = CudaRunCpuOffloadStrategy(None)
    leaf = torch.randn(8, requires_grad=True)
    x = leaf * 2
    result, args, _ = strategy.run_call_function(lambda value: value.square(), (x,), {})
    assert args[0] is x
    result.sum().backward()
    torch.testing.assert_close(leaf.grad, leaf * 8)
