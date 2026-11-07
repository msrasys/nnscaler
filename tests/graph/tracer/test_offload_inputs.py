import pytest
import torch

from nnscaler.graph.tracer import concrete_trace, wrap_utils
from nnscaler.graph.tracer.trace_strategy import CudaRunCpuOffloadStrategy, ReuseCacheStrategy


pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


def test_repeated_inputs_are_copied_back_with_shared_identity():
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.arange(24.).reshape(4, 6)
    result, args, kwargs = strategy.run_call_function(
        lambda value, *, other: value + other, (x,), {'other': x})
    torch.testing.assert_close(result, x * 2)
    assert args[0] is kwargs['other'] and args[0] is not x
    torch.testing.assert_close(args[0], x)
    result, args, _ = strategy.run_call_function(lambda value: value, (x,), {})
    assert result is args[0] and result is not x
    torch.testing.assert_close(result, x)


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


def test_cpu_mutation_through_closure_does_not_replace_executed_inputs():
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.ones(8)

    def operation(value):
        x.add_(3)
        return value

    result, args, _ = strategy.run_call_function(operation, (x,), {})
    assert result is args[0] and result is not x
    torch.testing.assert_close(result, torch.ones(8))
    torch.testing.assert_close(x, torch.full((8,), 4.))


def test_requires_grad_change_without_version_change_is_copied_back():
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


def test_copied_nonleaf_input_keeps_real_autograd_gradient():
    strategy = CudaRunCpuOffloadStrategy(None)
    leaf = torch.randn(8, requires_grad=True)
    x = leaf * 2
    result, args, _ = strategy.run_call_function(lambda value: value.square(), (x,), {})
    assert args[0] is not x
    torch.testing.assert_close(args[0], x)
    result.sum().backward()
    torch.testing.assert_close(leaf.grad, leaf * 8)


def _fill_count_with_triton(count):
    import triton
    import triton.language as tl

    @triton.jit
    def fill_count(count):
        tl.store(count, 5)

    version = count._version
    fill_count[(1,)](count)
    assert count._version == version
    return count


@pytest.mark.parametrize('strategy_cls', [CudaRunCpuOffloadStrategy, ReuseCacheStrategy])
@pytest.mark.parametrize('return_input', [False, True])
def test_triton_input_mutation_without_version_change_is_copied_back(strategy_cls, return_input):
    pytest.importorskip('triton')
    strategy = strategy_cls(None)
    count = torch.zeros(1, dtype=torch.int64)

    def operation(value):
        _fill_count_with_triton(value)
        return value if return_input else None

    result, args, _ = strategy.run_call_function(operation, (count,), {})
    assert args[0].device.type == 'cpu'
    assert args[0].item() == 5
    assert count.item() == 0
    if return_input:
        assert result is args[0]
    else:
        assert result is None


class TritonCountModule(torch.nn.Module):
    def forward(self, count):
        _fill_count_with_triton(count)
        return torch.ones(int(count.item()))


@pytest.mark.parametrize('strategy', ['cuda_run_cpu_offload', 'reuse_cache'])
def test_triton_input_mutation_preserves_traced_values_and_shapes(strategy):
    pytest.importorskip('triton')
    traced = concrete_trace(
        TritonCountModule(), {'count': torch.zeros(1, dtype=torch.int64)},
        strategy=strategy, dce=False,
        autowrap_leaf_function={_fill_count_with_triton: wrap_utils.LeafWrapInfo([], True, None)},
    )
    item_node = next(node for node in traced.graph.nodes
                     if node.op == 'call_method' and node.target == 'item')
    ones_node = next(node for node in traced.graph.nodes
                     if node.op == 'call_function' and node.target is torch.ones)
    assert item_node.meta['tensor_meta'] == 5
    assert ones_node.meta['tensor_meta'].shape == (5,)
    assert traced(torch.zeros(1, device='cuda', dtype=torch.int64)).shape == (5,)
