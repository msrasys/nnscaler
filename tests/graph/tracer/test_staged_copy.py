import pytest
import torch

from nnscaler.flags import CompileFlag
from nnscaler.graph.tracer import staged_copy
from nnscaler.graph.tracer.trace_strategy import BaseTraceStrategy, CudaRunCpuOffloadStrategy


pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.fixture(autouse=True)
def small_stager(monkeypatch):
    # 1 MiB chunks and no size threshold, so small tensors take several chunks.
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy', True)
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy_min_mb', 0)
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy_chunk_mb', 1)
    monkeypatch.setattr(staged_copy, '_stager', None)
    calls = []
    original = staged_copy._StagedCopy.apply
    monkeypatch.setattr(staged_copy._StagedCopy, 'apply',
                        lambda *args: calls.append(args[1]) or original(*args))
    yield calls


def bits(tensor):
    tensor = tensor.detach().contiguous().cpu()
    return tensor.reshape(-1).view(torch.uint8)


def assert_same_as_to(actual, expected):
    assert type(actual) is type(expected)
    assert actual.device == expected.device and actual.dtype == expected.dtype
    assert actual.shape == expected.shape and actual.stride() == expected.stride()
    assert actual.requires_grad == expected.requires_grad and actual.is_leaf == expected.is_leaf
    assert torch.equal(bits(actual), bits(expected))


DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.uint8, torch.bool,
          torch.float8_e4m3fn]


def make(dtype, numel, device):
    values = torch.randn(numel, device=device) * 7
    if dtype == torch.bool:
        return values > 0
    return values.to(dtype)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('numel', [1, 1000, (3 << 20) // 4 + 17])
def test_device_to_host_matches_to(small_stager, dtype, numel):
    x = make(dtype, numel, 'cuda').reshape(numel, 1)
    result = staged_copy.to_device(x, 'cpu')
    assert small_stager
    assert_same_as_to(result, x.to('cpu'))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('numel', [1, 1000, (3 << 20) // 4 + 17])
def test_host_to_device_matches_to(small_stager, dtype, numel):
    x = make(dtype, numel, 'cpu').reshape(1, numel)
    result = staged_copy.to_device(x, 'cuda')
    assert small_stager
    assert_same_as_to(result, x.to('cuda'))


def test_offset_view_and_later_host_mutation(small_stager):
    base = torch.arange(3 << 20, dtype=torch.float32)
    x = base[5:].view(-1)
    result = staged_copy.to_device(x, 'cuda')
    base.zero_()
    assert torch.equal(result.cpu(), torch.arange(5, 3 << 20, dtype=torch.float32))


@pytest.mark.parametrize('direction', ['cpu', 'cuda'])
def test_autograd_relation_matches_to(small_stager, direction):
    source = 'cuda' if direction == 'cpu' else 'cpu'
    for grad_enabled in (True, False):
        x = torch.randn(4096, device=source, requires_grad=True)
        with torch.set_grad_enabled(grad_enabled):
            result = staged_copy.to_device(x, direction)
            expected = x.to(direction)
        assert_same_as_to(result, expected)
    result = staged_copy.to_device(x, direction)
    (result * 3).sum().backward()
    assert x.grad.device == x.device and torch.equal(x.grad, torch.full_like(x, 3))


@pytest.mark.parametrize('requires_grad', [True, False])
def test_parameter_matches_to(small_stager, requires_grad):
    x = torch.nn.Parameter(torch.randn(64, 64, device='cuda'), requires_grad=requires_grad)
    assert_same_as_to(staged_copy.to_device(x, 'cpu'), x.to('cpu'))
    y = torch.nn.Parameter(torch.randn(64, 64), requires_grad=requires_grad)
    assert_same_as_to(staged_copy.to_device(y, 'cuda'), y.to('cuda'))


@pytest.mark.parametrize('make_input', [
    lambda: torch.randn(64, 32, device='cuda').t(),
    lambda: torch.randn(64, device='cuda').expand(8, 64),
    lambda: torch.randn(64, dtype=torch.complex64, device='cuda').conj(),
    lambda: torch.randn(0, 4, device='cuda'),
    lambda: torch.randn(64, device='cpu').pin_memory(),
    lambda: torch.randn(64, device='cuda').to_sparse(),
])
def test_unsupported_inputs_use_to(small_stager, make_input):
    x = make_input()
    target = 'cuda' if x.device.type == 'cpu' else 'cpu'
    result = staged_copy.to_device(x, target)
    assert not small_stager
    expected = x.to(target)
    assert result.layout == expected.layout and result.device == expected.device
    if x.layout == torch.strided:
        assert result.stride() == expected.stride()
        assert torch.equal(result.resolve_conj(), expected.resolve_conj())


def test_same_device_and_disabled_flag_use_to(small_stager, monkeypatch):
    x = torch.randn(64)
    assert staged_copy.to_device(x, 'cpu') is x
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy', False)
    staged_copy.to_device(torch.randn(64, device='cuda'), 'cpu')
    assert not small_stager


def test_inference_tensors_use_to(small_stager):
    with torch.inference_mode():
        x = torch.randn(64, device='cuda')
        assert_same_as_to(staged_copy.to_device(x, 'cpu'), x.to('cpu'))
    assert_same_as_to(staged_copy.to_device(x, 'cpu'), x.to('cpu'))
    assert not small_stager


def test_place_tensors_to_keeps_identity_with_staging(small_stager):
    x = torch.randn(3 << 18, device='cuda', requires_grad=True)
    output, args, kwargs = BaseTraceStrategy._place_tensors_to(x, (x,), {'x': x}, device='cpu')
    assert small_stager == [torch.device('cpu')]
    assert output is args[0] is kwargs['x'] and not output.is_leaf
    (output.sum() + args[0].sum()).backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


def test_offload_strategy_round_trip_with_staging(small_stager):
    strategy = CudaRunCpuOffloadStrategy(None)
    x = torch.randn(3 << 18)
    result, args, _ = strategy.run_call_function(lambda value: value * 2, (x,), {})
    assert len(small_stager) == 3  # input to CUDA, result and input back
    assert result.device.type == 'cpu' and args[0].device.type == 'cpu' and args[0] is not x
    assert torch.equal(result, x * 2) and torch.equal(args[0], x)


@pytest.mark.parametrize('threads', [1, 3, 16])
def test_threaded_host_copy_in_large_chunks(small_stager, monkeypatch, threads):
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy_chunk_mb', 8)
    monkeypatch.setattr(CompileFlag, 'trace_staged_copy_threads', threads)
    numel = (40 << 20) + 3
    x = torch.randint(0, 256, (numel,), dtype=torch.uint8, device='cuda')
    host = staged_copy.to_device(x, 'cpu')
    assert_same_as_to(host, x.to('cpu'))
    y = torch.randn((20 << 20) + 5, dtype=torch.bfloat16)
    assert_same_as_to(staged_copy.to_device(y, 'cuda'), y.to('cuda'))
    assert len(small_stager) == 2
