import weakref

import pytest
import torch

from nnscaler.graph.tracer.trace_strategy import CpuStrategy, ReuseCacheStrategy
from nnscaler import register_op
from nnscaler.flags import CompileFlag


_fake_calls = []


def _pure_fake(x, weight):
    _fake_calls.append(1)
    return torch.zeros_like(x, requires_grad=torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad))


@register_op('n, n -> n', fake_fn=_pure_fake, fake_fn_on_cpu=True, fake_fn_cacheable=True)
def _runtime_multiply(x, weight):
    return x * weight


def test_registered_cache_keeps_real_graph_arguments(monkeypatch):
    from nnscaler.flags import CompileFlag
    from nnscaler.graph.parser.converter import to_fx_graph

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.ones(8))
            self.w2 = torch.nn.Parameter(torch.full((8,), 2.))

        def forward(self, x):
            return _runtime_multiply(x, self.w1), _runtime_multiply(x, self.w2)

    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'reuse_cache')
    _fake_calls.clear()
    model, x = Model(), torch.arange(8.)
    graph = to_fx_graph(model, {'x': x})
    assert len(_fake_calls) == 1
    calls = [node for node in graph.graph.nodes if node.op == 'call_function']
    assert len(calls) == 2 and all(node.target is _runtime_multiply for node in calls)
    actual = graph(x)
    torch.testing.assert_close(actual, model(x))
    torch.testing.assert_close(actual[0], x)
    torch.testing.assert_close(actual[1], x * 2)


def test_cache_requires_an_explicit_cpu_fake():
    with pytest.raises(ValueError, match='requires fake_fn_on_cpu'):
        register_op('n -> n', fake_fn=_pure_fake, fake_fn_cacheable=True)(_runtime_multiply)


def test_cpu_fake_reuses_outputs_without_substituting_layer_weights():
    strategy = ReuseCacheStrategy(None)
    calls = []

    def fake(x, weight, *, width):
        calls.append(weight)
        return torch.zeros(x.shape[0], width)

    x = torch.randn(4, 8)
    w1, w2 = torch.ones(8, 8), torch.full((8, 8), 2.)
    first, args1, _ = strategy.run_cpu_fake(fake, (x, w1), {'width': 6}, cacheable=True)
    second, args2, _ = strategy.run_cpu_fake(fake, (x, w2), {'width': 6}, cacheable=True)
    assert first is second and len(calls) == 1
    assert args1[1] is w1 and args2[1] is w2
    different, _, _ = strategy.run_cpu_fake(fake, (x, w2), {'width': 7}, cacheable=True)
    assert different.shape == (4, 7) and len(calls) == 2
    with torch.no_grad():
        frozen, _, _ = strategy.run_cpu_fake(fake, (x, w2), {'width': 6}, cacheable=True)
    assert frozen is not first and len(calls) == 3


@pytest.mark.parametrize('location', ['positional', 'nested_positional', 'nested_keyword'])
def test_cpu_fake_distinguishes_input_storage_offsets(monkeypatch, location):
    monkeypatch.setattr(CompileFlag, 'cache_cpu_fake_outputs', True)
    strategy = ReuseCacheStrategy(None)
    calls = []

    def fake(value):
        x = value if location == 'positional' else value['views'][0]
        calls.append(x.storage_offset())
        return torch.zeros(x.storage_offset() + 1)

    def run(view):
        value = view if location == 'positional' else {'views': [view]}
        args, kwargs = ((), {'value': value}) if location == 'nested_keyword' else ((value,), {})
        return strategy.run_cpu_fake(fake, args, kwargs, cacheable=True)[0]

    base = torch.arange(8.)
    first, second = run(base[:4]), run(base[1:5])
    assert first.shape == (1,)
    assert second.shape == (2,)
    # Equal metadata should still reuse outputs across distinct allocations.
    other = torch.full((8,), 42.)
    assert run(other[:4]) is first
    assert run(other[1:5]) is second
    assert calls == [0, 1]


@pytest.mark.parametrize('mutation', ['add_', 'requires_grad_', 'transpose_'])
def test_mutated_fake_outputs_are_not_served_again(mutation):
    strategy = ReuseCacheStrategy(None)
    x = torch.ones(3, 5)

    def fake(x):
        return torch.zeros_like(x), torch.zeros(1)

    first, _, _ = strategy.run_cpu_fake(fake, (x,), {}, cacheable=True)
    args = {'add_': (3,), 'requires_grad_': (True,), 'transpose_': (0, 1)}[mutation]
    getattr(first[0], mutation)(*args)
    second, _, _ = strategy.run_cpu_fake(fake, (x,), {}, cacheable=True)
    assert first[0] is not second[0]
    assert second[0].shape == x.shape and not second[0].requires_grad
    assert second[0].count_nonzero() == 0


def test_fake_cache_is_opt_in_and_specific_to_reuse_strategy():
    x = torch.zeros(8)
    calls = []

    def fake(x):
        calls.append(1)
        return torch.zeros_like(x)

    for strategy, cacheable in [(ReuseCacheStrategy(None), False), (CpuStrategy(None), True)]:
        strategy.run_cpu_fake(fake, (x,), {}, cacheable=cacheable)
        strategy.run_cpu_fake(fake, (x,), {}, cacheable=cacheable)
    assert len(calls) == 4


def test_aliases_and_inference_results_remain_uncached():
    strategy = ReuseCacheStrategy(None)
    x, y = torch.zeros(8), torch.ones(8)
    for value in (x, y):
        result, _, _ = strategy.run_cpu_fake(lambda t: t, (value,), {}, cacheable=True)
        assert result is value
    assert not strategy.fake_cache
    with torch.inference_mode():
        strategy.run_cpu_fake(torch.zeros_like, (x,), {}, cacheable=True)
    assert not strategy.fake_cache


@pytest.mark.parametrize('tuple_output', [False, True])
def test_inference_mode_bypasses_a_warm_no_grad_cache(monkeypatch, tuple_output):
    monkeypatch.setattr(CompileFlag, 'cache_cpu_fake_outputs', True)
    strategy = ReuseCacheStrategy(None)
    x, calls = torch.ones(8), []

    def fake(value):
        calls.append(1)
        result = torch.zeros_like(value)
        return (result, torch.zeros_like(value)) if tuple_output else result

    with torch.no_grad():
        cached, _, _ = strategy.run_cpu_fake(fake, (x,), {}, cacheable=True)
    with torch.inference_mode():
        for _ in range(2):
            result, _, _ = strategy.run_cpu_fake(fake, (x,), {}, cacheable=True)
            outputs = result if tuple_output else (result,)
            assert all(torch.is_inference(value) for value in outputs)
            with pytest.raises(RuntimeError, match='version counter'):
                _ = outputs[0]._version
    with torch.no_grad():
        reused, _, _ = strategy.run_cpu_fake(fake, (x,), {}, cacheable=True)
    assert reused is cached
    assert len(calls) == 3  # Inference calls bypass, but do not evict, the normal cache.


def test_clear_cache_releases_fake_outputs():
    strategy = ReuseCacheStrategy(None)
    result, _, _ = strategy.run_cpu_fake(torch.zeros_like, (torch.ones(8),), {}, cacheable=True)
    ref = weakref.ref(result)
    del result
    assert ref() is not None
    strategy.clear_cache()
    assert ref() is None


def test_disable_fake_cache_keeps_regular_cache(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'cache_cpu_fake_outputs', False)
    strategy = ReuseCacheStrategy(None)
    # Execute ordinary calls on CPU so this test does not require CUDA.
    monkeypatch.setattr(strategy, '_run_cuda_and_offload',
                        lambda runner, target, args, kwargs: runner(target, args, kwargs, device='cpu'))
    x = torch.ones(8)
    first, _, _ = strategy.run_cpu_fake(torch.zeros_like, (x,), {}, cacheable=True)
    second, _, _ = strategy.run_cpu_fake(torch.zeros_like, (x,), {}, cacheable=True)
    assert first is not second and not strategy.fake_cache
    regular, _, _ = strategy.run_call_function(torch.neg, (x,), {})
    repeated, _, _ = strategy.run_call_function(torch.neg, (x,), {})
    assert repeated is regular and len(strategy.cache) == 1


def test_cache_statistics_deduplicate_views_and_both_caches(caplog):
    strategy = ReuseCacheStrategy(None)
    result, _, _ = strategy.run_cpu_fake(torch.zeros_like, (torch.ones(8),), {}, cacheable=True)
    size = result.untyped_storage().nbytes()
    assert strategy.cache_memory_stats() == {'regular': 0, 'cpu_fake': size, 'total': size}
    # These are multiple references/views of the SAME allocation, including
    # a fake result subsequently retained as an ordinary operation's input.
    strategy.cache['regular'] = (result[:2], (result,), {'view': result[2:]})
    assert strategy.cache_memory_stats() == {'regular': size, 'cpu_fake': size, 'total': size}
    extra = torch.zeros(3)
    strategy.cache['extra'] = (extra, (), {})
    assert strategy.cache_size == size + extra.untyped_storage().nbytes()
    with caplog.at_level('INFO'):
        strategy.clear_cache()
    assert 'cpu_fake=' in caplog.text and 'total(unique)=' in caplog.text
    assert strategy.cache_memory_stats() == {'regular': 0, 'cpu_fake': 0, 'total': 0}


def test_cache_statistics_follow_resize_and_invalidation():
    strategy = ReuseCacheStrategy(None)
    x = torch.ones(4)
    first, _, _ = strategy.run_cpu_fake(torch.zeros_like, (x,), {}, cacheable=True)
    first.resize_(32)
    assert strategy.cache_size == first.untyped_storage().nbytes()
    second, _, _ = strategy.run_cpu_fake(torch.zeros_like, (x,), {}, cacheable=True)
    assert second is not first and second.shape == x.shape
    assert strategy.cache_size == second.untyped_storage().nbytes()


@pytest.mark.parametrize('method', [False, True])
def test_regular_cache_statistics_include_nested_tensors(monkeypatch, method):
    strategy = ReuseCacheStrategy(None)
    monkeypatch.setattr(strategy, '_run_cuda_and_offload',
                        lambda runner, target, args, kwargs: runner(target, args, kwargs, device='cpu'))
    x = torch.ones(8)
    if method:
        output, _, _ = strategy.run_call_method('clone', (x,), {})
    else:
        output, _, _ = strategy.run_call_function(torch.neg, (x,), {})
    assert strategy.cache_size == x.untyped_storage().nbytes() + output.untyped_storage().nbytes()
