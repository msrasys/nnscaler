import torch

import nnscaler.runtime.utils as runtime_utils


def _reset_empty_cache_state():
    runtime_utils._EMPTY_CACHE_RELEASE_COUNTER = 0
    runtime_utils._EMPTY_CACHE_LAST_CALL_COUNTER = -1


def test_maybe_empty_cache_after_release_is_interval_and_cooldown_throttled(monkeypatch):
    _reset_empty_cache_state()
    calls = []

    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_ON_RELEASE', '1')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_RELEASE_INTERVAL', '4')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_COOLDOWN_INTERVAL', '8')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_THRESHOLD_MB', '512')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_FRAGMENTATION_RATIO', '0.10')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_SYNC', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 10 * 1024 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 8 * 1024 * 1024 * 1024)
    monkeypatch.setattr(
        torch.cuda, 'memory_stats',
        lambda: (_ for _ in ()).throw(AssertionError('memory_stats should not be needed')),
    )
    monkeypatch.setattr(
        torch.cuda, 'empty_cache',
        lambda: calls.append(runtime_utils._EMPTY_CACHE_RELEASE_COUNTER),
    )

    for _ in range(12):
        runtime_utils.maybe_empty_cache_after_release()

    assert calls == [4, 12]


def test_maybe_empty_cache_after_release_uses_inactive_split_as_fallback(monkeypatch):
    _reset_empty_cache_state()
    calls = []

    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_ON_RELEASE', '1')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_RELEASE_INTERVAL', '1')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_COOLDOWN_INTERVAL', '0')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_THRESHOLD_MB', '512')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_FRAGMENTATION_RATIO', '0.10')
    monkeypatch.setenv('NNSCALER_EMPTY_CACHE_SYNC', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 10 * 1024 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 9900 * 1024 * 1024)
    monkeypatch.setattr(
        torch.cuda, 'memory_stats',
        lambda: {'inactive_split_bytes.all.current': 2 * 1024 * 1024 * 1024},
    )
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty'))

    runtime_utils.maybe_empty_cache_after_release()

    assert calls == ['empty']
