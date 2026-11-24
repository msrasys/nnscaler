"""Dispatch contracts independent of GPU availability; integration follows on CUDA."""
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from nnscaler.runtime.adapter.collectives import _async_pair_p2p


@pytest.mark.parametrize('send', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_pair_batch_uses_global_peer_and_returns_work(monkeypatch, send, enabled):
    monkeypatch.setenv('NNSCALER_BATCHED_P2P', str(int(enabled)))
    tensor = SimpleNamespace(is_cuda=True)
    group = SimpleNamespace(size=lambda: 2)
    work = object()
    calls = []

    def native(*args, **kwargs):
        calls.append(('native', args, kwargs))
        return work

    monkeypatch.setattr(dist, 'isend' if send else 'irecv', native)
    monkeypatch.setattr(dist, 'P2POp', lambda op, tensor, **kw: (op, tensor, kw))
    def batch(ops):
        calls.append(('batch', ops))
        return [work]
    monkeypatch.setattr(dist, 'batch_isend_irecv', batch)
    assert _async_pair_p2p(native, tensor, 7, group, 1) is work
    if enabled:
        assert calls == [('batch', [(native, tensor, {'peer': 7, 'group': group})])]
    else:
        assert calls == [('native', (tensor,), {'group': group, 'group_dst' if send else 'group_src': 1})]


@pytest.mark.parametrize('group_size,cuda', [(None, True), (4, True), (2, False)])
@pytest.mark.parametrize('send', [False, True])
def test_implicit_larger_or_cpu_groups_keep_native_path(monkeypatch, group_size, cuda, send):
    monkeypatch.setenv('NNSCALER_BATCHED_P2P', '1')
    group = None if group_size is None else SimpleNamespace(size=lambda: group_size)
    tensor = SimpleNamespace(is_cuda=cuda)
    calls = []
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return 'work'
    monkeypatch.setattr(dist, 'isend' if send else 'irecv', native)
    monkeypatch.setattr(dist, 'batch_isend_irecv', lambda _: pytest.fail('unexpected batch'))
    assert _async_pair_p2p(native, tensor, 7, group, 1) == 'work'
    expected = ((tensor, 7), {}) if group is None else (
        (tensor,), {'group': group, 'group_dst' if send else 'group_src': 1})
    assert calls == [expected]
