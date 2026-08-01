from types import SimpleNamespace

import pytest
import torch

import nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen as zigzag_wrapper


def _inputs(kv_tokens=8):
    q = torch.randn(4, 2, 8)
    k = torch.randn(kv_tokens, 2, 8)
    v = torch.randn(kv_tokens, 2, 8)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    return q, k, v, cu_seqlens


def _patch_multi_gpu_path(monkeypatch):
    captured = {}
    local_process_group = object()

    monkeypatch.setattr(
        zigzag_wrapper,
        'DeviceGroup',
        lambda: SimpleNamespace(
            get_group=lambda ranks: captured.setdefault(
                'resolved_group', (tuple(ranks), local_process_group)
            )[1]
        ),
    )

    def fake_zigzag(q, k, v, *args, **kwargs):
        captured['q'] = q
        captured['k'] = k
        captured['v'] = v
        captured['process_group'] = kwargs['process_group']
        return q, torch.zeros(1)

    monkeypatch.setattr(
        zigzag_wrapper,
        'zigzag_allgather_attn_varlen_func',
        fake_zigzag,
    )
    return captured, local_process_group


def _call_wrapper(q, k, v, cu_seqlens, **kwargs):
    return zigzag_wrapper.wrap_zigzag_allgather_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        causal=True,
        process_group=(2, 3),
        cp_size=2,
        **kwargs,
    )


def test_legacy_annotation_and_runtime_do_not_gather_kv(monkeypatch):
    q, k, v, cu_seqlens = _inputs()
    expected_annotation = (
        'l num_heads hd^, al^ num_heads hd^, al^ num_heads vd^, '
        'e^, e^, ? -> l num_heads vd^, ?'
    )
    assert zigzag_wrapper.flash_attention_anno(
        q, k, v, cu_seqlens, cu_seqlens, None
    ) == expected_annotation

    captured, local_process_group = _patch_multi_gpu_path(monkeypatch)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError('legacy K/V must not be gathered inside the wrapper')

    monkeypatch.setattr(
        zigzag_wrapper, 'allgather_reducescatter', unexpected_gather
    )
    _call_wrapper(q, k, v, cu_seqlens)

    assert captured['k'] is k
    assert captured['v'] is v
    assert captured['resolved_group'][0] == (2, 3)
    assert captured['process_group'] is local_process_group


def test_cp_sharded_kv_annotation_and_runtime_gather_only_cp_group(monkeypatch):
    q, k, v, cu_seqlens = _inputs(kv_tokens=4)
    annotation = zigzag_wrapper.flash_attention_anno(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        cp_sharded_kv=True,
    )
    assert annotation == (
        'l num_heads hd^, l num_heads hd^, l num_heads vd^, '
        'e^, e^, ? -> l num_heads vd^, ?'
    )

    captured, _ = _patch_multi_gpu_path(monkeypatch)
    gather_calls = []

    def fake_gather(tensor, dim, ranks):
        gather_calls.append((tensor, dim, tuple(ranks)))
        return torch.cat((tensor, tensor), dim=dim)

    monkeypatch.setattr(
        zigzag_wrapper, 'allgather_reducescatter', fake_gather
    )
    _call_wrapper(
        q,
        k,
        v,
        cu_seqlens,
        cp_sharded_kv=True,
    )

    assert gather_calls == [(k, 0, (2, 3)), (v, 0, (2, 3))]
    assert captured['k'].shape[0] == 8
    assert captured['v'].shape[0] == 8
    assert all(ranks != (0, 1, 2, 3) for _, _, ranks in gather_calls)


def test_pre_gathered_cp_sharded_kv_skips_internal_gather(monkeypatch):
    q, k, v, cu_seqlens = _inputs()
    captured, _ = _patch_multi_gpu_path(monkeypatch)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError('pre-gathered K/V must not be gathered again')

    monkeypatch.setattr(
        zigzag_wrapper, 'allgather_reducescatter', unexpected_gather
    )
    _call_wrapper(
        q,
        k,
        v,
        cu_seqlens,
        cp_sharded_kv=True,
        kv_is_gathered=True,
    )

    assert captured['k'] is k
    assert captured['v'] is v
    # kv_is_gathered is a runtime override and must not restore legacy al^.
    assert zigzag_wrapper.flash_attention_anno(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        cp_sharded_kv=True,
        kv_is_gathered=True,
    ).startswith('l num_heads hd^, l num_heads hd^, l num_heads vd^')


@pytest.mark.parametrize(
    ('name', 'value'),
    [('cp_sharded_kv', 1), ('kv_is_gathered', 1)],
)
def test_kv_layout_flags_require_bool(name, value):
    q, k, v, cu_seqlens = _inputs()
    with pytest.raises(ValueError, match=f'{name} must be a bool'):
        _call_wrapper(q, k, v, cu_seqlens, **{name: value})
