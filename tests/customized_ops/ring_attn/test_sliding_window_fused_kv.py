import torch

from nnscaler.customized_ops.ring_attention.core import (
    sliding_window_attn_implementation as sliding,
)


def _metadata(input_splits, output_splits):
    return sliding.SlidingWindowMetadata(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=1,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
    )


def test_fused_pair_supports_receive_only_and_different_payload_shapes(monkeypatch):
    first = torch.empty((0, 2, 3))
    second = torch.empty((0, 1, 4))
    expected_first = torch.arange(12).reshape(2, 2, 3).float()
    expected_second = (torch.arange(8) + 20).reshape(2, 1, 4).float()
    expected_fused = torch.cat(
        (expected_first.reshape(2, 6), expected_second.reshape(2, 4)), dim=1
    )
    calls = []

    def fake_all_to_all(tensor, input_splits, output_splits, group):
        calls.append((tensor, input_splits, output_splits, group))
        return expected_fused

    monkeypatch.setattr(sliding, "_all_to_all_varlen", fake_all_to_all)

    received_first, received_second = sliding._all_to_all_varlen_pair(
        first, second, [0, 0], [2, 0], object()
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (0, 10)
    torch.testing.assert_close(received_first, expected_first)
    torch.testing.assert_close(received_second, expected_second)


def test_fused_pair_supports_empty_send_and_receive(monkeypatch):
    calls = 0

    def fake_all_to_all(tensor, input_splits, output_splits, group):
        nonlocal calls
        calls += 1
        return tensor.new_empty((0, tensor.shape[1]))

    monkeypatch.setattr(sliding, "_all_to_all_varlen", fake_all_to_all)

    first, second = sliding._all_to_all_varlen_pair(
        torch.empty((0, 2, 3)),
        torch.empty((0, 1, 4)),
        [0, 0],
        [0, 0],
        object(),
    )

    assert calls == 1
    assert first.shape == (0, 2, 3)
    assert second.shape == (0, 1, 4)


def test_sliding_forward_and_backward_each_use_one_collective(monkeypatch):
    calls = 0

    def fake_all_to_all(tensor, input_splits, output_splits, group):
        nonlocal calls
        calls += 1
        return tensor.clone()

    monkeypatch.setattr(sliding, "_all_to_all_varlen", fake_all_to_all)
    metadata = _metadata([1], [1])
    k = torch.arange(12).reshape(3, 2, 2).float()
    v = (torch.arange(9) + 20).reshape(3, 1, 3).float()

    extended_k, extended_v = sliding._a2a_communicate_kv(k, v, metadata, object())

    assert calls == 1
    torch.testing.assert_close(extended_k, torch.cat((k[-1:], k), dim=0))
    torch.testing.assert_close(extended_v, torch.cat((v[-1:], v), dim=0))

    dk_recv = torch.full((1, 2, 2), 2.0)
    dv_recv = torch.full((1, 1, 3), 3.0)
    dk_from_next, dv_from_next = sliding._a2a_communicate_grad(
        dk_recv,
        dv_recv,
        torch.zeros_like(k),
        torch.zeros_like(v),
        metadata,
        object(),
    )

    assert calls == 2
    torch.testing.assert_close(dk_from_next, dk_recv)
    torch.testing.assert_close(dv_from_next, dv_recv)
