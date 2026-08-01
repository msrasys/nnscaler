from functools import partial
import os

import pytest
import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_func

from nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen import (
    wrap_zigzag_allgather_attn_varlen_func,
)
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import torchrun


def _lane_tensor(seq_len, nheads, head_dim, lane, kind, device):
    values = torch.arange(
        seq_len * nheads * head_dim,
        dtype=torch.float32,
        device=device,
    )
    phase = {'q': 0.13, 'k': 0.37, 'v': 0.71}[kind]
    scale = {'q': 0.011, 'k': 0.017, 'v': 0.023}[kind]
    return torch.sin(values * scale + lane * 1.3 + phase).reshape(
        seq_len, nheads, head_dim
    ).to(torch.bfloat16)


def _zigzag_indices(seq_len, cp_size, cp_rank, device):
    slice_len = seq_len // (2 * cp_size)
    front = torch.arange(
        cp_rank * slice_len,
        (cp_rank + 1) * slice_len,
        device=device,
    )
    back_slice = 2 * cp_size - cp_rank - 1
    back = torch.arange(
        back_slice * slice_len,
        (back_slice + 1) * slice_len,
        device=device,
    )
    return torch.cat((front, back))


def _cp2_ep4_zigzag_worker():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # All ranks create both groups in the same order.  Each pair is one CP
    # lane; the four-rank world is the EP/plan group and must never be used by
    # the K/V gather.
    device_group = DeviceGroup()
    device_group.get_group([0, 1])
    device_group.get_group([2, 3])
    cp_ranks = (rank // 2 * 2, rank // 2 * 2 + 1)
    cp_rank = rank % 2
    lane = rank // 2

    seq_len, nheads, head_dim = 128, 2, 32
    cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32, device=device
    )
    q_full_data = _lane_tensor(
        seq_len, nheads, head_dim, lane, 'q', device
    )
    k_full_data = _lane_tensor(
        seq_len, nheads, head_dim, lane, 'k', device
    )
    v_full_data = _lane_tensor(
        seq_len, nheads, head_dim, lane, 'v', device
    )

    # Single-GPU reference for this lane.  A position-dependent loss makes
    # the K/V gradient comparison sensitive to accidental cross-lane mixing.
    q_ref = q_full_data.clone().requires_grad_(True)
    k_ref = k_full_data.clone().requires_grad_(True)
    v_ref = v_full_data.clone().requires_grad_(True)
    ref_out = flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        causal=True,
        deterministic=True,
    )
    loss_weight = torch.linspace(
        0.25, 1.25, ref_out.numel(), device=device, dtype=torch.float32
    ).reshape_as(ref_out)
    (ref_out.float() * loss_weight).sum().backward()

    q_indices = _zigzag_indices(seq_len, 2, cp_rank, device)
    kv_start = cp_rank * (seq_len // 2)
    kv_end = kv_start + seq_len // 2
    q = q_full_data.index_select(0, q_indices).clone().requires_grad_(True)
    k = k_full_data[kv_start:kv_end].clone().requires_grad_(True)
    v = v_full_data[kv_start:kv_end].clone().requires_grad_(True)

    out = wrap_zigzag_allgather_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        causal=True,
        deterministic=True,
        process_group=cp_ranks,
        cp_size=2,
        cp_sharded_kv=True,
        return_lse=False,
    )
    (out.float() * loss_weight.index_select(0, q_indices)).sum().backward()

    torch.testing.assert_close(
        out, ref_out.index_select(0, q_indices), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        q.grad, q_ref.grad.index_select(0, q_indices), rtol=8e-2, atol=8e-2
    )
    torch.testing.assert_close(
        k.grad, k_ref.grad[kv_start:kv_end], rtol=8e-2, atol=8e-2
    )
    torch.testing.assert_close(
        v.grad, v_ref.grad[kv_start:kv_end], rtol=8e-2, atol=8e-2
    )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason='needs four GPUs for CP2/EP4 lane-isolation validation',
)
def test_cp2_ep4_zigzag_forward_and_qkv_gradients():
    partial(torchrun, 4, _cp2_ep4_zigzag_worker)()
