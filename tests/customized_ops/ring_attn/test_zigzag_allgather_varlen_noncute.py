#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Regression test for the non-cute path of zigzag_allgather_attn_varlen_func.

On main, zigzag_allgather_attn_varlen_func is absent, so this test is skipped.
On yutao/attention / PR #32, the symbol exists and this test exercises the
non-cute branch that previously returned only the attention output while callers
unpacked it as (out, lse).
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist

from tests.launch_torchrun import torchrun


try:
    from flash_attn import flash_attn_varlen_func  # noqa: F401
except ImportError:
    pytest.skip("flash_attn_varlen_func not available", allow_module_level=True)

try:
    from nnscaler.customized_ops.ring_attention.core.zigzag_allgather_attn_varlen_implementation import (  # noqa: E501
        zigzag_allgather_attn_varlen_func,
    )
except ImportError:
    pytest.skip(
        "zigzag_allgather_attn_varlen_func not present on this branch",
        allow_module_level=True,
    )


def _zigzag_allgather_noncute_worker():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    nheads, head_dim = 4, 32
    local_seq_len = 64 * world_size
    seq_len = 64 * 2 * world_size
    q = torch.randn(local_seq_len, nheads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, nheads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, nheads, head_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    out = zigzag_allgather_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        process_group=dist.group.WORLD,
        causal=True,
        use_cute=False,
    )

    assert out.shape == q.shape, f"unexpected output shape {out.shape}"
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="needs >=2 GPUs to exercise the CP branch",
)
def test_zigzag_allgather_attn_varlen_noncute_two_gpus():
    partial(torchrun, 2, _zigzag_allgather_noncute_worker)()
