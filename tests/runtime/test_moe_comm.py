#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only (no distributed launch or GPU needed) unit tests for
``nnscaler.runtime.adapter.moe`` (Step C's real MoE expert-parallel
all-to-all communication primitive).

Covers validation/config error paths (illegal ``ep_ranks``, buffer-shape/
capacity mismatch, ``channel`` without ``max_outstanding``) and the
degenerate single-EP-rank identity case, all of which are reachable without
any real ``torch.distributed`` collective actually executing (the
validation happens before any collective call is issued -- see module
docstring). Real, multi-GPU communication correctness (the actual
all-to-all, real async issue/deferred-wait, real gradient flow) is proven by
``tests/parallel_module/test_phase_moe_e2e.py``/
``test_phase_moe_multistage_e2e.py`` instead -- that is what genuinely needs
a live NCCL process group, which a CPU-only unit test cannot honestly fake.

Non-uniform, per-token-varying routing/capacity-drop logic (dependent on
``tests.parallel_module.phase_moe_common``'s ``_capacity_scatter``/
``_capacity_gather``, not on ``nnscaler.runtime.adapter.moe`` itself) is
separately unit-tested in :mod:`test_moe_capacity_routing` below (also
CPU-only, pure-tensor -- no nnScaler compilation or distributed context
needed since these are plain, ordinary ``torch`` functions).
"""
import pytest
import torch

from nnscaler.runtime.adapter.moe import (
    MoECommError,
    moe_dispatch,
    moe_dispatch_wait,
    moe_combine,
    moe_combine_wait,
    _check_ep_ranks,
)


# ---------------------------------------------------------------------------
# _check_ep_ranks validation
# ---------------------------------------------------------------------------

def test_check_ep_ranks_rejects_empty():
    with pytest.raises(MoECommError):
        _check_ep_ranks(torch.zeros(2, 3, 4), ())


def test_check_ep_ranks_rejects_duplicates():
    with pytest.raises(MoECommError):
        _check_ep_ranks(torch.zeros(2, 3, 4), (0, 0))


def test_check_ep_ranks_rejects_shape_mismatch():
    """buffer's leading dim (2) must equal len(ep_ranks) (3)."""
    with pytest.raises(MoECommError):
        _check_ep_ranks(torch.zeros(2, 3, 4), (0, 1, 2))


def test_check_ep_ranks_rejects_0d_buffer():
    with pytest.raises(MoECommError):
        _check_ep_ranks(torch.zeros(()), (0,))


def test_check_ep_ranks_accepts_matching_shape():
    ep_ranks = _check_ep_ranks(torch.zeros(2, 3, 4), [0, 1])
    assert ep_ranks == (0, 1)  # normalized to a tuple


# ---------------------------------------------------------------------------
# Degenerate single-EP-rank identity path (no real communication needed --
# see moe_dispatch's own docstring: "ep_ranks=(r,) degenerates to a true
# no-op identity").
# ---------------------------------------------------------------------------

def test_moe_dispatch_single_rank_is_identity():
    buf = torch.randn(1, 4, 8)
    out = moe_dispatch(buf, (0,))
    assert out is buf


def test_moe_combine_single_rank_is_identity():
    buf = torch.randn(1, 4, 8)
    out = moe_combine(buf, (3,))
    assert out is buf


def test_moe_dispatch_wait_is_identity_for_already_resolved_tensor():
    """A tensor never issued through AsyncCommHandler (e.g. the single-rank
    identity path above) is safely a no-op to wait on."""
    buf = torch.randn(1, 4, 8)
    out = moe_dispatch_wait(buf)
    assert torch.equal(out, buf)


def test_moe_combine_wait_is_identity_for_already_resolved_tensor():
    buf = torch.randn(1, 4, 8)
    out = moe_combine_wait(buf)
    assert torch.equal(out, buf)


# ---------------------------------------------------------------------------
# channel/max_outstanding validation -- raised before any real collective is
# issued (see nnscaler/runtime/adapter/moe.py's `_issue_all_to_all_ep`), so
# reachable even with >= 2 "ranks" and no real torch.distributed context.
# ---------------------------------------------------------------------------

def test_moe_dispatch_rejects_channel_without_max_outstanding():
    buf = torch.randn(2, 4, 8)
    with pytest.raises(MoECommError):
        moe_dispatch(buf, (0, 1), channel='ch0', max_outstanding=None)


def test_moe_combine_rejects_channel_without_max_outstanding():
    buf = torch.randn(2, 4, 8)
    with pytest.raises(MoECommError):
        moe_combine(buf, (0, 1), channel='ch0', max_outstanding=None)


def test_moe_dispatch_rejects_bad_shape_before_any_communication():
    """Shape validation for a would-be-real (len(ep_ranks) > 1) dispatch
    must also fail fast, before any collective is attempted."""
    buf = torch.randn(3, 4, 8)  # leading dim 3 != len(ep_ranks) 2
    with pytest.raises(MoECommError):
        moe_dispatch(buf, (0, 1))


def test_moe_combine_rejects_bad_shape_before_any_communication():
    buf = torch.randn(3, 4, 8)
    with pytest.raises(MoECommError):
        moe_combine(buf, (0, 1))


# ---------------------------------------------------------------------------
# Registration sanity: dispatch/combine (and their waits) are genuinely
# distinct, registered custom ops (not aliases of one function) -- see
# nnscaler/runtime/adapter/moe.py's module docstring for why this matters
# for gencode text distinguishability.
# ---------------------------------------------------------------------------

def test_dispatch_and_combine_are_distinct_registered_ops():
    from nnscaler.graph.parser.register import CustomizedOps
    dispatch_sig = 'nnscaler.runtime.adapter.moe.moe_dispatch'
    combine_sig = 'nnscaler.runtime.adapter.moe.moe_combine'
    assert CustomizedOps.exist(dispatch_sig)
    assert CustomizedOps.exist(combine_sig)
    assert CustomizedOps.kOpRuntime[dispatch_sig] is not CustomizedOps.kOpRuntime[combine_sig]
    assert CustomizedOps.kOpFakeRuntime[dispatch_sig] is not None
    assert CustomizedOps.kOpFakeRuntime[combine_sig] is not None


def test_fake_runtime_is_communication_free_and_shape_preserving():
    """The registered `fake_fn` (used during tracing, see register_op's own
    docstring) must be a plain, communication-free, shape-preserving
    substitute -- calling it directly (as tracing would, on a CPU tensor,
    no distributed context) must not raise and must preserve shape/dtype."""
    from nnscaler.graph.parser.register import CustomizedOps
    fake = CustomizedOps.kOpFakeRuntime['nnscaler.runtime.adapter.moe.moe_dispatch']
    buf = torch.randn(2, 4, 8)
    out = fake(buf, (0, 1), None, None)
    assert out.shape == buf.shape and out.dtype == buf.dtype
