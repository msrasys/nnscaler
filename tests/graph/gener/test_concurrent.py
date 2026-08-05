#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy

import pytest
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.cten import IRCell
from nnscaler.graph.gener.concurrent import ConcurrentGener, CompileFlag, \
    AllToAllPrim, ReduceScatterPrim, _logger
from ...utils import catch_log


def test_path_retry():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    indmap = []
    for dimlen in ftensor.shape:
        indmap.append((0, dimlen))
    indmap[0] = (0, 2)
    sub1 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (2, 4)
    sub2 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (4, 6)
    sub3 = ftensor.select(tuple(indmap), (0, 1))

    wrong_called = False
    right_called = False
    def path_with_reduce_scatter(*args, **kwargs):
        nonlocal wrong_called, right_called
        if not CompileFlag.disable_reduce_scatter_adapter:
            # the parameter is fake, just for testing
            wrong_called = True
            return [ReduceScatterPrim([sub1, sub2], [sub3], dim=0), AllToAllPrim([sub1, sub3], [sub2], idim=0, odim=1)]
        else:
            right_called = True
            return [AllToAllPrim([sub1, sub2], [sub3], idim=0, odim=1)]

    with catch_log(_logger, 'WARNING') as log_stream:
        assert ConcurrentGener._path(path_with_reduce_scatter, None, None, None)
        assert right_called and wrong_called
        assert 'Detected invalid AllToAllPrim' in log_stream.getvalue()

    called = 0
    def path_without_rc(*args, **kwargs):
        nonlocal called
        called += 1
        return [AllToAllPrim([sub1, sub3], [sub2], idim=0, odim=1)]

    with pytest.raises(RuntimeError, match='Invalid primitives detected.*'):
        with catch_log(_logger) as log_stream:
            ConcurrentGener._path(path_without_rc, None, None, None)

    assert called == 1
    assert 'Detected invalid AllToAllPrim' not in log_stream.getvalue()


def _replica_lanes(ftensor, ranks):
    """Make a pure RVD R(len(ranks)),V(1),D(1,...) layout for a unit test."""
    base = ftensor.tosub()
    tensors = []
    for rank in ranks:
        tensor = copy.copy(base)
        cell = IRCell('lane', '', 0, 0)
        cell.device = rank
        tensor.cell = cell
        tensors.append(tensor)
    return tensors


def test_independent_replica_lanes_identity_mapping_is_noop(monkeypatch):
    ftensor = IRFullTensor((4, 8), requires_grad=True).mark_independent_replica_lanes()
    assert ftensor.like().independent_replica_lanes
    assert ftensor.grad.independent_replica_lanes

    def unexpected_rvd_path(*args, **kwargs):
        raise AssertionError('identity independent lanes must not enter an RVD path')

    monkeypatch.setattr(ConcurrentGener, 'gen_intra_rvd', staticmethod(unexpected_rvd_path))
    monkeypatch.setattr(ConcurrentGener, 'gen_inter_rvd', staticmethod(unexpected_rvd_path))
    fsrc = _replica_lanes(ftensor, [0, 1])
    fdst = _replica_lanes(ftensor, [0, 1])
    bsrc = _replica_lanes(ftensor.grad, [0, 1])
    bdst = _replica_lanes(ftensor.grad, [0, 1])
    assert ConcurrentGener.gen(fsrc, fdst, bsrc, bdst) is None


@pytest.mark.parametrize('consumer_ranks', ([1, 0], [2, 3], [2, 3, 4]))
def test_independent_replica_lanes_nonidentity_fail_closed_before_rvd_collectives(
        monkeypatch, consumer_ranks):
    ftensor = IRFullTensor((4, 8), requires_grad=True).mark_independent_replica_lanes()

    def unexpected_rvd_path(*args, **kwargs):
        raise AssertionError('non-identity independent lanes must fail before an RVD path')

    monkeypatch.setattr(ConcurrentGener, 'gen_intra_rvd', staticmethod(unexpected_rvd_path))
    monkeypatch.setattr(ConcurrentGener, 'gen_inter_rvd', staticmethod(unexpected_rvd_path))
    fsrc = _replica_lanes(ftensor, [0, 1])
    fdst = _replica_lanes(ftensor, consumer_ranks)
    bsrc = _replica_lanes(ftensor.grad, consumer_ranks)
    bdst = _replica_lanes(ftensor.grad, [0, 1])
    with pytest.raises(ValueError, match='non-identity PP boundary are not yet executable safely'):
        ConcurrentGener.gen(fsrc, fdst, bsrc, bdst)
