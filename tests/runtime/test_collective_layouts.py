#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.adapter import prim as primitives
from nnscaler.runtime.adapter import collectives, nn
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.executor import AsyncCommHandler
from tests.launch_torchrun import torchrun
from tests.runtime.test_runtime_collectives import _init_distributed


def _placed_shards(full, dim, ranks):
    tensors = []
    width = full.shape[dim] // len(ranks)
    for index, rank in enumerate(ranks):
        slices = [(0, size) for size in full.shape]
        slices[dim] = (index * width, (index + 1) * width)
        tensors.append(IR.set_object_device(full.select(tuple(slices), (0, 1)), rank))
    return tensors


def _alltoall_layout_worker(subgroup, different_layout):
    _init_distributed(4)
    rank = torch.distributed.get_rank()
    ranks = (3, 0, 2) if subgroup else (2, 0, 3, 1)
    dsts = ((2, 3, 0) if subgroup else (1, 3, 0, 2)) if different_layout else ranks
    # All ranks initialize a subgroup, including nonmembers.
    DeviceGroup().get_group(sorted(ranks))
    if rank in ranks:
        count = len(ranks)
        full_value = torch.arange(6 * count * count, dtype=torch.float64).reshape(2 * count, 3 * count)
        full = IRFullTensor(full_value.shape, dtype=torch.float64, requires_grad=True)
        for idim, odim in ((0, 1), (1, 0)):
            prim = primitives.AllToAllPrim(
                _placed_shards(full, idim, ranks), _placed_shards(full, odim, dsts), idim, odim,
            )
            local = full_value.chunk(count, dim=idim)[ranks.index(rank)].detach()
            expected = full_value.chunk(count, dim=odim)[dsts.index(rank)]
            for operation in (collectives.all_to_all, collectives.all_to_all_single):
                for async_op in (False, True):
                    actual = operation(local, **prim.kwargs, async_op=async_op)
                    if async_op:
                        actual = AsyncCommHandler().wait(actual)
                    torch.testing.assert_close(actual, expected)
                    AsyncCommHandler().check_clear()

            reference = full_value.detach().clone().requires_grad_()
            gradients = [
                (torch.arange(part.numel(), dtype=part.dtype).reshape(part.shape) + destination * 100)
                .T.contiguous().T
                for destination, part in zip(dsts, reference.chunk(count, dim=odim))
            ]
            loss = sum((part * grad).sum() for part, grad in zip(reference.chunk(count, dim=odim), gradients))
            loss.backward()
            for operation in (nn.AllToAllAllToAll.apply, nn.alltoall_alltoall):
                value = local.clone().requires_grad_()
                # Construct arguments from IR metadata, so the old implementation
                # fails numerically instead of merely lacking a new keyword.
                args = [value, idim, odim, prim.kwargs['ranks']]
                if 'dsts' in prim.kwargs:
                    args.append(prim.kwargs['dsts'])
                actual = operation(*args)
                torch.testing.assert_close(actual, expected)
                actual.backward(gradients[dsts.index(rank)])
                torch.testing.assert_close(value.grad, reference.grad.chunk(count, dim=idim)[ranks.index(rank)])
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('subgroup', [False, True])
@pytest.mark.parametrize('different_layout', [False, True])
def test_alltoall_layouts_and_backward(subgroup, different_layout):
    torchrun(4, _alltoall_layout_worker, subgroup, different_layout)


def _rooted_layout_worker(async_op):
    _init_distributed(4)
    rank = torch.distributed.get_rank()
    order = (3, 1, 2)
    group_order = (2, 0, 3, 1)
    full = IRFullTensor((12, 6), dtype=torch.float64)
    root = IR.set_object_device(full.tosub(), 0)
    shards = _placed_shards(full, 0, order)
    values = [IR.set_object_device(full.select(((0, 12), (0, 6)), (index, 3)), device)
              for index, device in enumerate(order)]
    cases = [
        (primitives.RDScatterPrim([root], shards, 0), collectives.rdscatter),
        (primitives.RDGatherPrim(shards, [root], 0), collectives.rdgather),
        (primitives.RVScatterPrim([root], values), collectives.rvscatter),
        (primitives.RVGatherPrim(values, [root]), collectives.rvgather),
        (primitives.BroadcastPrim([root], [root] + [IR.set_object_device(full.tosub(), r) for r in order]),
         collectives.broadcast),
    ]
    base = torch.arange(72, dtype=torch.float64).reshape(12, 6)
    for prim, operation in cases:
        local = prim.dispatch(rank)
        kwargs = {**local.kwargs, 'dtype': torch.float64}
        if operation is collectives.rdgather:
            value = None if rank == 0 else torch.arange(24, dtype=torch.float64).reshape(4, 6) + rank * 100
        else:
            value = base + rank * 100 if local.inputs() else None
        actual = operation(value, **kwargs, async_op=async_op)
        if async_op and local.outputs():
            actual = AsyncCommHandler().wait(actual)
        if local.outputs():
            if operation is collectives.rdscatter:
                expected = base.chunk(3, dim=0)[order.index(rank)]
            elif operation is collectives.rdgather:
                expected = torch.cat([torch.arange(24, dtype=torch.float64).reshape(4, 6) + source * 100 for source in order])
            elif operation is collectives.rvscatter:
                expected = base / 3
            elif operation is collectives.rvgather:
                expected = sum(base + source * 100 for source in order)
            else:
                expected = base
            torch.testing.assert_close(actual, expected)
        AsyncCommHandler().drain_sends()
        AsyncCommHandler().check_clear()

    # These operations use global endpoints or return identical replicas;
    # their results do not depend on the group's logical rank order.
    actual = collectives.all_reduce(base + rank * 100, group_order, async_op=async_op)
    if async_op:
        actual = AsyncCommHandler().wait(actual)
    torch.testing.assert_close(actual, base * 4 + 600)
    for src, dst in ((3, 1), (1, 3)):
        if rank in (src, dst):
            actual = collectives.move(base + src * 100 if rank == src else None,
                                      base.shape, base.dtype, src, dst, async_op=async_op)
            if rank == dst:
                if async_op:
                    actual = AsyncCommHandler().wait(actual)
                torch.testing.assert_close(actual, base + src * 100)
            AsyncCommHandler().drain_sends()
            obj = collectives.move_object({'source': src} if rank == src else None, src, dst)
            assert obj == {'source': src}
        torch.distributed.barrier()
    obj = collectives.broadcast_object({'source': 3} if rank == 3 else None, 3, group_order)
    assert obj == {'source': 3}
    AsyncCommHandler().check_clear()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('async_op', [False, True])
def test_rooted_and_order_invariant_collective_layouts(async_op):
    torchrun(4, _rooted_layout_worker, async_op)


def _autograd_layout_worker():
    _init_distributed(4)
    rank = torch.distributed.get_rank()
    ranks = (2, 0, 3, 1)
    index = ranks.index(rank)
    base = torch.arange(8, dtype=torch.float64)
    for operation, factor in ((nn.allreduce_identity, 1), (nn.identity_allreduce, 4), (nn.allreduce_allreduce, 4)):
        value = (base + rank * 10).requires_grad_()
        actual = operation(value, ranks)
        expected = base + rank * 10 if operation is nn.identity_allreduce else base * 4 + 60
        torch.testing.assert_close(actual, expected)
        actual.backward(torch.ones_like(actual))
        torch.testing.assert_close(value.grad, torch.full_like(value, factor))
    for operation, factor in ((nn.allgather_split, 1), (nn.allgather_reducescatter, 4)):
        value = (base + rank * 10).requires_grad_()
        actual = operation(value, 0, ranks)
        torch.testing.assert_close(actual, torch.cat([base + source * 10 for source in ranks]))
        grad = torch.arange(32, dtype=torch.float64)
        actual.backward(grad)
        torch.testing.assert_close(value.grad, grad.chunk(4)[index] * factor)
    for operation in (nn.split_allgather, nn.reducescatter_allgather):
        value = torch.arange(32, dtype=torch.float64).requires_grad_()
        actual = operation(value, 0, ranks)
        expected = value.detach().chunk(4)[index]
        torch.testing.assert_close(actual, expected * (4 if operation is nn.reducescatter_allgather else 1))
        actual.backward(base + rank * 10)
        torch.testing.assert_close(value.grad, torch.cat([base + source * 10 for source in ranks]))
    for operation in (nn.ReduceBroadcast.apply, nn.BroadcastReduce.apply):
        value = (base + rank * 10).requires_grad_()
        actual = operation(value, 3, ranks)
        if operation == nn.ReduceBroadcast.apply:
            if rank == 3:
                torch.testing.assert_close(actual, base * 4 + 60)
            actual.backward(torch.full_like(actual, rank + 1))
            torch.testing.assert_close(value.grad, torch.full_like(value, 4))
        else:
            torch.testing.assert_close(actual, base + 30)
            actual.backward(torch.full_like(actual, rank + 1))
            if rank == 3:
                torch.testing.assert_close(value.grad, torch.full_like(value, 10))


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
def test_other_autograd_collectives_respect_layouts():
    torchrun(4, _autograd_layout_worker)
