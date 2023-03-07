"""
intra-primitives:
    torchrun --nproc_per_node=2 tests/runtime/test_runtime_collectives.py

inter-primitives:
    torchrun --nproc_per_node=3 tests/runtime/test_runtime_collectives.py
"""

from typing import List

import cube
import torch



cube.init()

mydevice = torch.cuda.current_device()
myrank = torch.distributed.get_rank()
ndevices = torch.distributed.get_world_size()


def _get_tensor(shape: List[int], dtype: torch.dtype = torch.float32, rank=myrank) -> torch.Tensor:
    global mydevice, myrank
    tensor = torch.ones(shape, dtype=dtype, device=mydevice)
    tensor = tensor * rank
    return tensor


def test_runtime_move():
    assert ndevices == 2
    shape = [128, 256]

    # synchronize
    tensor = _get_tensor(shape)
    res = _get_tensor(shape, rank=0)
    tensor = cube.runtime.adapter.move(tensor, shape, torch.float32, 0, 1)

    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"
    
    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.move(tensor, shape, torch.float32, 0, 1, async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass move')


def test_runtime_allreduce():
    assert ndevices == 2
    shape = [128, 256]

    # synchronize
    tensor = _get_tensor(shape)
    cube.runtime.adapter.all_reduce(tensor, [0, 1])
    res =  _get_tensor(shape, rank=0) + _get_tensor(shape, rank=1)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.all_reduce(tensor, [0, 1], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass allreduce')


def test_runtime_allgather():
    assert ndevices == 2
    shape = [128, 256]

    # synchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.all_gather(tensor, 0, [0, 1])
    res = torch.concat([_get_tensor(shape, rank=0), _get_tensor(shape, rank=1)], dim=0)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.all_gather(tensor, 0, [0, 1], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass allgather')


def test_runtime_reduce_scatter():
    assert ndevices == 2
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = _get_tensor(shape, rank=0) + _get_tensor(shape, rank=1)
    res = res.chunk(2, dim=0)[myrank]

    # synchronize
    tensor = cube.runtime.adapter.reduce_scatter(tensor, 0, [0, 1])
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.reduce_scatter(tensor, 0, [0, 1], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass reduce scatter')


def test_runtime_all2all():
    assert ndevices == 2
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = torch.concat([_get_tensor(shape, rank=0), _get_tensor(shape, rank=1)], dim=0)
    res = res.chunk(2, dim=1)[myrank]

    # synchronize
    tensor = cube.runtime.adapter.all_to_all(tensor, 0, 1, [0, 1])
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.all_to_all(tensor, 0, 1, [0, 1], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass all2all')


def test_runtime_exchange():
    assert ndevices == 2
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = _get_tensor(shape, rank=(myrank + 1) % 2)

    tensor = cube.runtime.adapter.exchange(tensor, [0, 1])
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.exchange(tensor, [0, 1], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass exchange')


def test_runtime_rdscatter():
    assert ndevices == 3
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = _get_tensor(shape, rank=0).chunk(ndevices-1, dim=0)[myrank-1]

    # synchronize
    tensor = cube.runtime.adapter.rdscatter(
        tensor, shape, torch.float32, dim=0, src=0, dsts=[1,2])
    if myrank > 0:
        assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # synchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.rdscatter(
        tensor, shape, torch.float32, dim=0, src=0, dsts=[1,2], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    if myrank > 0:
        assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass rdscatter')


def test_runtime_rdgather():
    assert ndevices == 3
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = torch.cat((_get_tensor(shape, rank=1), _get_tensor(shape, rank=2)), dim=0)

    # synchronize
    tensor = cube.runtime.adapter.rdgather(
        tensor, shape, torch.float32, dim=0, srcs=[1,2], dst=0)
    if myrank == 0:
        assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.rdgather(
        tensor, shape, torch.float32, dim=0, srcs=[1,2], dst=0, async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    if myrank == 0:
        assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass rdgather')


def test_runtime_broadcast():
    assert ndevices == 3
    shape = [128, 256]

    tensor = _get_tensor(shape)
    res = _get_tensor(shape, rank=0)

    # synchronize
    tensor = cube.runtime.adapter.broadcast(
        tensor, shape, torch.float32, src=0, ranks=[0,1,2])
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    # asynchronize
    tensor = _get_tensor(shape)
    tensor = cube.runtime.adapter.broadcast(
        tensor, shape, torch.float32, src=0, ranks=[0,1,2], async_op=True)
    tensor = cube.runtime.executor.AsyncCommHandler().wait(tensor)
    assert torch.allclose(tensor, res), f"mismatch rank{myrank}: {tensor[0,0]} vs. {res[0, 0]}"

    print(f'rank[{myrank}]: pass broadcast')


if __name__ == '__main__':

    if ndevices == 2:
        test_runtime_move()
        test_runtime_allreduce()
        test_runtime_allgather()
        test_runtime_reduce_scatter()
        test_runtime_all2all()
        test_runtime_exchange()

    if ndevices == 3:
        test_runtime_rdscatter()
        test_runtime_rdgather()
        test_runtime_broadcast()
