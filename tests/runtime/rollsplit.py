"""
CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    tests/runtime/rollsplit.py

"""


import torch

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


def test_roll_parallel():

    # input size
    group = None
    world_size = torch.distributed.get_world_size(group=group)
    input_size = [1, 224 // world_size, 224, 256]
    # input_size = torch.arange(0, )
    input = torch.randn(input_size).cuda() * 10
    
    CudaTimer().warmup(seconds=2)

    torch.distributed.barrier()
    CudaTimer().start(field_name='roll_halo')
    for _ in range(1000):
        roll_out = cube.runtime.function.roll_dim_parallel(
            input, (9 // 2), 1, list(range(world_size)), group
        )
    CudaTimer().stop(field_name='roll_halo')
    ref1 = roll_out
    # print_each_rank(ref1, rank_only=0)
    assert roll_out.shape == input.shape
    span = CudaTimer().duration(times=1000, field_name='roll_halo')
    print_each_rank('span on halo exchange: {:.2f} ms'.format(span))


    torch.distributed.barrier()
    CudaTimer().start(field_name='roll_allgather')
    for _ in range(1000):
        roll_out = cube.runtime.function.roll_dim_allgather(
            input, (9 // 2), 1, group
        )
    CudaTimer().stop(field_name='roll_allgather')
    ref2 = roll_out
    # print_each_rank(ref2, rank_only=0)
    span = CudaTimer().duration(times=1000, field_name='roll_allgather')
    print_each_rank('span on allgather exchange: {:.2f} ms'.format(span))

    if not torch.allclose(ref1, ref2, atol=1e-3, rtol=1e-3):
        print('correctness test failed')
    else:
        print('correctness test passed')


def test_roll_grid_parallel():
        # input size
    group = None
    world_size = torch.distributed.get_world_size(group=group)
    myrank = torch.distributed.get_rank()
    assert world_size == 4
    # input_size = [1, 224 // 2, 224 // 2, 256]
    # input = torch.randn(input_size).cuda() * 10
    input = torch.arange(myrank * 4, (myrank + 1) * 4).view(1, 2, 2, 1).float()
    input = input.cuda()
    print_each_rank(f'input: {input.view(-1)}')

    CudaTimer().warmup(seconds=2)

    torch.distributed.barrier()
    CudaTimer().start(field_name='roll_halo')
    for _ in range(1):
        roll_out = cube.runtime.function.roll_grid_parallel(
            input, (9 // 2, 9 // 2), (1, 2), 2, 2, group
        )
    CudaTimer().stop(field_name='roll_halo')
    ref1 = roll_out
    # print_each_rank(ref1, rank_only=0)
    assert roll_out.shape == input.shape
    span = CudaTimer().duration(times=1000, field_name='roll_halo')
    print_each_rank('span on halo exchange: {:.2f} ms'.format(span))

    torch.distributed.barrier()
    CudaTimer().start(field_name='roll_allgather')
    for _ in range(1):
        roll_out = cube.runtime.function.roll_dim_allgather(
            input, (9 // 2), 1, group
        )
        roll_out = cube.runtime.function.roll_dim_allgather(
            roll_out, (9 // 2), 2, group
        )
    CudaTimer().stop(field_name='roll_allgather')
    ref2 = roll_out
    # print_each_rank(ref2, rank_only=0)
    span = CudaTimer().duration(times=1000, field_name='roll_allgather')
    print_each_rank('span on allgather exchange: {:.2f} ms'.format(span))


def test_roll_parallel_autograd():

    group = None
    world_size = torch.distributed.get_world_size(group=group)
    input_size = [1, 224 // world_size, 224, 256]
    # input_size = torch.arange(0, )
    input = torch.randn(input_size).cuda() * 10
    input = input.requires_grad_()

    out = cube.runtime.function.roll_dim_parallel(
        input, (9 // 2), 1, group
    )
    loss = torch.sum(out)
    loss.backward()
    print(loss)
    print(input.grad)


def test_grid_partition():

    group = None
    world_size = torch.distributed.get_world_size(group=group)
    assert world_size == 4
    input_size = [1, 56, 56, 256]
    input = torch.randn(input_size).cuda() * 10
    input = input.requires_grad_()
    out = cube.runtime.function.grid_partition(input, 2, 2, group = None)
    print(out.shape)
    assert out.shape == torch.Size([1, 56 // 2, 56 // 2, 256])
    loss = torch.sum(out)
    loss.backward()
    # print(input.grad)


def test_grid_collection():

    group = None
    world_size = torch.distributed.get_world_size(group=group)
    assert world_size == 4
    input_size = [1, 56 // 2, 56 // 2, 256]
    input = torch.randn(input_size).cuda() * 10
    input = input.requires_grad_()
    out = cube.runtime.function.grid_collection(input, 2, 2, group = None)
    assert out.shape == torch.Size([1, 56, 56, 256])
    loss = torch.sum(out)
    loss.backward()
    # print(input.grad)


if __name__ == '__main__':

    cube.init()
    test_roll_parallel()
    # test_roll_grid_parallel()
    # test_roll_parallel_autograd()
    # test_grid_partition()
    # test_grid_collection()