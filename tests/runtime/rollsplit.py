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
            input, -(9 // 2), 1, group
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
            input, -(9 // 2), 1, group
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


if __name__ == '__main__':

    cube.init()
    test_roll_parallel()