
"""

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    tests/test_nccl.py

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_IP}" \
    --master_port=${MASTER_PORT} \
    tests/test_nccl.py

OMP_NUM_THREADS=4 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_IP}" \
    --master_port=${MASTER_PORT} \
    tests/test_nccl.py

"""

import torch
import time
import sys
import os
import argparse


def print_each_rank(msg, select=True, outfile=''):
    myrank = torch.distributed.get_rank()
    outfile = sys.stdout if outfile == '' else outfile
    for rank in range(torch.distributed.get_world_size()):
        if select:
            if myrank == rank:
                f = open(outfile, 'a') if outfile != sys.stdout else sys.stdout
                f.write('rank [{}]: {}\n'.format(rank, msg))
                if outfile != sys.stdout:
                    f.close()
        torch.distributed.barrier()


def test_nccl(size, local_rank):
    msg = torch.ones((size,)).cuda()
    # warm up
    for _ in range(20):
        out = torch.distributed.all_reduce(msg)
        torch.cuda.synchronize()
    # profile
    tic = time.perf_counter()
    for _ in range(100):
        out = torch.distributed.all_reduce(msg)
    torch.cuda.synchronize()
    toc = time.perf_counter()

    span = (toc - tic) * 1000 / 100  # in ms
    bandwidth = size / span / 1e6 # in GB/s
    print_each_rank(
        'NCCL Allreduce | Msg Size: {:.0f} MB | Algo Bandwidth: {:.2f} GB/s'.format(
            size / 1024 / 1024, bandwidth),
        select=(local_rank==0),
    )

def test_allgather(size, local_rank):
    msg = torch.ones((size,)).cuda()
    tensor_list = [torch.empty_like(msg) for _ in range(torch.distributed.get_world_size())]

    tic = time.perf_counter()
    for _ in range(100):
        out = torch.distributed.all_gather(tensor_list, msg)
    torch.cuda.synchronize()
    print_each_rank('Passed all-gather')
    toc = time.perf_counter()


def benchmark(args):
    size = args.begin
    while size <= args.end:
        # test_allgather(size * 1024 * 1024, args.local_rank)
        test_nccl(size * 1024 * 1024, args.local_rank)  # MB to B
        size *= 2
    print_each_rank('test on nccl is done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--begin', type=int, default=4,
                        help='start message size in MB')
    parser.add_argument('--end', type=int, default=64,
                        help='end message size in MB')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl')
    print(f'{torch.distributed.get_rank()} launches')

    args.local_rank = int(os.environ.get('LOCAL_RANK'))
    torch.cuda.set_device(args.local_rank)
    benchmark(args)
