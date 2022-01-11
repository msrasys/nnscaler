"""
Directly loading generated file for training

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/inspector.py

OMP_NUM_THREADS=4 torchrun --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/inspector.py
"""
import torch
import argparse
import time

import cube
from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank

# gpt
# L, N, E = (512, 8, 3072)
# kBatchDims = (0, 0)
# kDataShapes = ([N, L], [N, L])
# kDTypes = (torch.float, torch.long)

# mlp
kBatchDims = (0,)
kDataShapes = ([8192, 8192],)
kDTypes = (torch.float,)

# transformer
# kBatchDims  = (1, )
# kDataShapes = ([512, 4, 3072],)
# kDTypes = (torch.float,)


def load_module(filename: str):
    import importlib.util
    rank = torch.distributed.get_rank()
    print(f'> [{rank}] loading generated spatial moduel from {filename}')
    spec = importlib.util.spec_from_file_location("GenModel", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_module = module.GenModel().cuda()
    # sync parameters before start training
    loaded_module.sync_params()
    return loaded_module


def load_train_fn(filename: str):
    import importlib.util
    rank = torch.distributed.get_rank()
    print(f'> [{rank}] loading generated schedule from {filename} ...')
    spec = importlib.util.spec_from_file_location(
        "_train_step", filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._train_step


def train(args):
    global kDataShapes
    global kDTypes
    global kBatchDims
    dataloader = cube.runtime.syndata.SynDataLoader(
        kDataShapes, kDTypes, kBatchDims
    )

    genfile = args.genfile.format(rank=torch.distributed.get_rank())
    model = load_module(genfile)
    train_fn = load_train_fn(genfile)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    iter_num = args.iter_num

    def train_iters():
        for step in range(iter_num):
            if step >= 40:
                CudaTimer(enable=True).start('e2e')
            train_fn(model, dataloader)
            optimizer.step()
            optimizer.zero_grad()
            if step == 1:
                print('passed 1 iteration')
            if step >= 40:
                CudaTimer().stop('e2e')
            if (step + 1) % 20 == 0:
                print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
            time.sleep(0.05)

    if args.profile:
        with torch.profiler.profile() as prof:
            train_iters()
        prof.export_chrome_trace(f"trace{torch.distributed.get_rank()}.json")
    else:
        train_iters()

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-40)
    memory_summary()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='inspect')
    parser.add_argument('--genfile', type=str,
                        default='gencode{rank}.py')
    parser.add_argument('--iter-num', type=int,
                        default=128)
    parser.add_argument('--profile', dest='profile', action='store_true')
    args = parser.parse_args()

    cube.init()
    train(args)
