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
"""
import torch
import argparse
import time

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


kDataShapes = ([128, 1024],)


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
    
    dataloader = cube.runtime.syndata.SynDataLoader(1280, *kDataShapes)

    genfile = args.genfile.format(rank=torch.distributed.get_rank())
    model = load_module(genfile)
    train_fn = load_train_fn(genfile)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer().warmup()
    torch.distributed.barrier()
    with torch.profiler.profile() as prof:
        iter_num = args.iter_num
        for step in range(iter_num):
            if step >= 40:
                CudaTimer().start('e2e')
            train_fn(model, dataloader)
            optimizer.step()
            optimizer.zero_grad()
            if step >= 40:
                CudaTimer().stop('e2e')
            if (step + 1) % 20 == 0:
                print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
            time.sleep(0.05)

    prof.export_chrome_trace(f"trace{torch.distributed.get_rank()}.json")

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='inspect')
    parser.add_argument('--genfile', type=str,
                        default='gencode{rank}.py')
    parser.add_argument('--iter-num', type=int,
                        default=128)
    args = parser.parse_args()

    cube.init()
    train(args)
