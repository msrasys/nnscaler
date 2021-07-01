import torch
import time
import argparse


def keep(rank, args):

    torch.cuda.set_device(rank)
    a = torch.rand((8192, 8192)).cuda()
    b = torch.rand((8192, 8192)).cuda()

    while True:
        tic = time.time()
        for _ in range(5000):
            c = a * b
        torch.cuda.synchronize()
        toc = time.time()
        if rank == 0:
            print('time span: {}s'.format(toc - tic))
        time.sleep(args.interval)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    torch.multiprocessing.spawn(keep, args=(args,), nprocs=args.gpus, join=True)
