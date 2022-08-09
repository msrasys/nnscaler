"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/vision/swin/train.py
"""

import torch
from examples.vision.swin.model import Config, SwinTransformer, ImageDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary

import examples.vision.swin.policy.spmd as spmd

PAS = spmd.PASSingle


def train():

    batch_size = 1

    cfg = Config()
    model = SwinTransformer()
    dataloader = ImageDataLoader(batch_size, cfg.img_size, cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    model = cube.SemanticModel(model, dataloader.shapes)
    @cube.compile(model, dataloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        imgs, labels = next(dataloader)
        loss = model(imgs, labels)
        loss.backward()
    model = model.get_gen_module()

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 10, 2
    for step in range(iter_num):

        if step >= warmup:
            CudaTimer(enable=True).start('e2e')

        # training
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step >= warmup:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 4 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()
