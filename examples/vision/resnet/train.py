"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/vision/resnet/train.py
"""

import torch
from examples.vision.resnet.model_alpa import WideResNet, ImageDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary



def train():

    batch_size = 32
    nmicros = 1536 // batch_size


    model = WideResNet()
    model = model.cuda()

    cnt = 0
    for param in model.parameters():
        cnt += param.nelement()
    print(f'param#: {cnt / 1e6} M')

    dataloader = ImageDataLoader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    print_each_rank('model weight consumpition:')
    memory_summary()

    def train_iter(model, dataloader):
        imgs, labels = next(dataloader)
        loss = model(imgs, labels)
        loss.backward()

    CudaTimer(enable=False).warmup()
    iter_num = 10
    for step in range(iter_num):

        # if step == 0:
        #     model_summary(model, next(dataloader))

        if step >= 4:
            CudaTimer(enable=True).start('e2e')

        # training
        for _ in range(nmicros):
            train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step >= 4:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-4, field_name='e2e')))
    memory_summary()

if __name__ == '__main__':

    cube.init()
    train()
