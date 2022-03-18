"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/nlp/mbart/train.py
"""


import torch

from examples.nlp.mbart.model import MBartForSentenceClassification
from examples.nlp.mbart.model import MBartDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary


def train():

    batch_size = 1

    model = MBartForSentenceClassification().cuda()
    dataloader = MBartDataLoader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    print_each_rank('model weight consumpition:')
    memory_summary()

    def train_iter(model, dataloader):
        input_ids, decoder_input_ids, labels = next(dataloader)
        loss = model(input_ids, decoder_input_ids, labels)
        loss.backward()

    CudaTimer(enable=False).warmup()
    iter_num = 64
    for step in range(iter_num):

        if step == 0:
            model_summary(model, next(dataloader))

        if step >= 20:
            CudaTimer(enable=True).start('e2e')

        # training
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step >= 20:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))
    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()