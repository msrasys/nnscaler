"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    examples/nlp/gpt/train.py --fp16
"""


import torch
import time

from examples.nlp.gpt.model import GPT
from examples.nlp.gpt.model import GPTDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary

from examples.nlp.gpt.policy.mpmd import PASMegatron as PAS
import examples.nlp.gpt.policy.spmd as spmd
import examples.nlp.gpt.policy.mpmd as mpmd

import argparse

from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.profiler.database import ProfileDataBase
from cube.algorithm.ops.dimops import gen_partitions
from cube.graph.function.anchor import IRGraphAnchor

parser = argparse.ArgumentParser(description='GPT Train')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
args = parser.parse_args()

cube.init()


def train():
    batch_size = 1

    model = GPT()
    model = model if not args.fp16 else model.half()
    dataloader = GPTDataLoader(batch_size)

    model = cube.SemanticModel(model, dataloader.shapes)

    def profile(graph, resource):
        db = ProfileDataBase()
        mem_sum = 0
        for node in graph.select(ntype=IRFwOperation):
            if isinstance(node, IRGraphAnchor):
                continue
            partition_nodes = gen_partitions(node, 1)
            for partition_node in partition_nodes:
                fw_span, bw_span, infer_mem, train_mem = db.profile(partition_node)
                mem_sum = mem_sum + train_mem
        db.dump('db.json', override=True)
        print('estimated train mem: ', mem_sum / 1024 / 1024 / 1024)

        for node in graph.nodes():
            if not isinstance(node, IRBpOperation):
                graph.assign(node, 0)
        return graph

    @cube.compile(model, dataloader, PAS=profile, override=True)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        loss.backward()
    model = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    # CudaTimer(enable=False).warmup()
    iter_num = 4
    warmup = 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')

        train_iter(model, dataloader)
        memory_summary()
        optimizer.step()
        memory_summary()
        optimizer.zero_grad()
        memory_summary()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)

    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()