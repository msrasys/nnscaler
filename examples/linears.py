"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/e2e.py
"""
from typing import List

import torch
from torch import nn

import cube
from cube.tschedule.su import ScheduleUnit
from cube.tschedule.suseq import SUSequence


def trans_policy(graph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    ndevice = resource.ngpus
    for node in graph.nodes():
        algorithm = node.algorithms('data_parallel')
        graph.select(node, algorithm, config=dict(chunk_size=ndevice))
    return graph


def schedule_policy(seq: SUSequence, resource):
    """
    The schedule policy uses 1F1B (interleaved) pipeline
    """
    ndevice = resource.ngpus

    # batch_seqs[idx]: the idx-th forward-backward 4 linear forward + backward
    batch_seqs: List[List[ScheduleUnit]] = group_by_batches(seq.sus())
    num_fsus = len(seq.sus()) // len(batch_seqs) // 2

    # assign devices -- intra device order
    for batch_seq in batch_seqs:
        for idx, su in enumerate(batch_seq):
            stage = idx // (num_fsus // ndevice)
            if idx < num_fsus:
                seq.assign(su, stage)
            else:
                seq.assign(su, ndevice - stage % ndevice)

    
    # assign devices -- inter device order
    f = lambda stage, micro_batch_id: batch_seqs[micro_batch_id][stage]
    b = lambda stage, micro_batch_id: batch_seqs[micro_batch_id][-stage]
    
    reorder = list()
    # warmup
    for stage in range(ndevice):
        for micro_batch_id in range(stage):
            reorder = reorder.append(f(stage, micro_batch_id))
    # steady + cooldown
    for stage in range(ndevice):
        # backward
        for micro_batch_id in range(len(batch_seqs)):
            reorder.append(b(stage, micro_batch_id))
        # forward
        for stage in range(ndevice):
            f_mirco_batch_id = micro_batch_id + 1 + ndevice - stage
            if f_mirco_batch_id >= len(batch_seqs):
                continue
            reorder.append(f(stage, f_mirco_batch_id))
    
    for idx, su in enumerate(reorder):
        seq.move(su, idx)





class FakeDataLoader:
    def __init__(self, shape, num=640):
        self.shape = shape
        self.length = num
        self.pos = 0
    def __iter__(self):
        self.pos = 0
        return self
    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration 
        return torch.randn(self.shape).cuda()


class MLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult, bias=False)
        self.linear4 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        return output


def train():
    batch_size = 64
    dim = 1024

    model = MLP(dim=dim)
    model = model.cuda()

    dataloader = FakeDataLoader((batch_size, dim))

    def train_iter(model, dataloader):
        for _ in range(4):
            data = next(dataloader)
            output = model(data)
            loss = torch.sum(output) / 1000
            print(f'loss={loss.item()}')
            loss.backward()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':

    train()