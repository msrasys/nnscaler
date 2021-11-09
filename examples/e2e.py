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

import torch
from torch import nn

import cube
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph


def trans_policy(ir_graph, resource):
    return ir_graph

def schedule_policy(sugraph: SUGraph, resource):
    # put to micro-batch forward-backward sequence
    fb_op_seqs = list()
    for fsu in sugraph.fsus():
        for fb_seq in fb_op_seqs:
            for ksu in fb_seq[::-1]:
                if sugraph.happen_before(ksu, fsu):
                    fb_seq.append(fsu)
                    break
            else:
                continue
            break
        else:
            fb_op_seqs.append([fsu])
    
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
    
    print(f'> collect {len(fb_op_seqs)} forward-backward sequence')
    for fb_sus in fb_op_seqs:
        for idx, su in enumerate(fb_sus):
            if idx < 3:
                sugraph.assign(su, 0)
                sugraph.assign(su.mirror, 0)
            else:
                sugraph.assign(su, 1)
                sugraph.assign(su.mirror, 1)
    return sugraph


class FakeDataLoader:
    def __init__(self, batch_size, num=640):
        self.batch_size = batch_size
        self.length = num
        self.pos = 0
    def __iter__(self):
        self.pos = 0
        return self
    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration 
        return torch.randn((self.batch_size, 1024)).cuda()


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, data):
        output = self.linear1(data)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.classifier(output)
        loss = torch.sum(output)
        return loss

def init_weight(parameters):
    for param in parameters:
        with torch.no_grad():
            torch.nn.init.uniform_(param)


def train():
    batch_size = 64

    model = FeedForward(dim=1024)
    model = cube.schedule.SemanticModel(
        model, input_shapes=([batch_size,1024],),
    )

    dataloader = FakeDataLoader(batch_size)

    @cube.schedule.schedule(model, dataloader, transform_policy=trans_policy, schedule_policy=schedule_policy)
    def train_iter(model, dataloader):
        # for _ in range(1):
        #     data = next(dataloader)
        #     loss = model(data)
        #     loss.backward()
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()

    init_weight(model.parameters())
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':

    cube.DeviceGroup()
    train()
