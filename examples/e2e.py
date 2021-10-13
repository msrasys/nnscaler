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
from cube.tschedule.su import SUType


def spolicy(ir_graph):
    for nid, node in enumerate(ir_graph.nodes()):
        if nid < 3:
            node.device = 0 
        else:
            node.device = 1
    return ir_graph

def tpolicy(seq):
    # put to micro-batch forward-backward sequence
    fb_op_seqs = list()
    for su in seq.sus():
        for fb_seq in fb_op_seqs:
            for ksu in fb_seq[::-1]:
                if seq.happen_before(ksu, su):
                    fb_seq.append(su)
                    break
            else:
                continue
            break
        else:
            fb_op_seqs.append([su])
    
    # merge to stages
    for fb_seq in fb_op_seqs:
        merged_su = fb_seq[0]
        for su in fb_seq[1:]:
            if su.stype == SUType.Backward:
                continue
            msu = seq.merge(merged_su, su)
            merged_su = su if msu is None else msu
    print(seq)
    return seq



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
    model = cube.sschedule.schedule(
        model, input_shapes=([batch_size,1024],),
        policy_fn=spolicy
    )

    dataloader = FakeDataLoader(batch_size)

    @cube.tschedule.schedule(model, dataloader, policy_fn=tpolicy)
    def train_iter(model, dataloader):
        for _ in range(4):
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
