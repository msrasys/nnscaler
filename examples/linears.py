"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    examples/linears.py
"""

import torch
from torch import nn

import cube
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph


def transform_policy(graph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    from cube.graph.operator.operator import IRDataOperation, IRFwOperation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation) or isinstance(node, IRFwOperation):
            algo = node.algorithms('data')
            assert algo is not None
            graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy
    """
    fb_seqs = list()
    for fsu in sugraph.fsus():
        for fb_seq in fb_seqs:
            for ksu in fb_seq[::-1]:
                if sugraph.happen_before(ksu, fsu):
                    fb_seq.append(fsu)
                    break
            else:
                continue
            break
        else:
            fb_seqs.append([fsu])
    
    # device assignment
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
    
    print(f'> collect {len(fb_seqs)} forward-backward sequence')
    for fb_seq in fb_seqs:
        for idx, su in enumerate(fb_seq):
            devid = idx % resource.ngpus
            sugraph.assign(su, devid)
            sugraph.assign(su.mirror, devid)

    # set partial order
    for fb_seq in fb_seqs:
        fb_seq += [fsu.mirror for fsu in fb_seq][::-1]

    seqs = list()
    for fb_seq in fb_seqs:
        seqs += fb_seq
    sugraph.partial_set_order(seqs)
    return sugraph


# =================== Semantic Model Description ====================

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
        loss = torch.sum(output)
        return loss


def train():
    batch_size = 64
    dim = 1024

    model = MLP(dim=dim)
    model = cube.schedule.SemanticModel(
        model, input_shapes=([batch_size, dim],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(640, [batch_size, dim])

    @cube.schedule.schedule(model, dataloader, transform_policy=transform_policy, schedule_policy=schedule_policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        # print(f'loss={loss.item()}')
        loss.backward()
    model = model.get_gen_module()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':

    cube.init()
    train()