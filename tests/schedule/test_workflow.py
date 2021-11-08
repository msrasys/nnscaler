import torch
from torch import nn

import cube
from cube.graph.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.pool import SchedulePool
from cube.schedule.sugraph import SUGraphGener
from cube.schedule.translator import IRDataLoader


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
        return torch.randn(self.shape)


def test_semantic_model():
    dim = 1024
    model = MLP(dim=dim)
    model = cube.schedule.SemanticModel(
        model,
        input_shapes=([64, dim],)
    )
    assert isinstance(model.ir_graph, IRGraph)
    assert model._loaded_module is None


def test_schedule():

    SchedulePool().clear()

    dim = 1024
    batch_size = 64

    model = MLP(dim=dim)
    model = cube.schedule.SemanticModel(
        model,
        input_shapes=([batch_size, dim],)
    )

    dataloader = FakeDataLoader((batch_size, dim))
    dataloader = IRDataLoader(dataloader)

    def policy(sugraph, resources):
        # dataloader
        sugraph.assign(sugraph.sus(0), 0)

        fsus = [su for su in sugraph.sus() if su.stype == SUType.Forward]
        for idx, fsu in enumerate(fsus):
            bsu = fsu.mirror
            if idx < 2:
                sugraph.assign(fsu, 0)
                sugraph.assign(bsu, 0)
            else:
                sugraph.assign(fsu, 1)
                sugraph.assign(bsu, 1)
        return sugraph

    def train_iter(model, dataloader):
        num_micro_batch = 1
        for _ in range(num_micro_batch):
            data = next(dataloader)
            output = model(data)
            output.backward()

    train_iter(model, dataloader)

    nodes = SchedulePool().nodes()
    graph = IRGraph(nodes, None, None, 'testmodel')
    print(graph)

    sugraph = SUGraphGener.gen_sugraph(nodes)

    sugraph = policy(sugraph, None)
    print(sugraph)

    assert len(sugraph.sus()) == 33
