import torch
from torch import nn

import cube
from cube.graph.adapter.adapter import IRAdapter
import cube.graph.parser as parser
from cube.logics.pool import SchedulePool
from cube.logics.translator import LogicTranslator
from cube.graph.adapter import AdapterGener
from cube.execplan import ExectuionPlan
from cube.execplan.planpass.grouping import Grouping

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        loss = torch.sum(output)
        return loss

model = MLP(dim=1024)
dataloader = cube.runtime.syndata.SynDataLoader(1280, [0], [128, 1024])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_iter(model, dataloader):
    data = next(dataloader)
    loss = model(data)
    loss.backward()


def test_p():
    SchedulePool().clear()

    graph = parser.convert_model(model, input_shapes=([128, 1024],))
    loader = parser.convert_dataloader(dataloader)

    train_iter(graph, loader)
    graph = LogicTranslator.gen_logic_graph()
    # print(graph.extra_repr())
    # assert False

    node1, node2, node3, node4 = graph.nodes()[1:5]
    for node in graph.nodes():
        graph.assign(node, rank=0)
    algo = node2.algorithms('column')
    subnodes = graph.partition(node2, algo, config=dict(chunk_num=4))
    for idx, subnode in enumerate(subnodes):
        graph.assign(subnode, rank=idx)

    # print(graph.extra_repr())

    graph = AdapterGener.gen(graph)
    # print(graph)
    # for node in graph.nodes():
    #     if isinstance(node, IRAdapter):
    #         print(node.extra_repr())
    
    execplan = ExectuionPlan(graph)
    # print(execplan)

    execplan = Grouping.apply(execplan)
    print(execplan)
    # print(execplan.graph.extra_repr())

    mcodegen = ModelCodeGen(execplan)
    tcodegen = ScheduleCodeGen(execplan)

    mcode = mcodegen.gen(device=0)
    tcode = tcodegen.gen(device=0)
    print(mcode)
    print(tcode)

    mcode = mcodegen.gen(device=1)
    tcode = tcodegen.gen(device=1)
    print(mcode)
    print(tcode)

    mcode = mcodegen.gen(device=2)
    tcode = tcodegen.gen(device=2)
    print(mcode)
    print(tcode)
    mcode = mcodegen.gen(device=3)
    tcode = tcodegen.gen(device=3)
    print(mcode)
    print(tcode)

    assert False
