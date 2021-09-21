import cube.graph as cgraph
from cube.codegen.codegen import SScheduleCodeGen

import torch
from torch import nn


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
        output = output + data
        output = self.classifier(output)
        return output


model = FeedForward(dim=1024)


def import_from_file(filename):
    print(f'> loading GenModel from {filename} ...')
    import importlib.util
    spec = importlib.util.spec_from_file_location("GenModel", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenModel


def init_weight(parameters):
    for param in parameters:
        with torch.no_grad():
            torch.nn.init.uniform_(param)


def test_codegen(model):
    graph = cgraph.parser.convert(model,
                           input_shapes=([1024,1024],))
    for node in graph.nodes():
        node.device = 0
    gener = SScheduleCodeGen(graph, device=0)
    code = gener.gen(outfile='code.py')
    
    # execute
    print("> ===== Generated Code =====")
    print(code)
    print("< ===== Generated Code =====")
    
    GenModel = import_from_file('code.py')
    model = GenModel().cuda()

    init_weight(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("> training 10 iterations...")

    for _ in range(10):
        data = torch.randn([64,1024], device=torch.device('cuda:0'))
        out = model(data)
        loss = torch.mean(out) / 1000
        print(f'> loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':

    test_codegen(model)