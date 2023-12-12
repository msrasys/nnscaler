import tempfile
import torch
import os
from cube.parallel import _gen_graph
from cube.ir.operator import IRFwOperation
from cube.algorithm.ops.dimops import gen_partitions

class NaiveFFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 4096, bias=False)
        self.linear2 = torch.nn.Linear(4096, 1024, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def test_gen_partitions():
    with tempfile.TemporaryDirectory() as tempdir:
        graph, _ = _gen_graph(NaiveFFN(), {'x': torch.randn(2, 128, 1024)}, tempdir, False)
        fc1, relu, fc2 = graph.select(ntype=IRFwOperation)
        assert len(gen_partitions(fc1, 1)) == 1
        # C(4, 1) + 1 = 5
        assert len(gen_partitions(fc1, 2)) == 5
        # C(4, 2) + 2 * C(4, 1) + 1 - 1 = 14
        assert len(gen_partitions(fc1, 4)) == 14
        # C(4, 1) + 1 - 1 = 4
        assert len(gen_partitions(fc1, 4, base=4, depth=1)) == 4
