import time
import tempfile

import pytest
import torch

from cube.parallel import _gen_graph
from cube.ir.operator import IRFwOperation
from cube.profiler.database import CompProfiler, ProfileDataBase


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_op_profile_times():
    with tempfile.TemporaryDirectory() as tempdir:
        graph, _ = _gen_graph(NaiveFFN(), {'x': torch.randn(2, 128, 1024)}, tempdir, False)
        fc1, relu, fc2 = graph.select(ntype=IRFwOperation)
        fn, shapes, dtypes, requires_grads, values, kwargs = ProfileDataBase.get_func(fc1)
        tic = time.perf_counter()
        CompProfiler.profile(fc1, fn, shapes, dtypes, requires_grads, values, **kwargs)
        toc = time.perf_counter()
        # this is always true because the op is very small.
        assert toc - tic < 20, f'op profile time is too long {toc - tic}'
