import gc
import weakref

import torch

from nnscaler import ComputeConfig, parallelize
from nnscaler.policies import pas_dp


class _TemporaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8))

    def forward(self, x):
        return (x * self.weight).sum()


def test_temporary_model_is_released_before_partitioning(tmp_path):
    references = []

    def factory():
        module = _TemporaryModel()
        references.extend((weakref.ref(module), weakref.ref(module.weight)))
        return module

    def policy(graph, config):
        gc.collect()
        assert all(ref() is None for ref in references)
        return pas_dp(graph, config)

    parallelize(_TemporaryModel, {'x': torch.ones(8)}, policy,
                ComputeConfig(1, 1, use_end2end=True, trace_strategy='cpu'), module_fn=factory,
                gen_savedir=tmp_path, load_module=False)
    assert len(list(tmp_path.rglob('gencode0.py'))) == 1
    index_path, = tmp_path.rglob('fullmodel.pt.index')
    index = torch.load(index_path, weights_only=True)
    assert len(index) == 1
    tid, chunk = next(iter(index.items()))
    content = torch.load(index_path.with_name(f'fullmodel.pt.{chunk}'), weights_only=True)
    torch.testing.assert_close(content[tid], torch.ones(8))
