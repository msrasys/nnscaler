"""Dynamic dictionary updates must survive FX DCE and IR constant folding."""

import pytest
import torch

from nnscaler import ComputeConfig, parallelize
from nnscaler.parallel import _load_parallel_module_class
from ..utils import replace_all_device_with, new_empty


class UpdatedKwargs(torch.nn.Module):
    def forward(self, sample):
        kwargs = dict(scale=2, bias=0)
        before = sample['x'] * kwargs['scale']
        result = kwargs.update(sample['settings'], bias=sample['bias'])
        assert result is None
        return before + sample['x'] * kwargs['scale'] + kwargs['bias']


class KeywordUpdate(torch.nn.Module):
    def forward(self, sample):
        kwargs = dict(causal=True)
        kwargs.update(max_seqlen_q=sample['limit'])
        return sample['x'] + kwargs['max_seqlen_q']


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('model_type', [UpdatedKwargs, KeywordUpdate])
def test_generated_dict_updates_follow_runtime_values(tmp_path, model_type):
    model = model_type()
    def sample(scale, bias, limit):
        return dict(x=torch.ones(2, 4), settings=dict(scale=scale, bias=99), bias=bias, limit=limit)
    parallelize(model, {'sample': sample(3, 5, 8)}, 'dp', ComputeConfig(1, 1, constant_folding=True),
                gen_savedir=tmp_path, load_module=False)
    cls = _load_parallel_module_class(model_type, gen_savedir=tmp_path, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    for data in [sample(7, 11, 16), sample(13, 17, 32)]:
        torch.testing.assert_close(generated(data), model(data))
