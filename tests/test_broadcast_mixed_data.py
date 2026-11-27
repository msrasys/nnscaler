#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.utils import _may_contain_tensor, broadcast_mixed_data
from tests.launch_torchrun import launch_torchrun


def test_may_contain_tensor():
    assert not _may_contain_tensor({'a': [1, 'x', None, (2.0, True)], 'b': {3: [4]}})
    assert _may_contain_tensor({'a': [{'b': torch.zeros(1)}]})
    assert _may_contain_tensor({torch.zeros(1): 'key'})            # dict key
    assert _may_contain_tensor(torch.Generator().manual_seed(1))   # pickled state
    assert _may_contain_tensor(lambda: 0)                          # unpicklable


def _broadcast_worker():
    torch.distributed.init_process_group('nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    shared = torch.arange(6.).view(2, 3)
    key_a, key_b = torch.empty(0), torch.empty(0)
    big = [{'i': i, 'l': [i, str(i), True]} for i in range(10000)]
    for device in ('cpu', None):
        data = {
            'tensors': [shared, {'same': shared, 'step': torch.tensor(7)}, shared[1]],
            'big': big,
            # distinct tensors with equal (data_ptr, shape, stride) as dict keys
            'slots': {key_a: 'a', key_b: 'b'},
            # objects that pickle their own tensors
            'rng': torch.Generator().manual_seed(123),
            'module': torch.nn.Linear(2, 1),
        } if rank == 0 else None
        out = broadcast_mixed_data(data, src_rank=0, device=device)

        want = torch.device('cpu') if device == 'cpu' else torch.device('cuda', rank)
        t = out['tensors']
        assert t[0].device == want and torch.equal(t[0].cpu(), shared)
        assert t[0] is t[1]['same']
        assert t[1]['step'].item() == 7 and torch.equal(t[2].cpu(), shared[1])
        assert out['big'] == big
        assert sorted(out['slots'].values()) == ['a', 'b']
        assert torch.equal(out['rng'].get_state(), torch.Generator().manual_seed(123).get_state())
        weight = out['module'].weight
        assert isinstance(weight, torch.nn.Parameter) and weight.requires_grad
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs 2 gpus')
def test_broadcast_mixed_data():
    launch_torchrun(2, _broadcast_worker)
