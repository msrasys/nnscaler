#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import OrderedDict

import pytest
import torch

from nnscaler.utils import (
    _BROADCAST_SPLIT_DEPTH, _fill_tensor_subtrees, _may_contain_tensor, _split_tensor_subtrees,
    broadcast_mixed_data, extract_tensors, refill_tensors,
)
from tests.launch_torchrun import launch_torchrun


class StateList(list):
    """A list subclass whose pickled form does not hold the tensors it contains."""
    def __reduce__(self):
        return (StateList, ([x.tolist() if isinstance(x, torch.Tensor) else x for x in self],))


def test_may_contain_tensor():
    assert not _may_contain_tensor({'a': [1, 'x', None, (2.0, True)], 'b': {3: [4]}})
    assert _may_contain_tensor({'a': [{'b': torch.zeros(1)}]})
    assert _may_contain_tensor({torch.zeros(1): 'key'})            # dict key
    assert _may_contain_tensor(torch.Generator().manual_seed(1))   # pickled state
    assert _may_contain_tensor(lambda: 0)                          # unpicklable
    assert _may_contain_tensor(StateList([torch.ones(2)]))         # subclass hides its tensors
    assert _may_contain_tensor({'a': [OrderedDict(b=1)]})          # any container subclass


def test_split_walks_only_tensor_subtrees():
    shared = torch.arange(4.)
    loader = [{'i': i, 'l': [i, str(i)]} for i in range(1000)] + [{'t': shared}]
    data = {
        'model': {'w': shared, 'b': torch.ones(1), 'meta': 'x'},
        'loader': loader,
        'plain': [1, 2],
        'state_list': StateList([torch.ones(2)]),
    }
    walked = []
    skeleton = _split_tensor_subtrees(data, _BROADCAST_SPLIT_DEPTH, walked)
    # one tensor-bearing loader entry does not send the other 1000 through the walk
    expected = [shared, data['model']['b'], loader[-1], data['state_list']]
    assert len(walked) == len(expected) and all(a is b for a, b in zip(walked, expected))

    walked_skeleton, tensors = extract_tensors(walked)
    assert len(tensors) == 3  # `shared` is deduplicated across subtrees
    out = _fill_tensor_subtrees(skeleton, refill_tensors(walked_skeleton, tensors))
    old = refill_tensors(*extract_tensors(data))
    assert out.keys() == old.keys()
    assert out['model']['w'] is out['loader'][-1]['t'] is shared
    assert out['loader'][:-1] == loader[:-1] and out['plain'] == [1, 2]
    # same result as walking everything: the subclass is converted and keeps its tensor
    assert type(out['state_list']) is type(old['state_list']) is list
    assert torch.equal(out['state_list'][0], torch.ones(2))

    # data that is not an exact dict/list is walked whole, as before
    walked = []
    od = OrderedDict(a=1)
    _split_tensor_subtrees(od, _BROADCAST_SPLIT_DEPTH, walked)
    assert len(walked) == 1 and walked[0] is od


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
            # a per-rank list where only one entry holds a tensor
            'loader': big[:100] + [{'t': shared}],
            # distinct tensors with equal (data_ptr, shape, stride) as dict keys
            'slots': {key_a: 'a', key_b: 'b'},
            # objects that pickle their own tensors
            'rng': torch.Generator().manual_seed(123),
            'module': torch.nn.Linear(2, 1),
            # a list subclass whose pickled form hides its tensors
            'state_list': StateList([torch.ones(2)]),
        } if rank == 0 else None
        out = broadcast_mixed_data(data, src_rank=0, device=device)

        want = torch.device('cpu') if device == 'cpu' else torch.device('cuda', rank)
        t = out['tensors']
        assert t[0].device == want and torch.equal(t[0].cpu(), shared)
        assert t[0] is t[1]['same'] is out['loader'][-1]['t']
        assert t[1]['step'].item() == 7 and torch.equal(t[2].cpu(), shared[1])
        assert out['big'] == big and out['loader'][:-1] == big[:100]
        assert sorted(out['slots'].values()) == ['a', 'b']
        assert torch.equal(out['rng'].get_state(), torch.Generator().manual_seed(123).get_state())
        weight = out['module'].weight
        assert isinstance(weight, torch.nn.Parameter) and weight.requires_grad
        sl = out['state_list']
        assert type(sl) is list and sl[0].device == want and torch.equal(sl[0].cpu(), torch.ones(2))

    # dict subclasses keep the old path (and their type when nothing is extracted)
    out = broadcast_mixed_data(OrderedDict(a=[1]) if rank == 0 else None, src_rank=0)
    assert type(out) is OrderedDict and out == OrderedDict(a=[1])
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs 2 gpus')
def test_broadcast_mixed_data():
    launch_torchrun(2, _broadcast_worker)
