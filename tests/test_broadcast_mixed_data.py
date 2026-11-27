#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pickle
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from nnscaler.utils import (
    _BROADCAST_SPLIT_DEPTH, _WalkedSubtree, _extract_tensor_subtrees, _may_contain_tensor,
    _refill_tensor_subtrees, _tensor_indexer, broadcast_mixed_data, extract_tensors,
)
from tests.launch_torchrun import launch_torchrun


class StateList(list):
    """A list subclass whose pickled form does not hold the tensors it contains."""
    def __reduce__(self):
        return (StateList, ([x.tolist() if isinstance(x, torch.Tensor) else x for x in self],))


class Cursor:
    """Pickling consumes the iterator, so `__reduce__` must run only once."""
    reduce_calls = 0

    def __init__(self, pending):
        self.pending = iter(pending)

    def __reduce__(self):
        Cursor.reduce_calls += 1
        return Cursor, (list(self.pending),)


@dataclass
class Point:
    x: int


def _split(data):
    """The source side of `broadcast_mixed_data`: `(changed, skeleton, tensors)`."""
    transform_fn, tensors = _tensor_indexer()
    changed, skeleton = _extract_tensor_subtrees(data, _BROADCAST_SPLIT_DEPTH, transform_fn)
    return changed, skeleton, tensors


def test_may_contain_tensor():
    assert not _may_contain_tensor({'a': [1, 'x', None, (2.0, True)], 'b': {3: [4]}})
    assert not _may_contain_tensor([np.arange(3), np.int64(1), torch.float32, {1, 2}, b'x'])
    assert _may_contain_tensor({'a': [{'b': torch.zeros(1)}]})
    assert _may_contain_tensor({torch.zeros(1): 'key'})            # dict key
    assert _may_contain_tensor(torch.Generator().manual_seed(1))   # pickled state
    assert _may_contain_tensor(lambda: 0)                          # unpicklable
    assert _may_contain_tensor(StateList([torch.ones(2)]))         # subclass hides its tensors
    assert _may_contain_tensor({'a': [OrderedDict(b=1)]})          # any container subclass
    assert _may_contain_tensor([Point(1)])                         # any other object
    # the probe stops before running an object's own pickling code
    Cursor.reduce_calls = 0
    assert _may_contain_tensor({'c': Cursor([1, 2])})
    assert Cursor.reduce_calls == 0


def test_extract_tensor_subtrees_matches_extract_tensors():
    shared = torch.arange(4.)
    loader = [{'i': i, 'l': [i, str(i)]} for i in range(1000)] + [{'t': shared}]
    data = {
        'model': {'w': shared, 'b': torch.ones(1), 'meta': 'x'},
        'loader': loader,
        'plain': [1, 2],
        'state_list': StateList([torch.ones(2)]),
        'point': Point(3),
    }
    changed, skeleton, tensors = _split(data)
    old_skeleton, old_tensors = extract_tensors(data)
    assert changed
    # same tensors, deduplicated across subtrees, in the same order
    assert len(tensors) == len(old_tensors) == 3
    assert all(a is b for a, b in zip(tensors, old_tensors))
    # one tensor-bearing loader entry does not send the other 1000 through the walk
    sk_loader = skeleton.value['loader'].value
    assert isinstance(sk_loader[-1], _WalkedSubtree)
    assert all(a is b for a, b in zip(sk_loader[:-1], loader[:-1]))
    # subtrees without extracted tensors are the original objects, not copies
    assert skeleton.value['plain'] is data['plain'] and skeleton.value['point'] is data['point']

    out = _refill_tensor_subtrees(pickle.loads(pickle.dumps(skeleton)), tensors)
    assert out['model']['w'] is out['loader'][-1]['t'] is shared
    assert out['loader'][:-1] == loader[:-1] and out['plain'] == [1, 2] and out['point'] == Point(3)
    # same result as walking everything: the subclass is converted and keeps its tensor
    assert type(out['state_list']) is list and torch.equal(out['state_list'][0], torch.ones(2))

    # nothing extracted: the data itself is the skeleton
    tensor_free = {'a': [OrderedDict(b=1)], 'p': Point(1), 'l': loader[:10]}
    assert _split(tensor_free)[:2] == (False, tensor_free)
    assert _split(tensor_free)[1] is tensor_free
    # data that is not an exact dict/list is walked whole, as before
    od = OrderedDict(a=1)
    assert _split(od)[1] is od


@pytest.mark.parametrize('receiver', [False, True])
def test_extract_tensor_subtrees_keeps_shared_objects(receiver):
    # the OrderedDict makes the probe walk `shared` (from both keys), but no
    # tensor is replaced in it, so it must not be copied
    shared = [OrderedDict()]
    _, skeleton, tensors = _split({'a': shared, 'b': shared, 't': torch.ones(1)})
    if receiver:
        skeleton = pickle.loads(pickle.dumps(skeleton))
    out = _refill_tensor_subtrees(skeleton, tensors)
    assert out['a'] is out['b']
    out['a'].append('new')
    assert out['b'] == [OrderedDict(), 'new']


def test_extract_tensor_subtrees_runs_reduce_once():
    Cursor.reduce_calls = 0
    data = {'cursor': Cursor([1, 2]), 'nested': [Cursor([3])], 't': torch.ones(1)}
    _, skeleton, tensors = _split(data)
    out = _refill_tensor_subtrees(pickle.loads(pickle.dumps(skeleton)), tensors)
    assert Cursor.reduce_calls == 2  # once per cursor, in the real pickling
    assert list(out['cursor'].pending) == [1, 2] and list(out['nested'][0].pending) == [3]


def _broadcast_worker():
    torch.distributed.init_process_group('nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    shared = torch.arange(6.).view(2, 3)
    key_a, key_b = torch.empty(0), torch.empty(0)
    big = [{'i': i, 'l': [i, str(i), True]} for i in range(10000)]
    for device in ('cpu', None):
        shared_list = [OrderedDict()]
        Cursor.reduce_calls = 0
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
            # a `__reduce__` with a side effect
            'cursor': Cursor([1, 2]),
            # one list referenced from two keys
            'shared_a': shared_list,
            'shared_b': shared_list,
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
        if rank == 0:
            assert Cursor.reduce_calls == 1  # only by the broadcast's own pickling
        else:
            assert list(out['cursor'].pending) == [1, 2]
        assert out['shared_a'] is out['shared_b']
        out['shared_a'].append('new')
        assert out['shared_b'] == [OrderedDict(), 'new']

    # dict subclasses keep the old path (and their type when nothing is extracted)
    out = broadcast_mixed_data(OrderedDict(a=[1]) if rank == 0 else None, src_rank=0)
    assert type(out) is OrderedDict and out == OrderedDict(a=[1])
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs 2 gpus')
def test_broadcast_mixed_data():
    launch_torchrun(2, _broadcast_worker)
