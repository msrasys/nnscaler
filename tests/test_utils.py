#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import OrderedDict
from dataclasses import dataclass
import pytest
import torch

from nnscaler.utils import (
    select_many, classproperty, fields, set_member_by_name, unchecked_fields,
    transform_recursively,
)


def test_select_many():
    assert list(select_many([1, 2], lambda k: [])) == []
    assert list(select_many([1, [2, 3]], lambda k: k if isinstance(k, list) else [k])) == [1, 2, 3]
    with pytest.raises(TypeError):
        list(select_many([1, [2, 3]], lambda k: k))


def test_classproperty_int():
    class A:
        _x = 1234567
        @classproperty
        def value(cls):
            return cls._x

    assert A.value == 1234567
    assert id(A().value) == id(A.value)

    with pytest.raises(AttributeError):
        A().value = 43

    assert A.value == 1234567


def test_classproperty_dict():
    class A:
        _x = {}
        @classproperty
        def cfg(cls):
            return cls._x.setdefault('a', {})

    x = A.cfg
    x[1] = 2
    assert A.cfg == {1: 2}
    assert id(A().cfg) == id(x)


def test_fields():
    @dataclass
    class A:
        x: int
        y: int

    assert fields(A).x == 'x'
    assert fields(A).y == 'y'
    with pytest.raises(AttributeError):
        fields(A).z

    assert unchecked_fields(A).x == 'x'
    assert unchecked_fields(A).y == 'y'
    assert unchecked_fields(A).z == 'z'

    a = A(x=0, y=0)
    assert unchecked_fields(a).x == 'x'
    assert unchecked_fields(a).y == 'y'
    assert unchecked_fields(a).z == 'z'

    class B:
        def __init__(self):
            self.a = A(x=1, y=2)

    assert unchecked_fields(B).x == 'x'
    b = B()
    assert unchecked_fields(b).x == 'x'
    assert unchecked_fields(b.a).x == 'x'


def test_set_member_by_name():
    model = torch.nn.Module()
    set_member_by_name(model, "x", 42)
    assert model.x == 42
    with pytest.raises(AttributeError):
        set_member_by_name(model, 'x.y.z', 43)

    set_member_by_name(model, 'a.b.c', 44)
    assert model.a.b.c == 44

    model = torch.nn.Module()
    child_module = torch.nn.Module()
    set_member_by_name(model, "x.y", child_module)
    assert model.x.y == child_module

    set_member_by_name(model, 'x.y.z', 45)
    assert model.x.y == child_module
    assert model.x.y.z == 45


def test_transform_recursively():
    data = {
        'a': torch.tensor([1]),
        'b': [torch.tensor(4), {'c': torch.tensor([5])}],
        'd': (7, torch.tensor(8)),
        'e': {1: 9, 2: torch.tensor(10)}.keys(),
        'f': {1: 9, 2: torch.tensor(11)}.items(),
        'g': {1: 9, 2: torch.tensor(12)}.values(),
        'h': {1: 9, 2: torch.tensor(13)},
        'i': slice(0, 10, None),
        'j': torch.Size([11, 12]),
        'k': OrderedDict({1: 9, 2: 10}),
        'l': {1: 9, 2: 10}.values(),
        'm': [1, 2, 3],
        'n': slice(0, 10, torch.tensor(2)),
        'o': {torch.tensor(1): 9, torch.tensor(2): 10},
        'p': {torch.tensor(1): 9, torch.tensor(2): 10}.items(),
        'q': {torch.tensor(1): 9, torch.tensor(2): 10}.keys()
    }

    def fn(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return x

    result1 = transform_recursively(
        data, fn,
        target_types=torch.Tensor,
        collection_types=None,
        skip_dict_keys=True,
    )

    result2 = transform_recursively(
        data, fn,
        target_types=torch.Tensor,
        collection_types=None,
        skip_dict_keys=False,
    )
    target = {
        'a': 1,
        'b': [4, {'c': 5}],
        'd': (7, 8),
        'e': {1: 1, 2: 2}.keys(),
        'f': dict([(1, 9), (2, 11)]).items(),
        'g': {1: 9, 2: 12}.values(),
        'h': {1: 9, 2: 13},
        'i': slice(0, 10, None),
        'j': torch.Size([11, 12]),
        'k': OrderedDict({1: 9, 2: 10}),
        'l': data['l'],
        'm': [1, 2, 3],
        'n': slice(0, 10, 2),
    }
    # dict values are not comparable.
    assert list(target['g']) == list(result1.pop('g'))
    assert list(target['g']) == list(result2.pop('g'))
    target.pop('g')


    skip_key_target = {
        **target,
        'o': {torch.tensor(1): 9, torch.tensor(2): 10},
        'p': {torch.tensor(1): 9, torch.tensor(2): 10}.items(),
        'q': {1: 9, 2: 10}.keys()
    }
    noskip_key_target = {
        **target,
        'o': {1: 9, 2: 10},
        'p': dict([(1, 9), (2, 10)]).items(),
        'q': {1: 9, 2: 10}.keys()
    }

    from tests.parallel_module.common import assert_equal

    assert_equal(list(skip_key_target.pop('o')), list(result1.pop('o')))
    assert_equal(list(skip_key_target.pop('p')), list(result1.pop('p')))
    assert_equal(list(skip_key_target.pop('q')), list(result1.pop('q')))

    assert_equal(result1, skip_key_target)
    assert_equal(result2, noskip_key_target)
