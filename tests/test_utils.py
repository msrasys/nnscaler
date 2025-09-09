#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import dataclass
import pytest
import torch

from nnscaler.utils import select_many, classproperty, fields, set_member_by_name, unchecked_fields


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
