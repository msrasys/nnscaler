#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

from nnscaler.utils import select_many, classproperty


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
