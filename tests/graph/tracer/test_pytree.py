import pytest

from torch.utils._pytree import tree_flatten

from cube.graph.parser.fx.concrete_trace_utils.utils import (
    flatten_tree_with_spec,
    flatten_trees_with_func,
    flatten_trees_with_func_and_spec,
    map_trees_with_func
)


def test_flatten_tree_with_spec():
    pytree_1 = [1, (2, (3, 4))]
    pytree_2 = [1, (2, (3, (4, 5)))]
    pytree_3 = [1, (2, (3,))]
    _, spec = tree_flatten(pytree_1)

    assert flatten_tree_with_spec(pytree_2, spec) == [1, 2, 3, (4, 5)]

    # pytree_3 can not flatten by pytree_1 spec, so it should raise error
    with pytest.raises(RuntimeError):
        flatten_tree_with_spec(pytree_3, spec)


def test_flatten_trees_with_func():
    pytree_1 = [1, (2, {3: 4})]
    pytree_2 = [5, (6, {3: 5})]
    flat_args, spec = flatten_trees_with_func(lambda a, b: a + b, [pytree_1, pytree_2])
    assert flat_args == [6, 8, 9]
    assert spec == tree_flatten(pytree_1)[1]

    pytree_3 = [1, (2, (3,))]
    # pytree_3 has different spec with pytree_1 and pytree_2, so it should raise error
    with pytest.raises(RuntimeError):
        flatten_trees_with_func(lambda a, b, c: a + b + c, [pytree_1, pytree_2, pytree_3])


def test_flatten_trees_with_func_and_spec():
    pytree_0 = [1, (2, 3)]
    _, spec = tree_flatten(pytree_0)

    def merge(a, b):
        if isinstance(a, dict):
            assert isinstance(b, dict)
            return {**a, **b}
        else:
            return a + b

    pytree_1 = [1, (2, {3: 4})]
    pytree_2 = [5, (6, {4: 5})]
    assert flatten_trees_with_func_and_spec(merge, [pytree_1, pytree_2], spec) == [6, 8, {3: 4, 4: 5}]


def test_map_trees_with_func():
    pytree_1 = [1, (2, {3: 4})]
    pytree_2 = [5, (6, {3: 5})]

    assert map_trees_with_func(lambda a, b: a + b, [pytree_1, pytree_2]) == [6, (8, {3: 9})]
