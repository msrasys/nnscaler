import pytest

from torch.utils._pytree import tree_flatten

from nnscaler.graph.parser.fx.concrete_trace_utils.concrete_proxy import (
    ConcreteProxy,
    Node,
)
from nnscaler.graph.parser.fx.concrete_trace_utils.concrete_tracer import (
    update_tree_proxy_value
)
from nnscaler.graph.parser.fx.concrete_trace_utils.utils import (
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


def test_update_tree_proxy_value():
    class DummyNode:
        def __init__(self, name):
            self.name = name
            self.graph = None

    pytree_1 = ConcreteProxy(node=DummyNode('test_node'), value={'a': {'b': [1, 2]}}, tracer=None)
    pytree_2 = {'a': {'b': [1, 3]}}
    new_pytree = update_tree_proxy_value(pytree_1, pytree_2)
    assert str(new_pytree) == "ConcreteProxy(test_node, {'a': {'b': [1, 3]}})"

    pytree_1 = {'a': ConcreteProxy(node=DummyNode('test_node'), value={'b': [1, 2]}, tracer=None)}
    pytree_2 = {'a': {'b': [1, 3]}}
    new_pytree = update_tree_proxy_value(pytree_1, pytree_2)
    assert str(new_pytree) == "{'a': ConcreteProxy(test_node, {'b': [1, 3]})}"

    pytree_1 = ConcreteProxy(
        node=DummyNode('t1'), 
        value={'a': ConcreteProxy(
            node=DummyNode('t2'),
            value={'b': ConcreteProxy(
                node=DummyNode('t3'),
                value=[1, ConcreteProxy(
                    node=DummyNode('t4'),
                    value=2,
                    tracer=None
                )],
                tracer=None)
            },
            tracer=None)
        }, 
        tracer=None
    )
    pytree_2 = {'a': {'b': [1, 3]}}
    new_pytree = update_tree_proxy_value(pytree_1, pytree_2)
    assert str(new_pytree) == "ConcreteProxy(t1, {'a': ConcreteProxy(t2, {'b': ConcreteProxy(t3, [1, ConcreteProxy(t4, 3)])})})"

    pytree_1 = {'a': ConcreteProxy(node=DummyNode('test_node'), value={'b': [1, 2]}, tracer=None)}
    pytree_2 = {'b': {'a': [1, 3]}}
    new_pytree = update_tree_proxy_value(pytree_1, pytree_2)
    # because the spec of pytree_1 - {'a': {'b': *}} - and pytree_2 - {'b': {'a': *}} - is completely differet,
    # the result is directly pytree_2
    assert str(new_pytree) == "{'b': {'a': [1, 3]}}"
