# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import weakref

import torch

from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor


def op(name, inputs, output, **kwargs):
    node = IRFwOperation(name, name, inputs, 1, **kwargs)
    node.set_output(0, output)
    return node


def tensor(name):
    return IRFullTensor((1,), name=name, dtype=torch.float32).tosub()


def test_last_use_releases_container_owned_tensor_before_later_work():
    metadata = IRObject('metadata')
    value = tensor('value')
    result = tensor('result')
    make = op('make', [], metadata)
    consume = op('consume', [metadata], value)
    later = op('later', [value], result)
    lifetime = LifeCycle([make, consume, later], [], [result])
    assert metadata in lifetime.release_tensors_after_node(consume)
    activation = torch.randn(4)
    ref = weakref.ref(activation)
    namespace = {FuncEmission().tensor_name(metadata): {'hidden': activation}}
    del activation
    assert ref() is not None
    exec(FuncEmission().emit_release(lifetime.release_tensors_after_node(consume)), namespace)
    assert ref() is None


def test_nested_keyword_consumer_extends_container_and_tensor_lifetime():
    metadata = IRObject('metadata')
    value = tensor('tensor')
    middle = tensor('middle')
    result = tensor('result')
    first = op('first', [metadata, value], middle)
    last = op('last', [middle], result, options={'metadata': [metadata], 'tensor': value})
    lifetime = LifeCycle([first, last], [metadata, value], [result])
    assert metadata not in lifetime.release_tensors_after_node(first)
    assert value not in lifetime.release_tensors_after_node(first)
    assert metadata in lifetime.release_tensors_after_node(last)
    assert value in lifetime.release_tensors_after_node(last)


def test_returned_container_and_model_attributes_are_preserved():
    attribute = IRObject('attribute').as_attr()
    metadata = IRObject('metadata')
    node = op('make', [attribute], metadata)
    node.input(0).as_attr()
    lifetime = LifeCycle([node], [], [{'nested': metadata}])
    assert attribute not in lifetime.lifetime
    assert metadata not in lifetime.release_tensors_after_node(node)
    assert not lifetime.releasable_after_node(attribute, node)
