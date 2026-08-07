#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import FrozenInstanceError

import pytest

from nnscaler.graph import IRGraph
from nnscaler.graph.function import Linear
from nnscaler.ir import ModelSpec
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.tensor import IRFullTensor


def _tensor(shape=(8, 8), *, requires_grad=True):
    return IRFullTensor(shape, requires_grad=requires_grad).tosub()


def test_model_spec_is_immutable_and_slotted():
    spec = ModelSpec('attention', 'model.layers.0.self_attn', 'model.py:10-20')

    assert spec.component == 'attention'
    assert not hasattr(spec, '__dict__')
    with pytest.raises(FrozenInstanceError):
        spec.component = 'mlp'


@pytest.mark.parametrize('component', [
    'embedding',
    'attention',
    'moe',
    'mlp',
    'normalization',
    'output',
    'other',
])
def test_model_spec_accepts_supported_components(component):
    assert ModelSpec(component, 'model.block', 'model.py:1').component == component


@pytest.mark.parametrize(
    ('args', 'error'),
    [
        (('', 'model.block', 'model.py:1'), ValueError),
        (('invalid', 'model.block', 'model.py:1'), ValueError),
        (('mlp', '', 'model.py:1'), ValueError),
        (('mlp', ' model.block', 'model.py:1'), ValueError),
        (('mlp', 'model block', 'model.py:1'), ValueError),
        (('mlp', 'model=block', 'model.py:1'), ValueError),
        (('mlp', 'model.block', 'model.py:\n1'), ValueError),
        (('mlp', 'model.block', '/repo/model.py:1'), ValueError),
        (('mlp', 'model.block', r'C:\repo\model.py:1'), ValueError),
        (('mlp', 'model.block', '../model.py:1'), ValueError),
        (('mlp', 'model.block', 'llm/../model.py:1'), ValueError),
        (('mlp', 'model.block', r'llm\..\model.py:1'), ValueError),
        (('mlp', 1, 'model.py:1'), TypeError),
    ],
)
def test_model_spec_rejects_unstable_values(args, error):
    with pytest.raises(error):
        ModelSpec(*args)


def test_copy_node_meta_info_preserves_model_spec():
    source = Linear(_tensor(requires_grad=False), _tensor())
    destination = source.replicate()
    spec = ModelSpec('mlp', 'model.layers.0.mlp', 'model.py:30-40')
    source.model_spec = spec

    IRGraph.copy_node_meta_info(source, destination)

    assert destination.model_spec is spec


def test_graph_replicate_preserves_model_spec():
    data = _tensor(requires_grad=False)
    node = Linear(data, _tensor())
    node.set_output(0, _tensor())
    spec = ModelSpec('mlp', 'model.layers.0.mlp', 'model.py:30-40')
    node.model_spec = spec
    graph = IRGraph([node], [data], [node.output(0)], 'model')

    replicas = graph.replicate(node, 2)

    assert [replica.model_spec for replica in replicas] == [spec, spec]


def test_grouping_and_fusion_model_spec_consensus():
    first = Linear(_tensor(requires_grad=False), _tensor())
    first.set_output(0, _tensor())
    second = Linear(first.output(0), _tensor())
    second.set_output(0, _tensor())
    first_spec = ModelSpec('mlp', 'model.layers.0.mlp', 'model.py:30-40')
    second_spec = ModelSpec('attention', 'model.layers.0.attn', 'model.py:10-20')
    first.model_spec = first_spec
    second.model_spec = first_spec
    graph = IRGraph([first, second], [first.input(0)], [second.output(0)], 'model')

    assert graph.create_segment([first, second]).model_spec == first_spec

    second.model_spec = second_spec
    assert graph.create_segment([first, second]).model_spec is None

    first_adapter = IRAdapter([], [])
    second_adapter = IRAdapter([], [])
    first_adapter.model_spec = first_spec
    second_adapter.model_spec = first_spec
    assert IRAdapter.merge([first_adapter, second_adapter]).model_spec == first_spec

    second_adapter.model_spec = second_spec
    assert IRAdapter.merge([first_adapter, second_adapter]).model_spec is None
