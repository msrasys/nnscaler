# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""CPU coverage for effect facts crossing tracing and graph transformations."""

import copy

import pytest
import torch
from torch.fx import Graph

from nnscaler import register_op
import nnscaler.graph.function as F
from nnscaler.graph import IRGraph
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.graph.parser import effects as parser_effects
from nnscaler.graph.parser.parser import FxModuleParser
from nnscaler.graph.tracer.torch_fx_patcher import ExtraSEFPatcher
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from tests.utils import replace_all_device_with


@register_op('m n -> m n')
def opaque_effect_test_op(x):
    return x.sin()


def _fx_call(target=opaque_effect_test_op, **kwargs):
    graph = Graph()
    value = graph.placeholder('x')
    return graph.call_function(target, (value,), kwargs)


def _ir_op():
    return IRFwOperation('opaque', 'custom.opaque', [], 0)


def _parse_meta(fx_node, ir_node=None):
    ir_node = _ir_op() if ir_node is None else ir_node
    FxModuleParser._set_node_meta(fx_node, ir_node)
    return ir_node


@pytest.mark.parametrize('target, expected', [
    (torch.ops.aten.add.out, ('mutation',)),
    (torch.ops.aten.copy_.default, ('mutation',)),
    (torch.ops.aten.view.default, ()),
    (torch.ops.aten.rand.default, ()),
    (torch.rand, ()),
])
def test_schema_writes_are_barriers_but_aliases_and_managed_rng_are_not(target, expected):
    assert _parse_meta(_fx_call(target)).recompute_barriers == expected


def test_opaque_leaf_dce_retention_is_not_a_recompute_effect():
    fx_node = _fx_call()
    with ExtraSEFPatcher({opaque_effect_test_op}):
        assert fx_node.is_impure()
        assert _parse_meta(fx_node).recompute_barriers == ()


@pytest.mark.parametrize('kind', ['operator', 'higher_order'])
def test_registered_ordered_effect_is_a_barrier_without_schema_mutation(kind):
    effects = pytest.importorskip('torch._higher_order_ops.effects')
    if not hasattr(effects, '_get_effect') or not hasattr(effects, '_register_effectful_op'):
        pytest.skip('PyTorch does not expose effect registration')
    library, handle = None, None
    try:
        if kind == 'operator':
            library = torch.library.Library('nnscaler_effect_facts_test', 'FRAGMENT')
            library.define('record(Tensor x) -> Tensor')
            target = torch.ops.nnscaler_effect_facts_test.record.default
            assert not target._schema.is_mutable
            # No implementation is installed: collecting facts must not call it.
            handle = effects._register_effectful_op(target, effects._EffectType.ORDERED)
        else:
            target = effects.call_torchbind
        assert effects._get_effect(target) is not None
        assert _parse_meta(_fx_call(target)).recompute_barriers == ('side_effect',)
    finally:
        if handle is not None:
            handle.destroy()
        if library is not None:
            library._destroy()


def test_missing_effect_registry_keeps_other_known_facts(monkeypatch):
    monkeypatch.setattr(parser_effects, '_get_registered_effect', None)
    fx_node = _fx_call(torch.ops.aten.add.out, generator=torch.Generator(device='cpu'))
    fx_node.meta['nnscaler_recompute_barriers'] = ('custom_state',)
    assert _parse_meta(fx_node).recompute_barriers == ('custom_state', 'mutation', 'external_rng')


def test_explicit_opaque_barriers_combine_with_schema_and_existing_ir_facts():
    fx_node = _fx_call(torch.ops.aten.add.out)
    fx_node.meta['nnscaler_recompute_barriers'] = ['communication', 'mutation']
    ir_node = _ir_op()
    ir_node.recompute_barriers = ('custom_state', 'communication')
    assert _parse_meta(fx_node, ir_node).recompute_barriers == (
        'custom_state', 'communication', 'mutation')


@pytest.mark.parametrize('location', ['literal', 'fx_meta', 'ir_value', 'nested'])
def test_visible_explicit_generator_is_a_barrier(location):
    generator = torch.Generator(device='cpu')
    fx_node, ir_node = _fx_call(), _ir_op()
    if location == 'literal':
        fx_node.kwargs = {'generator': generator}
    elif location == 'fx_meta':
        fx_node.args[0].meta['type'] = torch.Generator
    elif location == 'ir_value':
        ir_node.set_kwarg('generator', IRObject('generator', value=generator))
    else:
        nested = {'settings': [None, {'generator': generator}]}
        nested['cycle'] = nested
        ir_node.set_kwarg('options', IRObject('options', value=nested))
    assert _parse_meta(fx_node, ir_node).recompute_barriers == ('external_rng',)


def test_generator_scan_does_not_evaluate_opaque_python_objects():
    class Opaque:
        @property
        def generator(self):
            raise AssertionError('must not inspect arbitrary properties')

        def __iter__(self):
            raise AssertionError('must not execute an arbitrary iterator')

    ir_node = _ir_op()
    ir_node.set_kwarg('options', IRObject('options', value=Opaque()))
    assert _parse_meta(_fx_call(), ir_node).recompute_barriers == ()


@pytest.mark.parametrize('declared', ['mutation', ('',)])
def test_malformed_explicit_facts_are_rejected(declared):
    fx_node = _fx_call()
    fx_node.meta['nnscaler_recompute_barriers'] = declared
    with pytest.raises(TypeError, match='nnscaler_recompute_barriers'):
        _parse_meta(fx_node)


def test_legacy_ir_without_effect_metadata_is_readable():
    ir_node = _ir_op()
    del ir_node._recompute_barriers
    assert ir_node.recompute_barriers == ()
    assert ir_node.replicate().recompute_barriers == ()


def _dim_op():
    value = IRFullTensor((8, 8), requires_grad=True).tosub()
    op = F.Tanh(value, signature='torch.tanh')
    op.set_output(0, IRFullTensor((8, 8), requires_grad=True).tosub())
    op.recompute_barriers = ('custom_state',)
    return op


def test_graph_transformations_preserve_effects_without_sharing_mutable_state():
    op = _dim_op()
    for replica in (copy.copy(op), copy.deepcopy(op), op.replicate()):
        assert replica.recompute_barriers == ('custom_state',)
        replica.recompute_barriers += ('communication',)
        assert op.recompute_barriers == ('custom_state',)
    op.device = 0
    assert op.dispatch(0).recompute_barriers == ('custom_state',)

    source, destination = _dim_op(), _dim_op()
    destination.recompute_barriers = ('communication',)
    IRGraph.copy_node_meta_info(source, destination)
    assert set(destination.recompute_barriers) == {'custom_state', 'communication'}

    op = _dim_op()
    original_factory = op._create_fn[0]

    def factory(*args, **kwargs):
        result = original_factory(*args, **kwargs)
        result.recompute_barriers = ('communication',)
        return result

    op._create_fn = (factory,)
    rebuilt = op.new(list(op.inputs()), list(op.outputs()))
    assert rebuilt.recompute_barriers == ('custom_state', 'communication')

    op = _dim_op()
    graph = IRGraph([op], [op.input(0)], [op.output(0)], 'effects')
    for partitions in (op.algorithm('dim').instantiate(idx=0, dim=0, num=2),
                       graph.partition(op, op.algorithm('dim'), idx=0, dim=0, num=2)):
        assert len(partitions) == 2
        assert all(part.recompute_barriers == ('custom_state',) for part in partitions)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('declared', [(), ('opaque_state',)])
def test_real_registered_leaf_fx_to_ir_preserves_only_declared_effects(tmp_path, declared):
    class Model(torch.nn.Module):
        def forward(self, x):
            return opaque_effect_test_op(x)

    inputs = {'x': torch.randn(4, 4, requires_grad=True)}
    traced = to_fx_graph(Model(), inputs)
    fx_op, = [node for node in traced.graph.nodes if node.op == 'call_function']
    if declared:
        fx_op.meta['nnscaler_recompute_barriers'] = declared
    graph = to_ir_graph(traced, inputs, tmp_path, constant_folding=False)
    ir_op, = graph.nodes()
    assert ir_op.recompute_barriers == declared
    assert ir_op.replicate().recompute_barriers == declared
