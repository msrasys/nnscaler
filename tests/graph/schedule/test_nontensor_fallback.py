#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import operator
from types import SimpleNamespace
import weakref

import pytest
import torch

from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.runtime.utils import MicroBatchDataLoader


def _fallback_code(lookups, *, root_input=False, root_produced=False, same_names=False):
    loader = IRObject('dataloader')
    sample = IRObject('data')
    nodes = [IRDataOperation(loader, (sample,))]
    value = sample
    for index, (operation, *args) in enumerate(lookups):
        output = IRObject('lookup' if same_names else f'lookup_{index}')
        signature = '_operator.getitem' if operation == 'getitem' else 'builtins.dict.get'
        node = IRPyFunc(signature, [value, *args], [output])
        nodes.append(node)
        value = node.output(0)
    graph = IRGraph(nodes, [loader], [value], 'metadata')
    originals = [sample, value] if root_input else [value]
    codegen = ScheduleCodeGen.__new__(ScheduleCodeGen)
    codegen.execplan = SimpleNamespace(graph=graph)
    stage = IRSegment([], originals, [])
    consumer = ExeReuseCell(stage, originals, [], micro_batch_id=1)
    sequence = [consumer]
    if root_produced:
        sequence.insert(0, ExeReuseCell(nodes[0], [loader], [sample], micro_batch_id=1))
    plan = codegen._plan_sample_reads(sequence)[consumer]
    return ('\n'.join(plan.codes), codegen.tensor_name(originals[-1]),
            codegen.tensor_name(originals[0]), codegen.tensor_name(loader))


@pytest.mark.parametrize(('lookups', 'sample', 'expected'), [
    ([('getitem', 'context'), ('get', 'scale', 7)], {'context': {}}, 7),
    ([('getitem', 'context'), ('get', 'scale')], {'context': {}}, None),
    ([('get', 'context', {'scale': 11}), ('getitem', 'scale')], {}, 11),
    ([('getitem', 'context'), ('get', 'scale', 7)], {'context': {'scale': 3}}, 3),
])
def test_fallback_preserves_dict_get_and_microbatch(lookups, sample, expected):
    code, result, _, loader_name = _fallback_code(lookups)
    first = {'context': {'scale': -1}}
    loader = MicroBatchDataLoader([first, sample])
    namespace = {'_operator': operator, loader_name: loader}

    exec(code, namespace)

    assert namespace[result] == expected
    assert next(loader) is first  # Random access must not advance the iterator.


def test_fallback_does_not_use_unproduced_sample_input():
    code, result, _, loader_name = _fallback_code(
        [('getitem', 'context'), ('getitem', 'scale')], root_input=True,
    )
    loader = MicroBatchDataLoader([
        {'context': {'scale': -1}}, {'context': {'scale': 5}},
    ])
    namespace = {'_operator': operator, loader_name: loader}

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_reuses_produced_sample_input():
    code, result, root, _ = _fallback_code(
        [('getitem', 'context'), ('getitem', 'scale')],
        root_input=True, root_produced=True,
    )
    # Deliberately omit the dataloader: the already-produced sample is enough.
    namespace = {'_operator': operator, root: {'context': {'scale': 5}}}

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_distinguishes_ir_objects_with_the_same_name():
    code, result, _, loader_name = _fallback_code(
        [('getitem', 'context'), ('get', 'scale', 7)], same_names=True,
    )
    namespace = {
        '_operator': operator,
        loader_name: MicroBatchDataLoader([{}, {'context': {'scale': 5}}]),
    }

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_preserves_getitem_key_error():
    code, _, _, loader_name = _fallback_code([('getitem', 'context'), ('getitem', 'scale')])
    namespace = {
        '_operator': operator,
        loader_name: MicroBatchDataLoader([{}, {'context': {}}]),
    }

    with pytest.raises(KeyError, match='scale'):
        exec(code, namespace)


def test_random_sample_plan_reuses_then_releases_the_same_owner():
    loader, sample = IRObject('dataloader'), IRObject('sample')
    reads = [IRDataOperation(loader, (sample,))]
    consumers = []
    for key in ('first', 'second'):
        lookup = IRPyFunc('_operator.getitem', [sample, key], [IRObject(key)])
        reads.append(lookup)
        stage = IRSegment([], [lookup.output(0)], [])
        consumers.append(ExeReuseCell(stage, stage.inputs(), [], micro_batch_id=1))
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(graph=IRGraph(reads, [loader], [], 'sample_replay'))
    plans = emitter._plan_sample_reads(consumers)
    first, second = [plans[node] for node in consumers]
    assert first.outputs[0] in second.inputs
    assert sum('get_micro_batch' in code for plan in plans.values() for code in plan.codes) == 1
    lifetime = LifeCycle(consumers, [], [], extra_inputs={node: plan.inputs for node, plan in plans.items()})

    class Loader:
        def get_micro_batch(self, mid):
            assert mid == 1
            tensor = torch.ones(8)
            self.reference = weakref.ref(tensor)
            return {'first': 3, 'second': 5, 'unused_tensor': tensor}

    runtime_loader = Loader()
    namespace = {'_operator': operator, emitter.tensor_name(loader): runtime_loader}
    for index, node in enumerate(consumers):
        exec('\n'.join(plans[node].codes), namespace)
        released = lifetime.release_tensors_after_node(node)
        exec(emitter.emit_release(released), namespace)
        assert (runtime_loader.reference() is not None) == (index == 0)


@pytest.mark.parametrize('scheduled', [False, True])
def test_replay_does_not_substitute_the_first_data_output(scheduled):
    loader = IRObject('dataloader')
    first, second = IRObject('first'), IRObject('second')
    read = IRDataOperation(loader, (first, second))
    lookup = IRPyFunc('_operator.getitem', [second, 'value'], [IRObject('value')])
    stage = IRSegment([], [lookup.output(0)], [])
    consumer = ExeReuseCell(stage, stage.inputs(), [], micro_batch_id=0)
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(graph=IRGraph([read, lookup], [loader], [], 'roots'))
    nodes = [consumer]
    if not scheduled:
        with pytest.raises(ValueError, match='dat[a-z]+ root is ambiguous'):
            emitter._plan_sample_reads(nodes)
        return
    nodes.insert(0, ExeReuseCell(read, [loader], [first, second], micro_batch_id=0))
    plan = emitter._plan_sample_reads(nodes)[consumer]
    namespace = {'_operator': operator, emitter.tensor_name(first): {'value': 3},
                 emitter.tensor_name(second): {'value': 7}}
    exec('\n'.join(plan.codes), namespace)
    assert namespace[emitter.tensor_name(lookup.output(0))] == 7


@pytest.mark.parametrize('other_delete', [False, True])
def test_dead_sample_delete_does_not_keep_the_dataloader_read(other_delete):
    code = '\n'.join([
        'def run(dataloader_1):',
        '    samples_5 = next(*(dataloader_1, ))',
        '    other = 4',
        '    del samples_5' + (', other' if other_delete else ''),
        '    return 7',
    ])
    code = ScheduleCodeGen._remove_dead_dataloader_reads(code)
    namespace = {}
    exec(code, namespace)
    source = iter([1, 2])
    assert namespace['run'](source) == 7
    assert next(source) == 1


def test_dead_segment_keeps_its_argument_without_reading_a_sample():
    code = '\n'.join([
        'def run(dataloader_1):',
        '    samples_5 = next(*(dataloader_1, ))',
        "    result = fexecute('segment2', samples_5)",
        '    del samples_5',
        '    return result',
    ])
    code = ScheduleCodeGen._remove_dead_dataloader_reads(code, {'segment2'})
    calls = []

    def execute(name, unused):
        calls.append((name, unused))
        return 7

    namespace = {'fexecute': execute}
    exec(code, namespace)
    assert namespace['run'](iter(())) == 7
    assert calls == [('segment2', None)]


def test_sample_with_a_real_consumer_is_not_pruned():
    code = '\n'.join([
        'def run(dataloader_1):',
        '    samples_5 = next(*(dataloader_1, ))',
        "    result = samples_5['value']",
        '    del samples_5',
        '    return result',
    ])
    namespace = {}
    exec(ScheduleCodeGen._remove_dead_dataloader_reads(code), namespace)
    source = iter([{'value': 7}, {'value': 9}])
    assert namespace['run'](source) == 7
    assert next(source)['value'] == 9


@pytest.mark.parametrize('use', ['return', 'keyword'])
def test_returned_or_keyword_sample_is_not_a_dead_segment_input(use):
    sample = IRObject('samples')
    if use == 'return':
        segment = IRSegment([], [sample], [sample])
    else:
        operation = IRFwOperation('consume', 'consume', [], 0, metadata={'sample': sample})
        segment = IRSegment([operation], [sample], [])
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    assert not emitter._find_dead_samples_segments([segment])
