#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import operator
from types import SimpleNamespace

import pytest

from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.graph import IRGraph
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.operator import IRDataOperation
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
    actuals = [IRObject(obj.name) for obj in originals]
    cell = IRCell('segment', 'segment', len(originals), 0)
    for index, obj in enumerate(originals):
        cell.set_input(index, obj)
    reuse = ExeReuseCell(cell, actuals, [], micro_batch_id=1)
    codegen = ScheduleCodeGen.__new__(ScheduleCodeGen)
    produced = {actuals[0].tid} if root_produced else set()
    codes = codegen._emit_missing_nontensor_inputs(
        reuse, produced, codegen._collect_getitem_info(SimpleNamespace(graph=graph)),
        'dataloader', {'sample_reads': {}, 'fallback_vars': {}},
    )
    return '\n'.join(codes), codegen.tensor_name(actuals[-1]), codegen.tensor_name(actuals[0])


@pytest.mark.parametrize(('lookups', 'sample', 'expected'), [
    ([('getitem', 'context'), ('get', 'scale', 7)], {'context': {}}, 7),
    ([('getitem', 'context'), ('get', 'scale')], {'context': {}}, None),
    ([('get', 'context', {'scale': 11}), ('getitem', 'scale')], {}, 11),
    ([('getitem', 'context'), ('get', 'scale', 7)], {'context': {'scale': 3}}, 3),
])
def test_fallback_preserves_dict_get_and_microbatch(lookups, sample, expected):
    code, result, _ = _fallback_code(lookups)
    first = {'context': {'scale': -1}}
    loader = MicroBatchDataLoader([first, sample])
    namespace = {'_operator': operator, 'dataloader': loader}

    exec(code, namespace)

    assert namespace[result] == expected
    assert next(loader) is first  # Random access must not advance the iterator.


def test_fallback_does_not_use_unproduced_sample_input():
    code, result, _ = _fallback_code(
        [('getitem', 'context'), ('getitem', 'scale')], root_input=True,
    )
    loader = MicroBatchDataLoader([
        {'context': {'scale': -1}}, {'context': {'scale': 5}},
    ])
    namespace = {'_operator': operator, 'dataloader': loader}

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_reuses_produced_sample_input():
    code, result, root = _fallback_code(
        [('getitem', 'context'), ('getitem', 'scale')],
        root_input=True, root_produced=True,
    )
    # Deliberately omit the dataloader: the already-produced sample is enough.
    namespace = {'_operator': operator, root: {'context': {'scale': 5}}}

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_distinguishes_ir_objects_with_the_same_name():
    code, result, _ = _fallback_code(
        [('getitem', 'context'), ('get', 'scale', 7)], same_names=True,
    )
    namespace = {
        '_operator': operator,
        'dataloader': MicroBatchDataLoader([{}, {'context': {'scale': 5}}]),
    }

    exec(code, namespace)

    assert namespace[result] == 5


def test_fallback_preserves_getitem_key_error():
    code, _, _ = _fallback_code([('getitem', 'context'), ('getitem', 'scale')])
    namespace = {
        '_operator': operator,
        'dataloader': MicroBatchDataLoader([{}, {'context': {}}]),
    }

    with pytest.raises(KeyError, match='scale'):
        exec(code, namespace)
