#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch
from nnscaler.codegen.emit import CodeEmission, IRValue
from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.execplan import ExecutionPlan
from nnscaler.graph.function import Dropout
from nnscaler.graph.function.function import MultiRef
from nnscaler.graph.gener.utils import DummyInputOuput
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.schedule.schedplan import SchedulePlan
from nnscaler.graph.segment import IRSegment
from nnscaler.flags import CompileFlag
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import ChunkPrim, MovePrim
from nnscaler.ir.tensor import IRFullTensor


def _make_send_adapter(src=None):
    src = src or IRFullTensor(
        (8, 4), name='x', requires_grad=True, dtype=torch.float32
    ).tosub()
    chunk = IRFullTensor((4, 4), name='x_chunk', dtype=torch.float32).tosub()
    adapter = IRAdapter([src], [])
    adapter.device = [0]
    chunk.cell = adapter
    adapter.prims = [
        ChunkPrim([src], [chunk], dim=0, ranks=[0, 1]),
        MovePrim([chunk], [], shape=(4, 4), dtype='torch.float32', src=0, dst=1),
    ]
    return adapter


def _make_pipeline_boundary(*, alias_output=False, dummy_boundary=False):
    inp = IRFullTensor(
        (8, 4), name='input', requires_grad=True, dtype=torch.float32
    ).tosub()
    producer = Dropout(
        inp,
        0.0,
        True,
        False,
        signature='torch.nn.functional.dropout',
    )
    producer.set_output(0, IRFullTensor(
        (8, 4), name='produced', requires_grad=True, dtype=torch.float32
    ).tosub())
    producer.device = [0]
    nodes = [producer]
    output = producer.output(0)
    if alias_output:
        alias = MultiRef(output, 2)
        for idx in range(2):
            alias.set_output(idx, IRFullTensor(
                (8, 4),
                name=f'alias_{idx}',
                requires_grad=True,
                dtype=torch.float32,
            ).tosub())
        alias.device = [0]
        nodes.append(alias)
        output = alias.output(0)
    segment = IRSegment(nodes, [inp], [output])
    boundary = segment.output(0)
    if dummy_boundary:
        boundary = DummyInputOuput(
            boundary,
            0,
            is_input=True,
        ).input(0)
    adapter = _make_send_adapter(boundary)
    IRCell.make_pair(adapter, IRAdapter([], []))
    return segment, adapter


def test_tensor_name():
    repr_expr = CodeEmission().tensor_name
    assert repr_expr(1, 'model.') == '1'
    assert repr_expr('1') == "'1'"

    assert repr_expr(IRObject('name', 111, 'value'), 'model.') == 'name_111'
    assert repr_expr(IRObject('name', 111, 'value').as_attr(), 'model.') == 'model.name_111'
    assert repr_expr((IRObject('name', 111, 'value').as_attr(),), 'model.') == '(model.name_111,)'

    assert repr_expr(slice(1, None, IRObject('name', 111, 'value').as_attr()), 'model.') == 'slice(1, None, model.name_111)'
    assert repr_expr({'a': 1, 'b': IRObject('name', 111, 'value')}, 'model.') == "{'a': 1, 'b': name_111}"
    assert repr_expr([1], 'model.') == '[1]'
    assert repr_expr((1,), 'model.') == '(1,)'

    assert repr_expr((1,...), ) == '(1, Ellipsis)'

    with pytest.raises(ValueError):
        from datetime import datetime
        repr_expr(datetime.now())



def test_emit_module_attr():
    dropout = Dropout(IRFullTensor([1024, 1024], requires_grad=True), p=0.5, training=IRValue('self.training'), signature='torch.nn.functional.dropout')
    code = FuncEmission().emit_fnode(dropout, runtime_devid=0, plan_ndevs=1, runtime_ndevs=1)
    print(code)
    assert 'training=self.training' in code[0]


def test_emit_pipeline_output_release_after_send(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    adapter = _make_send_adapter()
    code = '\n'.join(FuncEmission().emit_adapter(
        adapter,
        async_op=True,
        pseudo_free_source_tids={adapter.input(0).tid},
    ))

    assert 'release_after_send=x_' in code
    assert 'release_after_send=x_chunk' not in code


def test_emit_pipeline_output_release_after_send_disabled_by_default(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', False)
    adapter = _make_send_adapter()
    code = '\n'.join(FuncEmission().emit_adapter(
        adapter,
        async_op=True,
        pseudo_free_source_tids={adapter.input(0).tid},
    ))

    assert 'release_after_send=' not in code


def test_pipeline_output_release_requires_safe_source_allowlist(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    code = '\n'.join(FuncEmission().emit_adapter(
        _make_send_adapter(),
        async_op=True,
    ))

    assert 'release_after_send=' not in code


def test_pipeline_output_release_selects_single_boundary_consumer(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    segment, adapter = _make_pipeline_boundary()

    eligible = ModuleCodeGen._pipeline_pseudo_free_source_tids(
        [segment, adapter],
        pipeline_enabled=True,
    )

    assert eligible == {adapter.input(0).tid}


def test_module_codegen_marks_safe_pipeline_output(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    segment, adapter = _make_pipeline_boundary(dummy_boundary=True)
    graph = IRGraph(
        [segment, adapter],
        [segment.input(0)],
        [],
        'pipeline_pseudo_free_test',
    )
    SchedulePlan(graph, 1)
    execplan = ExecutionPlan(graph, [segment, adapter])

    code = ModuleCodeGen(execplan).gen(0)

    release_lines = [
        line for line in code.splitlines()
        if 'release_after_send=' in line
    ]
    assert len(release_lines) == 1
    assert 'release_after_send=produced_' in release_lines[0]


def test_pipeline_output_release_skips_later_forward_consumer(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    segment, adapter = _make_pipeline_boundary()
    local_op = Dropout(
        adapter.input(0),
        0.0,
        True,
        False,
        signature='torch.nn.functional.dropout',
    )
    local_op.set_output(0, IRFullTensor(
        (8, 4), name='local', requires_grad=True, dtype=torch.float32
    ).tosub())
    local_op.device = [0]
    local_segment = IRSegment(
        [local_op],
        [adapter.input(0)],
        [local_op.output(0)],
    )

    eligible = ModuleCodeGen._pipeline_pseudo_free_source_tids(
        [segment, adapter, local_segment],
        pipeline_enabled=True,
    )

    assert eligible == set()


def test_pipeline_output_release_skips_multiref_alias(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    segment, adapter = _make_pipeline_boundary(
        alias_output=True,
        dummy_boundary=True,
    )

    eligible = ModuleCodeGen._pipeline_pseudo_free_source_tids(
        [segment, adapter],
        pipeline_enabled=True,
    )

    assert eligible == set()
