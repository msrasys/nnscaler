#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.execplan import ExecutionPlan
from nnscaler.graph.function.function import MultiRef
from nnscaler.graph.gener.utils import DummyInputOuput
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.schedule.schedplan import SchedulePlan
from nnscaler.graph.segment import IRSegment
from nnscaler.flags import CompileFlag
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import ChunkPrim, MovePrim, SumPrim

import pytest
from nnscaler.codegen.emit import CodeEmission, IRValue
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function import Dropout
from nnscaler.ir.tensor import IRFullTensor


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


def test_emit_pipeline_output_release_after_send_can_be_disabled(monkeypatch):
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


@pytest.mark.parametrize('async_comm', [False, True])
def test_module_codegen_marks_safe_pipeline_output(monkeypatch, async_comm):
    monkeypatch.setattr(CompileFlag, 'pipeline_output_pseudo_free', True)
    monkeypatch.setattr(CompileFlag, 'async_comm', async_comm)
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
    assert len(release_lines) == int(async_comm)
    if async_comm:
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


@pytest.mark.parametrize('with_chunk', [False, True])
def test_async_receive_waits_at_first_consumer(monkeypatch, with_chunk):
    import nnscaler

    full = IRFullTensor((8,), name='received', dtype=torch.float32).tosub()
    part = full.parent.select(((0, 4),), (0, 1))
    adapter = IRAdapter([], [part if with_chunk else full])
    adapter.device = [1]
    full.cell = adapter
    adapter.prims = [
        MovePrim([], [full], shape=(8,), dtype='torch.float32', src=0, dst=1),
    ]
    if with_chunk:
        adapter.prims += [ChunkPrim([full], [part], dim=0, ranks=[1, 2])]
    pending = torch.zeros(8)
    events = []
    from nnscaler.runtime.executor import _AsyncCommHandler
    handler = _AsyncCommHandler()
    monkeypatch.setattr(nnscaler.runtime.executor, '_instance', handler)

    class Work:
        def is_completed(self):
            return False

        def wait(self):
            events.append('wait')
            pending.copy_(torch.arange(8, dtype=torch.float32))

    def receive(*args, **kwargs):
        assert kwargs['async_op'] is True
        events.append('receive')
        handler.submit(pending, [Work()])
        return pending

    def chunk(tensor, **kwargs):
        events.append('chunk')
        return tensor[:4]

    monkeypatch.setattr(nnscaler.runtime.adapter, 'move', receive)
    monkeypatch.setattr(nnscaler.runtime.adapter, 'chunk', chunk)
    emitter = FuncEmission()
    namespace = {'nnscaler': nnscaler, 'torch': torch}
    try:
        exec('\n'.join(emitter.emit_adapter(adapter, async_op=True)), namespace)
        result = namespace[emitter.tensor_name(adapter.output(0))]
        if with_chunk:
            assert events == ['receive', 'wait', 'chunk']
            assert torch.equal(result, torch.arange(4, dtype=torch.float32))
        else:
            # A receive without a local consumer remains asynchronous.
            assert events == ['receive']
            result = nnscaler.runtime.executor.fexecute(
                'consumer', lambda tensor: tensor * 2, result, requires_grad=False,
            )
            assert events == ['receive', 'wait']
            assert torch.equal(result, torch.arange(8, dtype=torch.float32) * 2)
        handler.check_clear()
    finally:
        handler.drain_sends()


def test_async_adapter_waits_for_each_merge_input(monkeypatch):
    import nnscaler
    from nnscaler.runtime.executor import _AsyncCommHandler

    inputs = [IRFullTensor((4,), name=f'received{index}', dtype=torch.float32).tosub() for index in range(2)]
    output = IRFullTensor((4,), name='merged', dtype=torch.float32).tosub()
    adapter = IRAdapter([], [output])
    adapter.device = [2]
    for tensor in inputs + [output]:
        tensor.cell = adapter
    adapter.prims = [
        MovePrim([], [tensor], shape=(4,), dtype='torch.float32', src=index, dst=2)
        for index, tensor in enumerate(inputs)
    ] + [SumPrim(inputs, output)]
    handler = _AsyncCommHandler()
    monkeypatch.setattr(nnscaler.runtime.executor, '_instance', handler)
    events = []

    def receive(*args, src, async_op, **kwargs):
        assert async_op
        pending = torch.zeros(4)
        events.append(('receive', src))

        class Work:
            def wait(self):
                events.append(('wait', src))
                pending.copy_(torch.arange(4, dtype=torch.float32) + 10 * src)

        handler.submit(pending, [Work()])
        return pending

    monkeypatch.setattr(nnscaler.runtime.adapter, 'move', receive)
    emitter = FuncEmission()
    namespace = {'nnscaler': nnscaler, 'torch': torch}
    exec('\n'.join(emitter.emit_adapter(adapter, async_op=True)), namespace)
    assert events == [('receive', 0), ('receive', 1), ('wait', 0), ('wait', 1)]
    torch.testing.assert_close(namespace[emitter.tensor_name(output)], torch.arange(4, dtype=torch.float32) * 2 + 10)
    handler.check_clear()
