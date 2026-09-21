#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import pytest
from nnscaler.codegen.emit import CodeEmission, IRValue
from nnscaler.ir.cten import IRObject
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function import Dropout
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import ChunkPrim, MovePrim, SumPrim


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
