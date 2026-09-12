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
from nnscaler.ir.adapter.prim import ChunkPrim, MovePrim


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


def test_async_receive_waits_before_local_chunk(monkeypatch):
    import nnscaler

    full = IRFullTensor((8,), name='received', dtype=torch.float32).tosub()
    part = full.parent.select(((0, 4),), (0, 1))
    adapter = IRAdapter([], [part])
    adapter.device = [1]
    full.cell = adapter
    adapter.prims = [
        MovePrim([], [full], shape=(8,), dtype='torch.float32', src=0, dst=1),
        ChunkPrim([full], [part], dim=0, ranks=[1, 2]),
    ]
    pending = torch.zeros(8)
    events = []
    from nnscaler.runtime.executor import _AsyncCommHandler
    handler = _AsyncCommHandler()
    monkeypatch.setattr(nnscaler.runtime.executor, '_instance', handler)

    class Work:
        def wait(self):
            events.append('wait')
            pending.copy_(torch.arange(8, dtype=torch.float32))

    def receive(*args, **kwargs):
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
        assert events == ['receive']
        namespace['__pending'] = namespace[emitter.tensor_name(part)]
        exec('\n'.join(emitter.emit_async_recv_adapter_wait(adapter)), namespace)
        assert events == ['receive', 'wait', 'chunk']
        assert torch.equal(namespace[emitter.tensor_name(part)], torch.arange(4, dtype=torch.float32))
        handler.check_clear()
    finally:
        handler.drain()
