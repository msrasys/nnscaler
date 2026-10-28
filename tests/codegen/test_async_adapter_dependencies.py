# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
import nnscaler

from nnscaler.codegen.emit import FuncEmission
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import AllGatherPrim, SelectPrim
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.executor import AsyncCommHandler


@pytest.mark.parametrize('transforms', [1, 2])
def test_generated_adapter_waits_before_local_transform(monkeypatch, transforms):
    def tensor(name, size):
        return IR.set_object_device(IRFullTensor((size,), name=name, dtype=torch.float32).tosub(), 0)

    source = tensor('source', 2)
    gathered = tensor('gathered', 4)
    outputs = [tensor(f'selected_{i}', 2) for i in range(transforms)]
    adapter = IRAdapter([source], [outputs[-1]])
    adapter.device = [0]
    prims = [AllGatherPrim([source], [gathered], dim=0, ranks=[0, 1])]
    previous = gathered
    for i, out in enumerate(outputs):
        prims.append(SelectPrim(previous, ((2, 4),) if i == 0 else ((0, 2),), (0, 1), out))
        previous = out
    adapter.prims = prims
    codegen = FuncEmission()
    code = '\n'.join(codegen.emit_adapter(adapter, async_op=True))
    events = []

    class Work:
        def wait(self):
            events.append('wait')

    def gather(value, dim, ranks, async_op=False):
        assert async_op  # Keep communication asynchronous up to its consumer.
        events.append('start')
        placeholder = torch.zeros(4)
        AsyncCommHandler().submit(placeholder, [Work()],
                                  lambda _: torch.cat([value, value + 10]))
        return placeholder

    monkeypatch.setattr(nnscaler.runtime.adapter, 'all_gather', gather)
    namespace = {'nnscaler': nnscaler, codegen.tensor_name(source): torch.tensor([1., 3.])}
    AsyncCommHandler().clear()
    try:
        exec(code, namespace)
        torch.testing.assert_close(namespace[codegen.tensor_name(outputs[-1])], torch.tensor([11., 13.]))
        assert events == ['start', 'wait']
        AsyncCommHandler().check_clear()
    finally:
        AsyncCommHandler().clear()
