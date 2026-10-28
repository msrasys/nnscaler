# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta

import pytest
import torch
import nnscaler

from nnscaler.codegen.emit import FuncEmission
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import AllGatherPrim, SelectPrim
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.executor import Executor, AsyncCommHandler
from tests.launch_torchrun import launch_torchrun


def _boundary_collective_worker():
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=90))
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    try:
        for collective in ('all_gather', 'all_to_all'):
            for split in (False, True):
                source = torch.full((2, 4), float(rank), device='cuda', requires_grad=True)
                if collective == 'all_gather':
                    pending = nnscaler.runtime.adapter.all_gather(source, 0, [0, 1], async_op=True)
                    expected = torch.cat([torch.full_like(source, float(r)) for r in (0, 1)])
                else:
                    pending = nnscaler.runtime.adapter.all_to_all(source, 0, 1, [0, 1], async_op=True)
                    expected = torch.cat([torch.full_like(source[:, :2], float(r)) for r in (0, 1)])
                callbacks = []
                Executor.register_input_grad_callback(pending, callbacks.append)
                output = Executor.fexecute('async', lambda x, y: x.sin() + y.sin(), pending, pending)
                torch.testing.assert_close(output, 2 * expected.sin())
                if split:
                    grads = Executor.backward_input('async', [pending] * 2, [output], [torch.ones_like(output)], [])
                    Executor.backward_weight('async', [])
                else:
                    grads = Executor.backward('async', [pending] * 2, [output], [torch.ones_like(output)])
                assert len(callbacks) == 1
                for grad in (*grads, callbacks[0]):
                    torch.testing.assert_close(grad, 2 * expected.cos())
                Executor.check_clear()
                AsyncCommHandler().check_clear()

        def tensor(name, size):
            return IR.set_object_device(IRFullTensor((size,), name=name, dtype=torch.float32).tosub(), rank)

        source, gathered, selected = tensor('source', 2), tensor('gathered', 4), tensor('selected', 2)
        adapter = IRAdapter([source], [selected])
        adapter.device = [rank]
        adapter.prims = [AllGatherPrim([source], [gathered], dim=0, ranks=[0, 1]),
                         SelectPrim(gathered, ((2, 4),), (0, 1), selected)]
        emission = FuncEmission()
        code = '\n'.join(emission.emit_adapter(adapter, async_op=True))
        namespace = {'nnscaler': nnscaler,
                     emission.tensor_name(source): torch.tensor([1., 3.], device='cuda') + rank * 10}
        exec(code, namespace)
        torch.testing.assert_close(namespace[emission.tensor_name(selected)],
                                   torch.tensor([11., 13.], device='cuda'))
        AsyncCommHandler().check_clear()
        torch.distributed.barrier()
    finally:
        Executor.clear()
        AsyncCommHandler().clear()
        torch.distributed.destroy_process_group()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
def test_async_collective_boundaries_and_local_consumer_2gpu():
    assert all(launch_torchrun(2, _boundary_collective_worker).values())
