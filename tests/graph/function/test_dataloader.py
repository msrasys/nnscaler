"""
pytest unit_tests/graph/function/test_dataloader.py
"""

import torch

from cube.ir.cten import IRObject
from cube.ir.tensor import IRFullTensor
from cube.ir.operator import IRDataOperation
from cube.runtime.utils import create_dummy_dataloader


def test_dummy_dataloader():
    samples = (
        torch.rand([256, 512], dtype=torch.float32),
        torch.rand([128, 224], dtype=torch.float16),
        4,
    )
    dataloader = create_dummy_dataloader(samples, batch_size=32)
    for idx, samples in enumerate(dataloader):
        assert samples[0].shape == torch.Size([32, 256, 512])
        assert samples[1].shape == torch.Size([32, 128, 224])
        assert torch.allclose(samples[2], torch.tensor([4] * 32, dtype=torch.int64))
        if idx == 4:
            break


def test_data_operation():

    data_op = IRDataOperation(
        IRObject('dataloader'),
        [IRFullTensor(shape=[32, 256, 512]).tosub(),
         IRFullTensor(shape=[32, 128, 224]).tosub(),])
    
    # cannot be partitioned
    assert not hasattr(data_op, 'algorithms')
    # test input / output
    assert all(isinstance(out, IRObject) for out in data_op.outputs())
    assert all(isinstance(inp, IRObject) for inp in data_op.inputs())
    # can be replicated
    data_op_replica = data_op.replicate()
    assert data_op_replica.input(0) == data_op.input(0)
    assert data_op_replica.output(0) == data_op.output(0)
    assert data_op_replica.output(1) == data_op.output(1)
    assert data_op_replica.cid == data_op.cid
