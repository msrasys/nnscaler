"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/tensor/test_logical_tensor.py
"""

from cube.tensor.indices import BaseIndices
from cube.tensor.logic.tensor import LogicalTensor
from cube.tensor.segment import Segment
from cube.device.physic.group import DeviceGroup

import torch


def test_logical_tensor_init():

    tensor = LogicalTensor(shape=(10,10,10))
    assert tensor.shape == (10, 10, 10)
    assert len(tensor.segments) == 0
    assert tensor.data is not None
    assert tensor.data.size() == torch.Size([10,10,10])


def test_logical_tensor_select():
    tensor = LogicalTensor(shape=(10,10,10))
    sparse_indices = (
        [2,3,1,4],
        [0,4,8,4],
        [7,5,9,4]
    )
    indices = BaseIndices(sparse_indices)
    segment = tensor.select(indices, shape=(2,2))
    assert isinstance(segment, Segment)
    assert segment.materialized is False


def test_logical_tensor_fill():

    myrank = DeviceGroup().rank

    tensor = LogicalTensor(shape=(10,10,10), init_data=False)
    sparse_indices = (
        [2,3,1,4],
        [0,4,8,4],
        [7,5,9,4]
    )
    indices = BaseIndices(sparse_indices)
    segment = tensor.select(indices, shape=(2,2))
    tensor.add_segment(segment)

    assert segment.materialized is False
    assert len(tensor.segments) == 1

    ranks = [1, 3]
    if myrank in ranks:
        phy_tensor = torch.randn((2,2)).cuda()
    else:
        phy_tensor = None
    tensor.fill([phy_tensor], [ranks])
    assert segment.materialized is True
    if myrank in ranks:
        assert tensor.get_physical_tensor(0) is not None
    else:
        assert tensor.get_physical_tensor(0) is None


def test_logical_tensor_transform():

    tensor = LogicalTensor(shape=(10,10,10))
    sparse_indices = (
        [2,3,1,4],
        [0,4,8,4],
        [7,5,9,4]
    )
    indices = BaseIndices(sparse_indices)
    segment = tensor.select(indices, shape=(2,2))

    ranks = [0,1,3]
    tensor.transform([segment], [ranks], [None])

    myrank = DeviceGroup().rank
    if myrank in ranks:
        assert tensor.get_physical_tensor(0) is not None
    else:
        assert tensor.get_physical_tensor(0) is None


if __name__ == '__main__':

    group = DeviceGroup()

    test_logical_tensor_init()
    test_logical_tensor_select()
    test_logical_tensor_fill()
    test_logical_tensor_transform()