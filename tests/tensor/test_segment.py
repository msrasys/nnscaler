"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/tensor/test_segment.py
"""

from cube.tensor.logic.tensor import LogicalTensor
from cube.tensor.segment import Segment
from cube.tensor.indices import BaseIndices, TileIndices
from cube.device.physic.group import DeviceGroup

import torch
import os
torch.manual_seed(121)


def test_segment_init():

    tensor = LogicalTensor((10,10,10))

    anchor = [3,4,5]
    ofst = [2,4,3]
    indices = TileIndices(anchor, ofst)

    segment = Segment(tensor, indices, ofst)

    assert segment.logical_tensor is tensor
    assert segment.shape == tuple(ofst)
    assert segment.physical_tensor is None
    assert len(segment.placement) == 0
    assert segment.group is None
    assert segment.deploy_op is None
    assert segment.materialized is False
    assert segment.merge_op is None


def test_segment_deploy():

    myrank = DeviceGroup().rank
    tensor = LogicalTensor((10,10,10))

    anchor = [3,4,5]
    ofst = [2,4,3]
    indices = TileIndices(anchor, ofst)

    segment = Segment(tensor, indices, ofst)

    ranks = [0,2]
    segment.deploy(ranks, value_map_op=None)

    physical_tensor = segment.get_physical_tensor()
    tensor_ref = tensor.data[indices.get()].cuda()
    if myrank in ranks:
        assert physical_tensor.device == torch.device('cuda:{}'.format(myrank))
        assert torch.allclose(physical_tensor, tensor_ref)
    else:
        assert physical_tensor is None
    assert segment.placement == ranks
    assert segment.group == DeviceGroup().get_group(ranks)
    assert segment.deploy_op is None
    assert segment.materialized is True
    assert segment.merge_op is None


def test_segment_recover():

    myrank = DeviceGroup().rank
    tensor = LogicalTensor((10,10,10))

    anchor = [3,4,5]
    ofst = [2,4,3]
    indices = TileIndices(anchor, ofst)

    segment = Segment(tensor, indices, ofst)

    ranks = [0,2]
    segment.deploy(ranks, value_map_op=lambda tensor: tensor / 2)

    # deploy check
    physical_tensor = segment.get_physical_tensor()
    tensor_ref = tensor.data[indices.get()].cuda() / 2
    if myrank in [0,2]:
        assert physical_tensor.device == torch.device('cuda:{}'.format(myrank))
        assert torch.allclose(physical_tensor, tensor_ref) is True
    else:
        assert physical_tensor is None

    # recover to get logical value
    def reduction_op(tensor, group):
        torch.distributed.all_reduce(tensor, group=group)
    segment.recover(reduction_op=reduction_op)
    physical_tensor = segment.get_physical_tensor()
    
    tensor_ref = tensor.data[indices.get()].cuda()
    if myrank in [0,2]:
        assert physical_tensor.device == torch.device('cuda:{}'.format(myrank))
        assert torch.allclose(physical_tensor, tensor_ref) is True
    else:
        assert physical_tensor is None


if __name__ == '__main__':

    group = DeviceGroup()

    test_segment_init()
    test_segment_deploy()
    test_segment_recover()
