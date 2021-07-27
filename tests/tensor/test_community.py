"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/tensor/test_community.py
"""

from cube.tensor.community import Community
import cube.tensor.logic.segment as segment
from cube.device.physic.group import DeviceGroup

import torch
import os
torch.manual_seed(121)


def test_community_init():

    tensor = torch.randn((10,10,10))
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4), reduction=segment.ReductionOp.Replica)
    community = Community(seg)

    assert community.segment == seg
    assert community.physical_tensor is None
    assert len(community.group) == 0
    assert community.materialized is False


def test_community_deploy():

    tensor = torch.randn((10,10,10))
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4),
        reduction=segment.ReductionOp.Replica)
    community = Community(seg)

    # policy for scaling out
    # using torch.Tensor to test
    ranks = [0,2]
    community.deploy(ranks, tensor, None)

    # check
    myrank = DeviceGroup().rank
    if myrank not in ranks:
        assert community.physical_tensor is None
    else:
        sub_tensor = community.physical_tensor
        assert torch.is_tensor(sub_tensor)
        assert sub_tensor.size() == torch.Size([4,4,4])
        assert sub_tensor.device == torch.device('cuda:{}'.format(myrank))
        assert torch.all(torch.eq(sub_tensor.cpu(), tensor[seg.get_indices()]))
        assert torch.distributed.get_world_size(community.group) == 2


def test_community_sync():
    tensor = torch.randn((10,10,10))
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4),
        reduction=segment.ReductionOp.Sum)
    community = Community(seg)

    # deploy with value modification
    ranks = [0,2]
    community.deploy(ranks, tensor, 
                     value_map_fn=lambda tensor: tensor / 2)

    # check
    sub_tensor = community.get_physical_tensor()
    ref_tensor = tensor[seg.get_indices()].cuda()
    myrank = DeviceGroup().rank
    if myrank in ranks:
        assert torch.all(torch.eq(sub_tensor, ref_tensor / 2))
    
    # sync to get logical value
    community.sync()
    sub_tensor = community.get_physical_tensor()
    if myrank not in ranks:
        assert sub_tensor is None
    else:
        # print('ref: {}'.format(ref_tensor))
        assert torch.allclose(sub_tensor, ref_tensor) is True


def test_community_set_physical_tensor():
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4),
        reduction=segment.ReductionOp.Sum)
    community = Community(seg)

    tensor = torch.randn((4,4,4))
    community.set_physical_tensor(tensor, [0,1,2])
    assert community.materialized is True
    assert community.group == DeviceGroup().get_group([0,1,2])
    assert community.physical_tensor is tensor


if __name__ == '__main__':

    group = DeviceGroup()
    torch.distributed.barrier()

    test_community_init()
    test_community_deploy()
    test_community_sync()
    test_community_set_physical_tensor()