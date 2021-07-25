"""
Test this with:

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=6000 \
    --use_env \
    tests/test_group.py
"""

from cube.device.physic.group import DeviceGroup

import torch


def test_sub_group():

    group = DeviceGroup()
    myrank = group.rank
    sub_group_1 = group.get_group([0,2])
    if myrank in [0,2]:
        assert torch.distributed.get_rank(sub_group_1) in [0,1]
    else:
        assert torch.distributed.get_rank(sub_group_1) == -1
    
    sub_group_2 = group.get_group([1,3])
    if myrank in [1,3]:
        assert torch.distributed.get_rank(sub_group_2) in [0,1]
    else:
        assert torch.distributed.get_rank(sub_group_2) == -1
    # print(group)


if __name__ == '__main__':

    # init distributed
    group = DeviceGroup()

    test_sub_group()
