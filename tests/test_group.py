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

from combo.physical.device.group import DeviceGroup



if __name__ == '__main__':

    # init distributed
    group = DeviceGroup()

    sub_group_1 = group.get_group([0,2])
    sub_group_2 = group.get_group([1,3])
    
    print(group)