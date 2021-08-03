"""
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=6000 \
    --use_env \
    examples/linear.py
"""

import cube
from cube import nn
from cube.tensor.logic.tensor import LogicalTensor
from cube.device.physic.group import DeviceGroup

import torch
import argparse

import z3

torch.manual_seed(100)


# Expert Policy

def select_policy(holistic_ops, outputs, *args, **kwargs):
    """
    Args:
        Candidates: holistic_ops
        *args, **kwargs: op input
    """
    return holistic_ops.get_op(0, outputs, *args, **kwargs)


def segment_policy(holist_op, input, weight, bias):
    """
    Args:
        holistic_op (HolisticOp)
        *args, **kwargs: op input
    """
    solver = holist_op.solver
    attributes = holist_op.attributes
    input_layout = holist_op.input_layouts[0]
    weight_layout = holist_op.input_layouts[1]
    bias_layout = holist_op.input_layouts[2]
    output_layout = holist_op.output_layouts[0]

    # add restrictions based on device num
    holist_op.add_constraint(weight_layout.chunk_num == 4)
    
    # iterate all configs
    configs = list()
    while solver.check() == z3.sat:
        config = solver.model()
        if DeviceGroup().rank == 0:
            print('find config: {}'.format(config))
        configs.append(config)
        solver.add(
            z3.Or([z3.Not(attr == config[attr]) for attr in attributes])
        )
        if len(attributes) == 0:
            break
    # choose one config -- suppose to the first
    config = configs[0]
    if DeviceGroup().rank == 0:
        print('selected config: {}'.format(config))
    # deploy decisions
    chunk_num = config[weight_layout.chunk_num].as_long()
    input_ranks = [list(range(0, chunk_num)),]
    weight_ranks = list()
    for rank in range(chunk_num):
        weight_ranks.append([rank])
    bias_ranks = weight_ranks
    return config, [input_ranks, weight_ranks, bias_ranks]


cube.operator.logic.linear.Linear.set_default_policy(select_policy)
cube.operator.holist.linear.LinearColumnWeight.set_default_policy(segment_policy)



# User Network
class SingleLinear(nn.Module):

    def __init__(self, dim, mult):
        super().__init__()
        self.net = nn.Linear(dim, dim * mult)
    
    def forward(self, x):
        output = self.net(x)
        return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--mult', type=int, default=16)
    args = parser.parse_args()

    # init distributed env
    rank = DeviceGroup().rank

    model = SingleLinear(args.dim, args.mult)

    inputs = LogicalTensor((args.bs, args.dim))
    output = model(inputs)

    assert isinstance(output, LogicalTensor)
    assert torch.is_tensor(output.get_physical_tensor(rank))
    print('Done.')
