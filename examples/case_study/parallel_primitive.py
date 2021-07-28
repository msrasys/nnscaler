import torch
import os
from cube.device.physic.grou import DeviceGroup()

torch.manual_seed(121)


def linear_tensor_parallel(inputs, weight, bias):

    rank = DeviceGroup().rank

    M = 1024
    K = 1024
    N = 1024

    ### ============ Input Adapter ============ ###

    # select need to consider transformation from one segmentation to another
    inputs_segment = select(
        tensor = inputs, 
        indices = (slice(0, M), slice(0, K)),
        shape = (M, K)
    )

    weight_segment = select(
        tensor = weight,
        indices = (slice(rank * (N // 4), (rank + 1) * (N // 4)), slice(0, K)),
        shape = (N // 4, K)
    )

    bias_segment = select(
        tensor = bias,
        indices = (slice(rank * (N // 4), (rank + 1) * (N // 4)),),
        shape = (N // 4,)
    )

    inputs = deploy(
        segment = inputs_segment,
        ranks = [0, 1, 2, 3],
        val_map_op = IdentityForwardAllreduceBackward
    )

    weight = deploy(
        segment = weight_segment,
        ranks = [rank],
        val_map_op = None
    )

    bias = deploy(
        segment = bias_segment,
        ranks = [rank],
        val_map_op = None
    )
    ### ============ Input Adapter ============ ###


    ### ============ Compute ============ ###
    output = torch._C._nn.linear(inputs, weight, bias)
    ### ============ Compute ============ ###


    ### ============ Output Adapter ============ ###
    segment = recover(
        tensor = output,
        ranks = [0, 1, 2, 3],
        reduction_op = AllGatherForwardSplitBackward
    )
    # construct to logical tensor and return
    ### ============ Output Adapter ============ ###
