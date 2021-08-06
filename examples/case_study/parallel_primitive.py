import torch
import os
from cube.device.physic.grou import DeviceGroup()

torch.manual_seed(121)


def linear_tensor_parallel(inputs, weight, bias):

    rank = DeviceGroup().rank

    M = 1024
    K = 1024
    N = 1024

    inputs = select(
        tensor = inputs, 
        indices = (slice(0, M), slice(0, K)),
        val_op = None,
        ranks = [0, 1, 2, 3],
        shape = (M, K)
    )

    weight = select(
        tensor = weight,
        indices = (slice(rank * (N // 4), (rank + 1) * (N // 4)), slice(0, K)),
        val_op = None,
        ranks = [0, 1, 2, 3],
        shape = (N // 4, K)
    )

    bias = select(
        tensor = bias,
        indices = (slice(rank * (N // 4), (rank + 1) * (N // 4)),),
        val_op = None,
        ranks = [0, 1, 2, 3],
        shape = (N // 4,)
    )

    ### ============ Compute ============ ###
    # each rank do this
    output = torch._C._nn.linear(inputs, weight, bias)
    ### ============ Compute ============ ###

    output = merge(
        tensor = output,
        ranks = [0, 1, 2, 3],
        merge_op = all_gather
    )



    # inputs = deploy(
    #     segment = inputs_segment,
    #     ranks = [0, 1, 2, 3],
    #     val_map_op = IdentityForwardAllreduceBackward
    # )

    # weight = deploy(
    #     segment = weight_segment,
    #     ranks = [rank],
    #     val_map_op = None
    # )

    # bias = deploy(
    #     segment = bias_segment,
    #     ranks = [rank],
    #     val_map_op = None
    # )