import torch
import os

from functools import partial

torch.manual_seed(121)

# select from logical tensor with indices -> generate a logical tensor
def select(tensor, indices, val_op, shape): pass

# deploy logical tensor to devices
def deploy(tensor, ranks): pass

# merge logical tensors at `ranks` devices
def merge(tensor, ranks, merge_op): pass


class LogicalTensor: pass
class PhyiscalTensor: pass


def linear_tensor_parallel(inputs, weight, bias, output):
    """
    inputs: (M, K)
    weight: (N, K)
    bias: (N,)
    output: (M, N)

    Perform: (M, K) * (\delta N, K) + (\delta N,) = (M, \delta N)
    """

    M = 1024
    K = 1024
    N = 1024

    # Tensor Split #  -- System + policy generated
    inputs = select(
        tensor = inputs, 
        indices = (slice(0, M), slice(0, K)),
        val_op = None,
        shape = (M, K)
    )

    weights, biases, outputs = list(), list(), list()
    for cid in range(4):
        weights.append(select(
            tensor = weight,
            indices = (slice(cid * (N // 4), (cid + 1) * (N // 4)), slice(0, K)),
            val_op = None,
            shape = (N // 4, K)
        ))

        biases.append(select(
            tensor = bias,
            indices = (slice(cid * (N // 4), (cid + 1) * (N // 4)),),
            val_op = None,
            shape = (N // 4,)
        ))

        outputs.append(select(
            tensor = output,
            indices = (slice(slice(0, M), cid * (N // 4), (cid + 1) * (N // 4)),),
            val_op = None,
            shape = (M, N // 4)
        ))
    # Tensor Split #

    # Tensor Deployment # -- System + policy generated
    inputs = deploy(
        segment = inputs,
        ranks = [0, 1, 2, 3]
    )

    for rank, (weight, bias) in enumerate(zip(weights, biases)):
        weight = deploy(
            segment = weight,
            ranks = [rank],
        )
        bias = deploy(
            segment = bias,
            ranks = [rank],
        )
    # Tensor Deployment #

    # Compute #  -- Expert specified
    for weight, bias, output in enumerate(zip(weights, biases, outputs)):
        # physical tensor
        chunk = torch._C._nn.linear(inputs, weight, bias)
        output.fill(chunk)

    # Generate logical tensor -- System generated
    merge(
        tensor = output,
        ranks = [0, 1, 2, 3],
        merge_op = partial(all_gather, dim=1)
    )
