import torch
import os

from functools import partial

torch.manual_seed(121)

class LogicalOp: pass
class PhyiscalOp: pass

class LogicalTensor: pass
class PhyiscalTensor: pass

# select from logical tensor with indices -> generate a logical tensor
def select(tensor: LogicalTensor, indices, val_map_op, shape) -> LogicalTensor: pass

# deploy logical tensor to devices
def deploy(tensor: LogicalTensor, ranks) -> list(PhyiscalTensor): pass

# merge logical tensors at `ranks` devices
def merge(tensor: LogicalTensor, ranks, val_reduce_op): pass

# tensor movement: move physical tensor to rank
def move(tensor: PhyiscalTensor, rank): pass

# tensor release: release the data in physical tensor inside tensor
def release(tensor: PhyiscalTensor): pass

# tensor re-genrewate: bring back the data for the physical tensor
def generate(tensor: PhyiscalTensor, rank): pass



## =============== tensor parallelism on matmul ============== ##

def all_gather(tensors, dim): pass


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

    # Tensor split -- system + policy generated
    inputs = select(
        tensor = inputs, 
        indices = (slice(0, M), slice(0, K)),
        val_map_op = None,
        shape = (M, K)
    )

    weights, biases, outputs = list(), list(), list()
    for cid in range(4):
        weights.append(select(
            tensor = weight,
            indices = (slice(cid * (N // 4), (cid + 1) * (N // 4)), slice(0, K)),
            val_map_op = None,
            shape = (N // 4, K)
        ))

        biases.append(select(
            tensor = bias,
            indices = (slice(cid * (N // 4), (cid + 1) * (N // 4)),),
            val_map_op = None,
            shape = (N // 4,)
        ))

        outputs.append(select(
            tensor = output,
            indices = (slice(slice(0, M), cid * (N // 4), (cid + 1) * (N // 4)),),
            val_map_op = None,
            shape = (M, N // 4)
        ))

    # Algorithm -- Expert specified
    for weight, bias, output in enumerate(zip(weights, biases, outputs)):
        # physical tensor
        chunk = torch._C._nn.linear(inputs, weight, bias)
        # physical tensor fill in to logical tensor
        output.fill(chunk)
    
    # Tensor deployment -- system + policy generated
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

    # Logical tensor merge -- system + policy generated
    merge(
        tensor = outputs,
        ranks = [0, 1, 2, 3],
        merge_op = partial(all_gather, dim=1)
    )


## =============== tensor movement / re-generation ============== ##

def custom_op(forward_fn, backward_fn): pass

def offload(inputs: PhyiscalTensor, weights: list(PhyiscalTensor), ops: list(PhyiscalOp)):
    """
    offload a feature_map after forward the 3rd op
    retrieve (prefetch) the feature_map after backward the 5th op
    """
    feature_maps = [inputs]
    offload_step = 2
    retrieve_step = 4
    for step, (weight, op) in enumerate(zip(weights, ops)):
        tensor = feature_maps[-1]
        # retrieve
        if step == retrieve_step:
            feature_maps[-1] = custom_op(
                forward_fn=partial((lambda input: input), input=feature_maps[-1]),
                backward_fn=partial(move, feature_maps[offload_step + 1], rank=0)
            )
        # op calculation
        out = op(tensor, weight)
        # offload
        if step == offload_step:
            move(tensor, rank=-1)
        feature_maps.append(out)


def checkpoint(inputs: PhyiscalTensor, weights: list(PhyiscalTensor), ops: list(PhyiscalOp)):
    """
    checkpoint a feature_map after forward the 3rd op
    re-generate (possible for packing with other operator) after backward the 5th op
    """
    feature_maps = [inputs]
    release_step = 2
    recompute_step = 4
    released_tensor = None
    for step, (weight, op) in enumerate(zip(weights, ops)):
        tensor = feature_maps[-1]
        # retrieve
        if step == recompute_step:
            feature_maps[-1] = custom_op(
                forward_fn=partial((lambda input: input), input=feature_maps[-1]),
                backward_fn=partial(generate, feature_maps[release_step + 1], rank=0)
            )
        # op calculation
        out = op(tensor, weight)
        # offload
        if step == release_step:
            release(tensor)
        feature_maps.append(out)
