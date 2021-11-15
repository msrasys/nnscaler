r"""
SU Executor for runtime
"""

from typing import Tuple, Any, Callable
import torch


def fexecute(su: Callable, *input_tensors: Tuple[Any]):
    """
    forward the SUs
    """
    outputs = su(*input_tensors)
    # print('forwarding... ')
    return outputs


def backward(input_tensors, output_tensors, output_tensor_grads):
    """
    Backward the SUs
    """
    for tensor in input_tensors:
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.retain_grad()

    if len(output_tensor_grads) != len(output_tensors):
        raise RuntimeError(
            "Expected same length of out tensors and grads"
        )

    for tensor, grads in zip(output_tensors, output_tensor_grads):
        # print('backwarding... ')
        torch.autograd.backward(tensor, grad_tensors=grads)
    grads = list()
    for tensor in input_tensors:
        # print('backward input tensor: {}'.format(tensor))
        if torch.is_tensor(tensor) and tensor.requires_grad:
            grads.append(tensor.grad)
        else:
            grads.append(None)
    if    len(grads) == 0: return None
    elif  len(grads) == 1: return grads[0]
    else: return tuple(grads)
