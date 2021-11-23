r"""
SU Executor for runtime
"""

from typing import Tuple, Any, Callable, List
import torch


def fexecute(su: Callable, *input_tensors: Tuple[Any]):
    """
    forward the SUs
    """
    outputs = su(*input_tensors)
    # print('forwarding... ')
    return outputs


def backward(input_tensors: List[torch.Tensor], output_tensors, output_tensor_grads):
    """
    Backward the SUs
    """
    # for tensor in input_tensors:
    #     if torch.is_tensor(tensor) and tensor.requires_grad:
    #         tensor.retain_grad()

    if len(output_tensor_grads) != len(output_tensors):
        raise RuntimeError(
            "Expected same length of out tensors and grads"
        )

    inputs = list()
    indices = list()
    for idx, input in enumerate(input_tensors):
        if torch.is_tensor(input) and input.requires_grad:
            inputs.append(input)
            indices.append(idx)
    
    grads = [None] * len(input_tensors)
    if len(inputs) != 0:
        # print('backwarding... ')
        in_grads = torch.autograd.grad(output_tensors, inputs, output_tensor_grads)
        for idx, grad in zip(indices, in_grads):
            tensor = input_tensors[idx]
            if isinstance(tensor, torch.nn.Parameter):
                if tensor.grad is not None:
                    tensor.grad += grad
                else:
                    tensor.grad = grad
            grads[idx] = grad

    # if len(inputs) != 0:
    #     torch.autograd.backward(
    #         output_tensors,
    #         grad_tensors=output_tensor_grads,
    #         inputs=inputs
    #     )
    #     for idx, tensor in zip(indices, inputs):
    #         grads[idx] = tensor.grad

    # torch.autograd.backward(output_tensors, grad_tensors=output_tensor_grads)
    # grads = list()
    # for tensor in input_tensors:
    #     # print('backward input tensor: {}'.format(tensor))
    #     if torch.is_tensor(tensor) and tensor.requires_grad:
    #         grads.append(tensor.grad)
    #     else:
    #         grads.append(None)

    if    len(grads) == 0: return None
    elif  len(grads) == 1: return grads[0]
    else: return tuple(grads)
