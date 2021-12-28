r"""
Executor for runtime
"""

from typing import Tuple, Any, Callable, List
import torch


def fexecute(subgraph: Callable, *input_tensors: Tuple[Any]):
    """
    forward the sub-graph.
    """
    outputs = subgraph(*input_tensors)
    # print('forwarding... ')
    return outputs


def backward(input_tensors: List[torch.Tensor], output_tensors, output_tensor_grads):
    """
    Backward Procedure.

    input_tensors: List[torch.Tensor]:
        tensors that their gradient need to be computed, including parameters.
        Correspoinding forward input tensors.
    
    output_tensors:
        tensors that start for gradient backward computation.
        Corresponding to forward output tensors.

    output_tensor_grads:
        gradient tensors corresponding to output_tensors.

    Returns:
        gradient in order of non-parameter tensors in input_tensors.
        (Note parameter tnesors already have gradient accumulated at .grad attribute)
    """
    # print(f'{torch.distributed.get_rank()}: backwarding... ')
    inputs = list()
    for input in enumerate(input_tensors):
        # skip returning gradients of parameters
        if torch.is_tensor(input) and not isinstance(input, torch.nn.Parameter):
            inputs.append(inputs)
    
    grads = list()
    if len(inputs) != 0:
        in_grads = torch.autograd.grad(
            output_tensors, inputs, output_tensor_grads, allow_unused=True)
        for tensor, grad in zip(inputs, in_grads):
            if isinstance(tensor, torch.nn.Parameter):
                if tensor.grad is not None:
                    tensor.grad += grad
                else:
                    tensor.grad = grad
            else:
                grads.append(grad)

    if    len(grads) == 0: return None
    elif  len(grads) == 1: return grads[0]
    else: return tuple(grads)


def backwardV2(input_tensors: List[torch.Tensor], output_tensors, output_tensor_grads):
    inputs = list()
    for input in enumerate(input_tensors):
        # skip returning parameters
        if torch.is_tensor(input) and not isinstance(input, torch.nn.Parameter):
            inputs.append(inputs)
    for tensor in input_tensors:
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.retain_grad()
    torch.autograd.backward(
        output_tensors,
        grad_tensors=output_tensor_grads,
        inputs=input_tensors
    )
    grads = [input.grad for input in inputs]
    return grads
