from typing import Tuple, Any
import torch


def forward(model, *input_tensors: Tuple[Any]):
    """
    forward the model
    """
    outputs = model(*input_tensors)
    print('forwarding... ')
    return outputs


def backward(input_tensors, output_tensors, output_tensor_grads):
    """
    Backward on the tensors
    """
    for tensor in input_tensors:
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.retain_grad()

    # TODO: gen code should contain None in output_tensor_grads
    if len(output_tensor_grads) != len(output_tensors):
        output_tensor_grads = [None] * len(output_tensors)

    for tensor, grads in zip(output_tensors, output_tensor_grads):
        print('backwarding... ')
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
