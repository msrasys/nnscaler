from typing import Optional
import torch
from torch import Tensor
from torch.overrides import has_torch_function_variadic, handle_torch_function


def linear_op(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight):
        print('note: this branch should not pass')
        return handle_torch_function(linear, (input, weight), input, weight, bias=bias)

    # single GPU version
    if True:
        output = torch._C._nn.linear(input, weight, bias)

    # multi-GPU version 
    #   - Assume input is full
    #   - split cloumn: Y = XA + b where A = [A_1, ..., A_p]
    elif True:
        # forward: identity; backward: allreduce
        input = copy_to_tensor_model_parallel_region(input)
        output = torch._C._nn.linear(input, weight, bias)
        # forward: allgather; backward: scatter
        output = gather_from_tensor_model_parallel_region(output)
    
    # multi-GPU version
    #   - Assume input is full
    #   - split row: Y = XA + b where X = [X_1, ..., X_p], A = [A_1 || ... || A_p]
    elif True:
        # forward: scatter; backward: allgather
        input = scatter_to_tensor_model_parallel_region(input)
        output = torch._C._nn.linear(input, weight, bias)
        # forward: reduce; backward: identity
        output = reduce_from_tensor_model_parallel_region(output)

    return output