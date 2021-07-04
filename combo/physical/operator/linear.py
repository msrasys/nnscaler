from typing import Optional
import torch
from torch import Tensor
from torch.overrides import has_torch_function_variadic, handle_torch_function

import combo.physical.operator.comm as comm
from combo.physical.device.group import DeviceGroup


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

    devices = [0, 1]
    rank = torch.distributed.get_rank(DeviceGroup().get_group(devices))

    # single GPU version
    if True:
        output = torch._C._nn.linear(input, weight, bias)

    # multi-GPU version 
    #   - Assume input is full
    #   - split cloumn of W: Y = XW + b where W = [W_1, ..., W_p]
    elif False:
        # get weight chunk
        weight = torch.chunk(weight, chunks=len(devices), dim=0)[rank].contiguous()
        if bias is not None:
            bias = torch.chunk(bias, chunks=len(devices), dim=0)[rank].contiguous()
        # forward: identity; backward: allreduce
        input = comm.parallel_in(input, ranks=devices)
        output = torch._C._nn.linear(input, weight, bias)
        # forward: allgather; backward: split
        output = comm.gather_out(output, dim=-1, ranks=devices)
    
    # multi-GPU version Y = XW + b
    #   - Assume input is full
    #   - split row of W, column of X: 
    #       - Y = X = [X_1, ..., X_p]
    #       - W = [W_1 // ... // W_p]
    elif False:
        # get weight chunk
        weight = torch.chunk(weight, chunks=len(devices), dim=1)[rank]
        # forward: scatter; backward: allgather
        input = comm.scatter_in(input, dim=-1, ranks=devices)
        output = torch._C._nn.linear(input, weight, bias)
        # forward: reduce; backward: identity
        output = comm.reduce_out(output, ranks=devices)

    return output