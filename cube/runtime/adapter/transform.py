"""
Adapter: Tensor Transformation
"""

from typing import List, Tuple
import torch


def identity(tensor: torch.Tensor):
    """
    identity 
    """
    with torch.no_grad():
        tensor = tensor.detach().requires_grad_()
    return tensor


def select(tensor: torch.Tensor,
           indmap: Tuple[slice], valmap: Tuple[int, int]) -> torch.Tensor:
    """
    Select a part of tensor spatially and numerically.
    """
    with torch.no_grad():
        sub_tensor = tensor[indmap]
        if valmap != (0, 1):
            sub_tensor = sub_tensor / valmap[1]
        sub_tensor = sub_tensor.detach().requires_grad_()
    return sub_tensor


def chunk(itensor: torch.Tensor, dim: int, chunks: int, idx: int) -> torch.Tensor:
    """
    split dimension in n chunks and take idx-th chunk
    """
    with torch.no_grad():
        otensor = itensor.chunk(chunks, dim)[idx]
        otensor = otensor.detach().requires_grad_()
    return otensor


def smerge(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Runtime primitive of spatial merge.
    Concatenate the tensors along a dimension

    Args:
        tensors: a list of torch tensor
        dim: the dimension to concatenate.
    """
    with torch.no_grad():
        out = torch.concat(tuple(tensors), dim).requires_grad_()
    return out


def vmerge(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Runtime primitives of numerical merge.
    Sum the tensors.

    Args:
        tensors: a list of torch tensor
    """
    with torch.no_grad():
        out = tensors[0]
        for tensor in tensors[1:]:
            out = out + tensor
    return out.requires_grad_()
