"""
Adapter: Tensor Transformation
"""

from typing import List, Tuple, Optional
import torch


def select(tensor: torch.Tensor,
           indices: Tuple[slice], val_map: Tuple[int, int]) -> torch.Tensor:

    with torch.no_grad():
        sub_tensor = tensor[indices]
        if val_map != (0, 1):
            sub_tensor = sub_tensor / val_map[1]
        sub_tensor = sub_tensor.contiguous()
    return sub_tensor

def merge(tensors: List[torch.Tensor],
          concat: Optional[int] = None,
          add: bool = False):
    """
    Runtime primitive to finish tensor transformation.

    Warning: No contiguous is called!!! need to explicitly called
    before communication

    Args:
        tensors: a list of torch tensor
        concat: Optional[int]: the dimension to merge
        add: bool: whether to perform value merge
    """
    if not ((concat is not None) ^ (add is True)):  # xor condition
        raise RuntimeError("Expected concat or add")
    if concat is not None:
        with torch.no_grad():
            out = torch.cat(tensors, concat)
        return out
    if add is not None:
        with torch.no_grad():
            out = tensors[0]
            for tensor in tensors[1:]:
                out = out + tensor
        return out
