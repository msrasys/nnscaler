r"""
Synthetic Data Loader
"""

from typing import List, Optional
import copy
import torch


__all__ = ['CubeDataLoader', 'SynDataLoader']


class CubeDataLoader:
    r"""
    Cube Dataloader
    """
    def __init__(self, batch_dims: List[int], *shapes: List[List[int]]):
        """
        batch_dim:
            The batch dimension for each input shapes
        *shapes:
            The shape for each data
        """
        if not isinstance(batch_dims, list):
            raise RuntimeError("Expected a List[int] for batch dims")
        self.shapes = list(shapes)
        self.batch_dims = batch_dims

    def get_batch_dims(self, idx: Optional[int] = None) -> int:
        """
        Get batch dimension for idx-th data
        """
        if idx is not None:
            return self.batch_dims[idx]
        else:
            return copy.copy(self.batch_dims)

    def reset(self, batch_size: int):
        """
        Reset batch size
        """
        for bdim, shape in zip(self.batch_dims, self.shapes):
            shape[bdim] = batch_size
        print(f'> data loader output shape change to: {self.shapes}')


class SynDataLoader(CubeDataLoader):
    r"""
    Synthetic dataloader to produce tensors
    for given shape.
    """
    def __init__(self, num: int, batch_dim: List[int], *shapes: List[List[int]]):
        if len(shapes) != len(batch_dim):
            raise TypeError("Expected length of batch dim is same to shapes")
        super().__init__(batch_dim, *shapes)
        self.length = num
        self.pos = 0

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration
        datas = list()
        for shape in self.shapes:
            data = torch.randn(shape).cuda()
            datas.append(data)
        if len(datas) == 1: return datas[0]
        else: return tuple(datas)
