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

        self._buffer_num = None
        self.datas: torch.Tensor = list()
        self.set_data_buffer()

    def __iter__(self):
        self.pos = 0
        return self

    def set_data_buffer(self, buffer_num = 4):
        self.datas = list()
        self._buffer_num = buffer_num
        for _ in range(self._buffer_num):
            datas = list()
            for shape in self.shapes:
                data = torch.randn(shape).cuda()
                datas.append(data)
            self.datas.append(datas)

    def reset(self, batch_size: int):
        super().reset(batch_size)
        self.set_data_buffer()

    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration
        datas = self.datas[self.pos % self._buffer_num]
        if len(datas) == 1: return datas[0]
        else: return tuple(datas)


class SynTextDataLoader(SynDataLoader):

    def set_data_buffer(self, buffer_num=4, text_num=50257):
        self.datas = list()
        self._buffer_num = buffer_num
        for _ in range(self._buffer_num):
            datas = list()
            for shape in self.shapes:
                data = torch.randint(0, text_num, shape, dtype=torch.long).cuda()
                datas.append(data)
            self.datas.append(datas)
