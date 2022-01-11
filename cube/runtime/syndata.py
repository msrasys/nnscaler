r"""
Synthetic Data Loader
"""

from typing import List, Optional, Tuple
import copy
import torch


__all__ = ['CubeDataLoader', 'SynDataLoader']


class CubeDataLoader:
    r"""
    Cube Dataloader
    """
    def __init__(self, shapes: Tuple[List[int]], dtypes: Tuple[torch.dtype], batch_dims: Tuple[int]):
        """
        shapes Tuple[Tuple[int]]:
            The shape for each data
        dtypes Tuple[torch.dtype]:
            The dtype for each data
        batch_dims Tuple[int]:
            The batch dimension of each data
        """
        if not all(isinstance(shape, list) for shape in shapes):
            raise TypeError("Expected each shape in shapes to be a list")
        if len(shapes) != len(batch_dims) or len(shapes) != len(dtypes):
            raise TypeError("Expected number batch dim and dtypes to len(shapes)")
        self.shapes = shapes
        self.dtypes = dtypes
        self.batch_dims = batch_dims

    def get_batch_dims(self, idx: Optional[int] = None) -> int:
        """
        Get batch dimension for idx-th data
        """
        if idx is not None:
            return self.batch_dims[idx]
        else:
            return list(self.batch_dims)

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
    for given shapes, dtypes.
    """
    def __init__(self, shapes: Tuple[List[int]], dtypes: Tuple[torch.dtype] = None,
                 batch_dims: Tuple[int] = None, length: int = 1280):
        """
        shapes Tuple[Tuple[int]]:
            The shape for each data
        dtypes Tuple[torch.dtype]:
            The dtype for each data (Default None: use torch.float32)
        batch_dims Tuple[int]:
            The batch dimension of each data (Default None: dimension 0 is the batch dim)
        length int:
            Total number of sample batches. (Default 1280)
        """
        if batch_dims is None:
            batch_dims = tuple([0] * len(shapes))
        if dtypes is None:
            dtypes = tuple([torch.float] * len(shapes))

        super().__init__(shapes, dtypes, batch_dims)
        self.length = length
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
            for shape, dtype in zip(self.shapes, self.dtypes):
                data = torch.randn(shape, dtype=dtype).cuda()
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
