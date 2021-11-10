r"""
Synthetic Data Loader
"""

from typing import List
import torch


__all__ = ['SynDataLoader']


class SynDataLoader:
    r"""
    Synthetic dataloader to produce tensors
    for given shape.
    """
    def __init__(self, num: int, *shapes: List[List[int]]):
        self.shapes = list(shapes)
        self.length = num
        self.pos = 0

    def __iter__(self):
        self.pos = 0
        return self

    def reset(self, batch_size: int):
        """
        Reset batch size
        """
        for shape in self.shapes:
            shape[0] = batch_size

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
