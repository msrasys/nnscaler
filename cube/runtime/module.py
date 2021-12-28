from typing import List
import torch
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer


class CubeModule(torch.nn.Module):
    """
    The module is responsible for parameter synchronization
    before training
    """

    def __init__(self):
        super().__init__()
        self._reducers = list()

    def add_reducer(self, reducer: Reducer):
        if not isinstance(reducer, Reducer):
            raise RuntimeError(f"Expected a Reducer but got {type(reducer)}")
        self._reducers.append(reducer)

    def sync_params(self):
        for reducer in self._reducers:
            reducer.sync()

    def init_param(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param)

    def init_group(self, ranks: List[int]):
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected ranks to be List[int]")
        DeviceGroup().get_group(ranks)
