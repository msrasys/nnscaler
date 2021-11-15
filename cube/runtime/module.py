import torch
from cube.runtime.reducer import Reducer


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
