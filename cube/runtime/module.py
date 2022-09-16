from typing import List, Dict, Tuple, Optional
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
        self._fullmap : Dict[str, Tuple[int, Tuple[slice], int]] = dict()
        self._batch_size: Optional[int] = None

    def add_reducer(self, reducer: Reducer):
        if not isinstance(reducer, Reducer):
            raise RuntimeError(f"Expected a Reducer but got {type(reducer)}")
        self._reducers.append(reducer)

    def sync_params(self):
        for reducer in self._reducers:
            reducer.sync()

    def add_full_map(self, attr: str, tid: int, slicers: Tuple[slice], val_chunks: int):
        """
        Add an attribute map.
        The mapping includes current attribute name (str) to logical tensor id,
        and the mapping of logical tensor id including spatial (slice) and val chunks
        
        @param attr str: attribute name of this moudle
        @param tid int: full tensor id
        @param slicers Tuple[slice]: indexing from full tensor
        @param val_chunks int: the number of value chunks.
        """
        assert hasattr(self, attr), f"{attr} is not in the module"
        self._fullmap[attr] = (tid, slicers, val_chunks)

    def set_batch_size(self, bs: Optional[int]):
        assert (bs is None) or (isinstance(bs, int) and bs > 0)
        self._batch_size = bs

    def get_batch_size(self) -> Optional[int]:
        return self._batch_size

    def load_attr_content(self, filename: str):
        with torch.no_grad():
            full = torch.load(filename)
            for attr in self._fullmap.keys():
                tensor: torch.Tensor = getattr(self, attr)
                tid, slicers, nchunks = self._fullmap[attr]
                content = full[tid][slicers] / nchunks
                tensor.copy_(content)
                # print(f'attr {attr}:\n{getattr(self, attr)}')

    def init_group(self, ranks: List[int]):
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected ranks to be List[int]")
        DeviceGroup().get_group(ranks)
