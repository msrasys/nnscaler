import copy
from typing import Optional

import torch

from cube.runtime.syndata import CubeDataLoader


class IRDataLoader:

    def __init__(self, dataloader: CubeDataLoader, dtype_map):
        self.dataloader = iter(dataloader)
        self.batch_dims = dataloader.get_batch_dims()
        self.dtypes = list()
        self.shapes = list()

        datas = next(dataloader)
        if not isinstance(datas, tuple):
            datas = (datas,)
        
        for data in datas:
            if torch.is_tensor(data):
                self.dtypes.append(dtype_map.map(data.dtype))
                shape = tuple(data.shape)
                # special handler for scalar tensor shape
                if len(shape) == 0:
                    shape = (1,)
                self.shapes.append(shape)
            else:
                raise NotImplementedError("Data should be torch.Tensor")

    def get_batch_dims(self, idx: Optional[int] = None) -> int:
        if idx is None:
            return copy.copy(self.batch_dims)
        else:
            return self.batch_dims[idx]

    def __iter__(self):
        return self

    def __next__(self):
        from cube.logics.translator import LogicTranslator
        datas = LogicTranslator.load_data(self)
        return datas
