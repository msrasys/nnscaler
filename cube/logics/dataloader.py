from typing import Tuple
from cube.runtime.syndata import CubeDataLoader


class IRDataLoader:

    def __init__(self, dataloader: CubeDataLoader, dtype_map):
        if not isinstance(dataloader, CubeDataLoader):
            raise TypeError("Expected data loader derived from CubeDataLoader")
        self.dataloader: CubeDataLoader = iter(dataloader)
        self.dtypes = [dtype_map.map(dtype) for dtype in dataloader.dtypes]
        self.shapes = [list(shape) for shape in dataloader.shapes]

    def get_batch_dims(self) -> Tuple[int]:
        return tuple(self.dataloader.batch_dims)

    def get_batch_size(self) -> int:
        return self.dataloader.get_batch_size()

    def set_batch_size(self, bs: int):
        self.dataloader.set_batch_size(bs)
        return

    def __iter__(self):
        return self

    def __next__(self):
        from cube.logics.translator import LogicTranslator
        datas = LogicTranslator.load_data(self)
        return datas
