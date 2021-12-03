from typing import List

import torch

from cube.runtime.device import DeviceGroup


class ParallelModule(torch.nn.Module):

    def __init__(self, pp_ranks: List[int] = list(),
                       dp_ranks: List[int] = list(),
                       tp_ranks: List[int] = list()):

        super().__init__()
        self._pp_ranks = tuple(pp_ranks)
        self._pp_group = DeviceGroup().get_group(pp_ranks)

        self._dp_ranks = tuple(dp_ranks)
        self._dp_group = DeviceGroup().get_group(dp_ranks)

        self._tp_ranks = tuple(tp_ranks)
        self._tp_group = DeviceGroup().get_group(tp_ranks)

        self.in_size = None
        self.out_size = None

    @property
    def pp_ranks(self):
        return self._pp_ranks

    @property
    def pp_group(self):
        return self._pp_group

    def use_pp(self):
        return len(self._pp_ranks) > 1

    @property
    def dp_ranks(self):
        return self._dp_ranks

    @property
    def dp_group(self):
        return self._dp_group

    def use_dp(self):
        return len(self._dp_ranks) > 1

    @property
    def tp_ranks(self):
        return self._tp_ranks

    @property
    def tp_group(self):
        return self._tp_group

    @property
    def use_tp(self):
        return len(self._tp_ranks) > 1
    
    def set_in_size(self, size: List[int]):
        self.in_size = size

    def set_out_size(self, size: List[int]):
        self.out_size = size