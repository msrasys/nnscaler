from typing import Any, List, Tuple
import torch


class PipeStage(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._cached = dict()
        self._data = ()
        self._input_shapes = ()
        self._input_dtypes = ()
        self._output_shapes = ()
        self._output_dtypes = ()

        # pipeline information
        self._num_stages = None
        self._is_first_stage = None
        self._is_last_stage = None
        self._stage_grank = None    # global rank
        self._next_grank = None     # global rank
        self._prev_grank = None     # global rank
        self._stage_lrank = None    # local rank
        self._next_lrank = None     # local rank
        self._prev_lrank = None     # local rank

    @property
    def is_first_stage(self) -> bool:
        return self._is_first_stage

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    @property
    def next_stage_global_rank(self) -> int:
        return self._next_grank

    @property
    def prev_stage_global_grank(self) -> int:
        return self._prev_grank

    @property
    def stage_global_rank(self) -> int:
        return self._stage_grank

    @property
    def next_stage_local_rank(self) -> int:
        return self._next_lrank

    @property
    def prev_stage_local_rank(self) -> int:
        return self._prev_lrank

    @property
    def stage_local_rank(self) -> int:
        return self._stage_lrank

    @property
    def num_stages(self):
        return self._num_stages

    def set_pipeline(self, group_global_ranks: Tuple[int]):
        """
        Setup pipeline information given global ranks.
        Note NCCL group should be initialized outside
        """
        if len(group_global_ranks) == 0:
            group_global_ranks = (torch.distributed.get_rank(),)
        self._num_stages = len(group_global_ranks)
        self._stage_grank = torch.distributed.get_rank()
        self._stage_lrank = group_global_ranks.index(self._stage_grank)
        
        self._next_grank = group_global_ranks[(self._stage_lrank+1) % self.num_stages]
        self._prev_grank = group_global_ranks[(self._stage_lrank-1) % self.num_stages]

        self._next_lrank = (self._stage_lrank+1) % self.num_stages
        self._prev_lrank = (self._stage_lrank-1) % self.num_stages

        self._is_first_stage = self._stage_lrank == 0
        self._is_last_stage = self._stage_lrank == self.num_stages - 1

    def get_last(self, region: str = 'default') -> Any:
        return self._cached[region][-1]

    def pop_last(self, region: str = 'default') -> Any:
        return self._cached[region].pop(-1)

    def pop(self, region: str = 'default') -> Any:
        return self._cached[region].pop(0)

    def push_ahead(self, val: Any, region: str = 'default') -> Any:
        self._cached[region] = [val] + self._cached[region]

    def push(self, val: Any, region: str = 'default'):
        if region not in self._cached:
            self._cached[region] = []
        return self._cached[region].append(val)

    def assert_empty_cached(self):
        for key, vals in self._cached.items():
            assert len(vals) == 0, f"key {key} still has {len(vals)} values"

    @property
    def inputs_info(self) -> Tuple[Tuple, Tuple]:
        """
        return input shapes and dtypes 
        """
        return self._input_shapes, self._input_dtypes

    @inputs_info.setter
    def inputs_info(self, shapes_dtypes: Tuple[Tuple, Tuple]):
        self._input_shapes, self._input_dtypes = shapes_dtypes

    @property
    def outputs_info(self) -> Tuple[Tuple, Tuple]:
        """
        return output shapes and dtypes
        """
        return self._output_shapes, self._output_dtypes

    @outputs_info.setter
    def outputs_info(self, shapes_dtypes: Tuple[Tuple, Tuple]):
        self._output_shapes, self._output_dtypes = shapes_dtypes

    @property
    def data(self) -> Tuple:
        return self._data

    @data.setter
    def data(self, datas: Tuple):
        self._data = datas


def layer_division(times: List[int], num_stages: int, start_id: int = 0, limits: List[int] = None):
    """
    Computation balance division
    """
    divisions = []
    budget = sum(times) / num_stages
    nlayers = len(times)
    start, end = 0, 1
    if limits is None:
        limits = [None] * num_stages
    else:
        assert len(limits) == num_stages
    for idx in range(num_stages):
        accum = times[start]
        assert end <= nlayers
        while end != nlayers:
            if limits[idx] is not None and (end - start) == limits[idx]:
                break
            if times[end] > 0 and budget - accum < 0.5 * times[end]:
                break
            accum += times[end]
            end += 1
        if idx == num_stages - 1:
            end = nlayers
        divisions.append((start, end))
        if idx != num_stages - 1:
            budget = sum(times[end:]) / (num_stages - 1 - idx)
        start, end = end, end+1
    for sid in range(num_stages):
        start, end = divisions[sid]
        divisions[sid] = (start+start_id, end+start_id)
    return divisions
