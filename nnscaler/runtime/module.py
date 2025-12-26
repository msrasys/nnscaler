#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import functools
import pickle
from typing import Callable, List, Set, Dict, Tuple, Optional, TYPE_CHECKING, Any, Union, ClassVar
from typing_extensions import Self
import logging
import os
import sys
import gc
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.distributed as dist
from torch import device
from torch.autograd.graph import saved_tensors_hooks

from nnscaler.graph.parser import FxModuleParser

from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.adapter.reducer import Reducer
from nnscaler.runtime.executor import Executor
from nnscaler.runtime.gnorm import ParamsInfo
from nnscaler.runtime.utils import microbatches
from nnscaler.runtime.function import insert_backward_hook

from nnscaler import __version__ as runtime_version
from nnscaler.flags import CompileFlag
from nnscaler.utils import accum_mode, classproperty, unchecked_fields

if TYPE_CHECKING:
    from nnscaler.parallel import ComputeConfig


_logger = logging.getLogger(__name__)


@dataclass
class AttrMeta:
    # full tensor ID
    tid: int
    # is this a parameter
    is_param: bool
    # original name in the module
    orig_name: str
    # shape of the full tensor
    shape: Tuple[int, ...]
    # list of slicers to index the full tensor
    slicers: Tuple[slice, ...]
    # the number of the partitioned values, usually 1
    # (i.e., no partition on value -> no need to sum up)
    val_chunks: int
    # data type of the full tensor and sub tensor
    dtype: torch.dtype
    # shape of the sub tensor
    # it should be the shape of full_tensor[slicers]
    sub_shape: Tuple[int, ...]


@dataclass
class Zero3AttrMeta:
    """
    Used for loading merged state dict
    """
    # original name in the module
    orig_name: str
    # name in the module
    attr_name: str
    # start index of the sub tensor
    start: int
    # end index of the sub tensor
    end: int
    # chunk size of the sub tensor, can be bigger than end - start due to padding
    chunk_size: int


def dedup_attrs(rank2attr_area_map: Dict[int, Dict[str, Dict[str, AttrMeta]]]) -> Dict[int, Dict[str, Dict[str, AttrMeta]]]:
    '''
    Deduplicate the attributes according to `rank2attr_area_map`.
    For each `slicers` of a full tensor identified by its full qualified name, we only store its first appearance
    in the `rank2attr_area_map`. In nnscaler, this dedup process leads to:
    - If an attribute is not within the first scale unit, it will be deduplicated.
    - If an attribute is shared by different operators, it will be deduplicated.
    - If an attribute is replicated across several devices, we only save it at the devices with the smallest rank.
    - If an attribute is partitioned across several devices, all these sub tensors will be saved.
    - Note that nnscaler supports partition an operator across multiple dimensions, attributes in the operator may
      be saved at a subset of related devices.
    - Pipeline parallelism is supported since it is composed of different segments in nnscaler, which are different
      parallel modules with their own attribute maps at runtime.
    In addition, we will check
    - the shape of the full tensor is consistent across different ranks
    - the slicers of the full tensor are not intersected with each other
    - the slicers of the full tensor can cover the full tensor

    Args:
        rank2attr_area_map (
            Dict[int, # rank id
                Dict[str, # submodule prefix
                    Dict[str, # attribute name in parallel module (not original name)
                        AttrMeta]]]): fullmap information for all parallel modules in all ranks.

    Returns:
        Dict[int, Dict[str, Dict[str, AttrMeta]]]: the deduplicated fullmap info, the structure is the same as the input.
    '''
    # assume ranks in rank2attr_area_map are in increasing order
    ranks = list(rank2attr_area_map.keys())
    for i in range(1, len(ranks)):
        assert ranks[i - 1] < ranks[i], f'rank {ranks[i - 1]} should be less than rank {ranks[i]}'

    orig_name2slice_info = defaultdict(list)
    orig_name2shape = dict()

    def need_save(slicers: Tuple[slice, ...], saved_slicers_list: List[Tuple[slice, ...]]) -> bool:
        for saved_slicers in saved_slicers_list:
            assert len(slicers) == len(saved_slicers), f'If two slicers are related to one same full tensor, lengths should be equal, but get {slicers} vs {saved_slicers}'
            if slicers == saved_slicers:
                return False
            # if slicers intersect with saved_slicers, raise error
            for s, ss in zip(slicers, saved_slicers):
                if s == ss:
                    continue
                if s.start < ss.stop and s.stop > ss.start:
                    raise RuntimeError(f'intersected slicers {slicers} vs {saved_slicers}')
        return True

    ret = dict()
    for rank, module_fullmaps in rank2attr_area_map.items():
        dedup_module_fullmaps = dict()
        for module_name, attr_area_map in module_fullmaps.items():
            dedup_attr_area_map = dict()
            for attr, attr_meta in attr_area_map.items():
                assert attr_meta.val_chunks == 1, 'not support partitioning on value dimension'
                # use module_name.orig_name as the unique identifier for full tensor
                full_tensor_name = f"{module_name}.{attr_meta.orig_name}"
                if full_tensor_name not in orig_name2shape:
                    orig_name2shape[full_tensor_name] = attr_meta.shape
                else:
                    assert orig_name2shape[full_tensor_name] == attr_meta.shape, \
                        f'unmatched shape {orig_name2shape[full_tensor_name]} vs {attr_meta.shape}'
                if need_save(attr_meta.slicers, orig_name2slice_info[full_tensor_name]):
                    orig_name2slice_info[full_tensor_name].append(attr_meta.slicers)
                    dedup_attr_area_map[attr] = attr_meta
            if dedup_attr_area_map:  # only add non-empty maps
                dedup_module_fullmaps[module_name] = dedup_attr_area_map
        ret[rank] = dedup_module_fullmaps

    # since we
    # - skip saving when there are identical weights
    # - assert the slicers are disjoint
    # we can use the sum of the sub-slicers to verify the full tensor is covered
    for full_tensor_name, slicerss in orig_name2slice_info.items():
        shape = orig_name2shape[full_tensor_name]
        full_size = 1
        for s in shape:
            full_size *= s
        covered_size = 0
        for slicers in slicerss:
            size = 1
            for s in slicers:
                size *= s.stop - s.start
            covered_size += size
        assert full_size == covered_size, f'uncovered size for {full_tensor_name} with shape {shape}, slicerss {slicerss}'

    return ret


class CubeModule(torch.nn.Module):
    """
    The module is responsible for parameter synchronization
    before training
    """

    # whether the train_step/infer_step is using a scheduler,
    # will be assigned in the generated subclasses
    use_scheduler: bool
    # the number of microbatches in one scheduler train/infer step
    # 1 if no scheduler is used.
    # will be assigned in the generated subclasses
    nmicros_per_scheduler_step: int

    def __init__(self):
        super().__init__()
        self._reducers: List[Reducer] = list()
        # self._fullmap is mapping from the name of local attribute tensor
        # to its corresponding fulltensor meta
        # please note there can be multiple entries with same tid
        self._fullmap : Dict[str, AttrMeta] = dict()

    def get_non_persistent_buffers(self) -> Dict[str, torch.Tensor]:
        """
        Get non-persistent buffers in the module
        """
        non_persistent_buffers = {}
        for name, buffer in self.named_buffers(recurse=False):
            if name in self._non_persistent_buffers_set:
                non_persistent_buffers[name] = buffer
        return non_persistent_buffers

    @property
    def reducers(self):
        return self._reducers

    @property
    def fullmap(self) -> Dict[str, AttrMeta]:
        """
        Get the mapping from the name of local attribute tensor
        to its corresponding fulltensor meta
        """
        return self._fullmap

    def tid_of_param_name(self, name: str) -> int:
        # Return the tid of sub tensor with the parameter name
        # It is the last field of the parameter name, which is hacky
        if name not in self._fullmap:
            raise RuntimeError(f"Cannot find {name} in fullmap")
        return int(name.split('_')[-1])

    def add_reducer(self, reducer: Reducer):
        if not isinstance(reducer, Reducer):
            raise RuntimeError(f"Expected a Reducer but got {type(reducer)}")
        self._reducers.append(reducer)

    def zero_grad(self):
        """Make zero for gradients in weight reducer

        This only applies on the gradients of the parameters in each reducer.
        This function will be automatically inserted inside the generated code
        at the beginning of each iteration.

        If the function is under the context of `with nnscaler.accum_mode()`, the zero of gradients
        will be skipped.
        """
        for reducer in self._reducers:
            reducer.zero_grad()

    def parameters_for_optimizer(self) -> List[torch.nn.Parameter]:
        """Get parameter list for optimizer"""
        return list(self.get_opt_params().keys())

    def get_opt_params(self, prefix='', classify_param_cls_fn: Callable[[str], Any]=None) -> dict[torch.nn.Parameter, Any]:
        """
        Get all parameters and their classifications. Parameters in reducers come first.

        Args:
            prefix (str): The prefix of this module,
                which will be used to generate full names of parameters and further classify them.
            classify_param_cls_fn (Callable[[str], Any], optional): A function to classify parameters by name.

        Returns:
            dict[torch.nn.Parameter, Any]: A dictionary mapping parameters to their classifications.
        """
        params = {}
        reducer_pids = set()
        for reducer in self._reducers:
            params.update(reducer.get_opt_params())
            reducer_pids.update(id(p) for p in reducer.params)
        for name, param in self.named_parameters(prefix):
            if id(param) not in reducer_pids:
                params[param] = classify_param_cls_fn(name) if classify_param_cls_fn else None
        # print(f'> get out parameters: {sum(p.numel() for p in params)}')
        return params

    def parameters_for_broadcast(self) -> List[torch.nn.Parameter]:
        """
        This function is for broadcasting loaded weights from one scale unit to
        all other scale units to resume from sharded checkpoints globally.
        """
        params = []
        reducer_pids = set()
        for reducer in self._reducers:
            params.append(reducer._contiguous_params)
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                params.append(param)
        return params

    def parameters_for_calc_gnorm(self) -> List[ParamsInfo]:
        """Return the necessary information for calculating the gradient norm.

        Returns:
            List[Tuple[Tuple[int], List[torch.nn.Parameter], List[str], int]]:
                A list of tuples, each tuple contains the following information:
                    Tuple[int]: the ranks spanned by the parameters in the tuple
                    List[torch.nn.Parameter]: the contiguous parameters in the tuple
                    List[str]: the names of the original parameters in the tuple
                    int: the number of the ZeRO groups for the parameters
        """
        paramid2name = {}
        for name, param in self.named_parameters():
            paramid2name[id(param)] = name

        params_info_for_gnorm = []
        reducer_pids = set()
        for reducer in self._reducers:
            param_names = [paramid2name[id(p)] for p in reducer.params]
            # we should use `parameters_for_optimizer` here since calculating gnorm
            # is ahead of the optimizer step. When ZeRO is enabled, each device only
            # maintains a subset of the parameters. As a result, `param_names` may not
            # align with the value of `reducer.parameters_for_optimizer()`, only part of
            # parameters assigned to a bucket will be shown in `reducer.parameters_for_optimizer()`.
            params_info = ParamsInfo(reducer.ranks, reducer.parameters_for_optimizer(),
                                     param_names, reducer.zero_ngroups)
            params_info_for_gnorm.append(params_info)
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                # zero_ngroups is 1, since there is no reducer for it and multiplying 1 does not change the result.
                params_info = ParamsInfo((dist.get_rank(),), [param], [paramid2name[id(param)]], 1)
                params_info_for_gnorm.append(params_info)
        return params_info_for_gnorm

    def gather_params(self):
        """
        Gather parameters

        This won't take effect when zero is not enabled.
        """
        for reducer in self._reducers:
            if reducer.zero:
                reducer.gather_params()

    def add_full_map(self, attr: str, tid: int, is_param: bool, orig_name: str, shape: Tuple[int],
                     slicers: Tuple[slice], val_chunks: int):
        """Add an attribute map.

        Args:
            attr (str): attribute name of this module
            tid (int): full tensor id
            is_param (bool): whether this attribute is a parameter, otherwise it is a buffer
            orig_name (str): attribute name in the original module
            shape (Tuple[int]): shape of the full tensor
            slicers (Tuple[slicer]): indexing from full tensor
            val_chunks int: the number of value chunks.
        """
        assert hasattr(self, attr), f"{attr} is not in the module"
        attr_tensor: torch.Tensor = getattr(self, attr)
        meta = AttrMeta(tid, is_param, orig_name, shape, slicers, val_chunks, attr_tensor.dtype, tuple(attr_tensor.shape))
        self._fullmap[attr] = meta

    # TODO: remove this function, use the property instead
    def get_full_map(self):
        return self._fullmap

    def load_attr_content(self, filename: str):
        """Load module attribute (parameters and buffers) from file

        Args:
            filename (str): base file name (without '.0', '.1', etc.)
                that saved with model parameters
        """
        npartitions = 0
        while os.path.isfile(filename + f'.{npartitions}'):
            npartitions += 1
        if npartitions == 0:
            raise RuntimeError(f"Cannot find file {filename}.0 in load_attr_content")
        with torch.no_grad():
            _logger.info(f'loading partitioned model from {filename}, number of model parameter chunks: {npartitions}')
            attr_names = set(self._fullmap.keys())
            for file_idx in range(npartitions):
                # part_model contains a subset of attributes, where each attribute is a fulltensor
                # fulltensor.tid -> torch.Tensor
                part_model: Dict[int, torch.Tensor] = torch.load(filename + f'.{file_idx}')
                loaded_name = set()
                for attr_name in attr_names:
                    meta = self._fullmap[attr_name]
                    if meta.tid not in part_model:
                        continue
                    attr = getattr(self, attr_name)
                    content = part_model[meta.tid][meta.slicers]
                    if meta.val_chunks != 1:
                        content = content / meta.val_chunks
                    attr.copy_(content)
                    loaded_name.add(attr_name)
                for name in loaded_name:
                    attr_names.remove(name)
            if len(attr_names) != 0:
                raise RuntimeError(
                    f'remaining graph parameters / buffers cannot find in model files: {list(attr_names)}')

    def init_group(self, ranks: List[int]):
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected ranks to be List[int]")
        DeviceGroup().get_group(ranks)

    def get_checkpoint(self, optimizer: torch.optim.Optimizer = None):
        state_dict = super().state_dict()
        # backward compatibility
        # in old version, dist_param_map is not loaded in constructor
        # so we will try to load it from file on the fly.
        dist_param_map = getattr(self, 'dist_param_map', None)
        if not dist_param_map:
            module_file = Path(sys.modules[self.__module__].__file__)
            # load from the same directory as the module file
            dist_param_map = torch.load(module_file.with_name(FxModuleParser.ATTR_MAP_FILE), weights_only=False)
        param_area_map = self._fullmap
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        return state_dict, dist_param_map, param_area_map, optimizer_state_dict

    def save_checkpoint(self, optimizer: torch.optim.Optimizer = None, filename_prefix: str = None):
        filename_prefix = 'dist_checkpoint' if filename_prefix is None else filename_prefix
        filename = f"{filename_prefix}-{DeviceGroup().rank}.ckpt"
        state_dict, dist_param_map, param_area_map, optimizer_state_dict = self.get_checkpoint(optimizer)

        _logger.info(f'saving distributed checkpoint to {filename}')
        torch.save({
            'state_dict': state_dict,
            'dist_param_map': dist_param_map,
            'param_area_map': param_area_map,
            'optim_state_dict': optimizer_state_dict,
        }, filename)

    @classmethod
    def _safe_tensor_equal(cls, tensor1: Any, tensor2: Any):
        # in different versions, the data may be different types
        # for example, step in optimizer.state_dict can be scalar tensor or int.
        if type(tensor1) != type(tensor2):
            return False
        if not isinstance(tensor1, torch.Tensor):
            return tensor1 == tensor2
        if tensor1.shape != tensor2.shape:
            return False
        if tensor1.dtype != tensor2.dtype:
            return False
        if tensor1.device == tensor2.device:
            return torch.equal(tensor1, tensor2)
        else:
            return torch.equal(tensor1.cpu(), tensor2.cpu())

    @staticmethod
    def merge_model_state_dicts(
        state_dicts: List[Dict],
        fullmaps: List[Dict[str, AttrMeta]]
    ):
        """Merge model states from multiple shard into a single-model state.
        Here we assume the order of state_dicts and fullmaps are aligned, and is the same as the rank order.

        Note:
            Users only need to provide as fewer local model states as necessary to
            cover the full model state.

        Args:
            state_dicts (List[Dict[str, torch.Tensor]]): per-rank local model state dict from model.state_dict()
            fullmaps (List[Dict[str, AttrMeta]]): per-rank fullmap

        Returns:
            full_state_dicts (List[Dict[str, torch.Tensor]]): Full model state dict
        """
        if len(state_dicts) != len(fullmaps):
            raise ValueError("Expected model state dicts to have the same length as fullmaps")

        full_model_state_dict: Dict[str, torch.Tensor] = {}
        # used to track the merging status of each parameter to avoid inconsistence.
        # key is the parameter name, value is a set of slicers
        # Here we expand slice to (start, step, stop) tuple,
        # because before python 3.12, slice object is not hashable
        state_dict_merge_track: Dict[str, Set[Tuple[Tuple[Any, Any, Any], ...]]] = {}
        # the fill progress of zero3 parameters
        #  key: param name
        #  value: Dict[ tuple(start, step, stop) , filled size]
        #  used to track how many elements have been filled for each zero3 parameter
        zero3_current_filled: Dict[str, Dict[Tuple[Tuple[int, int, int], ...], int]] = {}
        # gather param/buffer full tensor
        for rank, (model_state_dict, local_fullmap) in enumerate(zip(state_dicts, fullmaps)):
            for local_name, meta in local_fullmap.items():
                if local_name not in model_state_dict:
                    # the parameter may not in model_state_dict (deduped with optimization)
                    # Another case is when this is a non persistent buffer, we should skip it,
                    # since non persistent buffer should be stored in the fullmap, but not in the model state dict
                    continue
                # create full tensor on cpu
                partial_tensor = model_state_dict[local_name]
                if meta.orig_name not in full_model_state_dict:
                    full_model_state_dict[meta.orig_name] = torch.empty(
                        meta.shape, dtype=partial_tensor.dtype)
                    state_dict_merge_track[meta.orig_name] = set()
                # assign partial tensor
                if meta.val_chunks > 1:
                    raise NotImplementedError("Not support of partitioning parameter / buffer at value dimension")

                state_dict_merge_track_id = tuple((i.start, i.step, i.stop) for i in meta.slicers)
                dest_tensor = full_model_state_dict[meta.orig_name][meta.slicers]
                if dest_tensor.shape == partial_tensor.shape and state_dict_merge_track_id in state_dict_merge_track[meta.orig_name]:
                    if not CubeModule._safe_tensor_equal(dest_tensor, partial_tensor):
                            raise ValueError(f"Conflict in merging {meta.orig_name} from rank {rank}")
                    _logger.debug(f'rank {rank}: skip merging duplicated model state for param {meta.orig_name} with slicers {meta.slicers}')
                else:
                    state_dict_merge_track[meta.orig_name].add(state_dict_merge_track_id)
                    if dest_tensor.shape == partial_tensor.shape:
                        dest_tensor.copy_(partial_tensor)
                    else:
                        # we assume zero3 is on when dest_tensor.shape != partial_tensor.shape
                        if len(partial_tensor.shape) != 1:
                            raise ValueError("Invalid tensor as a ZeRO3 parameter, expected a 1D tensor.")
                        fill_start = zero3_current_filled.setdefault(meta.orig_name, {}).setdefault(state_dict_merge_track_id, 0)
                        fill_len = partial_tensor.numel()
                        if fill_start >= dest_tensor.numel():
                            # already filled, let's check consistency
                            fill_start = fill_start % dest_tensor.numel()
                            if fill_start + fill_len > dest_tensor.numel():
                                # remove padding part
                                fill_len = dest_tensor.numel() - fill_start
                            if not CubeModule._safe_tensor_equal(dest_tensor.view(-1)[fill_start: fill_start + fill_len], partial_tensor[0:fill_len]):
                                raise ValueError(f"Conflict in merging {meta.orig_name} from rank {rank}")
                        else:
                            if fill_start + fill_len > dest_tensor.numel():
                                # remove padding part
                                fill_len = dest_tensor.numel() - fill_start
                            old_shape = dest_tensor.shape
                            dest_tensor = dest_tensor.reshape(-1)
                            dest_tensor[fill_start: fill_start + fill_len] = partial_tensor[0: fill_len]
                            full_model_state_dict[meta.orig_name][meta.slicers] = dest_tensor.view(old_shape)

                        zero3_current_filled[meta.orig_name][state_dict_merge_track_id] += fill_len

        return full_model_state_dict

    @staticmethod
    def get_origin_parameter_names(fullmaps: List[Dict[str, AttrMeta]]):
        """
        Get a list of original parameter names from the fullmaps.
        `merge_partial_states` will use this list to build the parameter order
        """
        origin_parameter_names: List[str] = []
        for local_fullmap in fullmaps:
            for _, meta in local_fullmap.items():
                if not meta.is_param: continue
                # shared parameters in CubeModule is already de-duplicated. So in the
                # local model state, we will not have multiple parameters sharing with same content
                # but in different names.
                if meta.orig_name not in origin_parameter_names:
                    origin_parameter_names.append(meta.orig_name)
        return origin_parameter_names

    @staticmethod
    def merge_partial_states(state_dicts: List,
                             zero_idx_maps=None):
        """Merge model and optimizer states from different shard into a single-model state.

        Warnings:
            * This function only supports merging optimizer states of Adam-like optimizers,
            in which the optimizer state is expected to contain 'state' keyword.
            * Only support single parameter group, i.e., code implementations like: `torch.optim.Adam(model.parameters(), lr=0.1)`

        Args:
            state_dicts (List[(Dict, Dict, Dict, Dict)]): per-rank states containing:
                * model_state_dicts (List[Dict[str, torch.Tensor]]): per-rank model state dict from model.state_dict()
                * optim_state_dicts (Optional[List[Dict]]): per-rank optimizer state dict from optimizer.state_dict()
                * dist_param_map: deprecated, will be removed in the future.
                * fullmaps (List[Dict[str, AttrMeta]]): per-rank fullmap
            zero_idx_maps (Optional[List[Dict]]): zero information for the model, `None` if zero is not enabled

        Returns:
            Dict[str, torch.Tensor]: Full model state dict
            Dict[str, Any]: Full optimizer state dict
        """
        # the filtering below is to be compatible with fairseq
        # which will set some model_state_dicts/optim_state_dicts to None for deduplication
        return CubeModule.merge_state_dicts(
            [state_dict[-1] for state_dict in state_dicts],
            [state_dict[0] for state_dict in state_dicts if state_dict[0] is not None],
            [state_dict[1] for state_dict in state_dicts if state_dict[1] is not None],
            zero_idx_maps
        )

    @staticmethod
    def merge_state_dicts(
        fullmaps: List[Dict[str, AttrMeta]],
        model_state_dicts: List[Dict[str, torch.Tensor]],
        optim_state_dicts: Optional[List[Dict[str, Any]]] = None,
        zero_idx_maps: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """Merge model and optimizer states from different shard into a single-model state.

        `fullmaps` should always have the information for all ranks.
        To support checkpoint deduplication, `model_state_dicts` and `optim_state_dicts`
        can contains only the first `dedup_group_size` items.

        Warnings:
            * This function only supports merging optimizer states of Adam-like optimizers,
            in which the optimizer state is expected to contain 'state' keyword.
            * Only support single parameter group, i.e., code implementations like: `torch.optim.Adam(model.parameters(), lr=0.1)`

        Args:
            fullmaps (List[Dict[str, AttrMeta]]): per-rank fullmap
            model_state_dicts (List[Dict[str, torch.Tensor]]): per-rank model state dict from model.state_dict()
            optim_state_dicts (Optional[List[Dict]]): per-rank optimizer state dict from optimizer.state_dict()
            zero_idx_maps (Optional[List[Dict]]): zero information for the model, `None` if zero is not enabled

        Returns:
            Dict[str, torch.Tensor]: Full model state dict
            Dict[str, Any]: Full optimizer state dict
        """
        # state dicts in the 1st scale unit may be a subset of `model_state_dicts`. Using `plan_ngpus` here to
        # help understand the whole logic. In other words, the real plan_ngpus is <= len(model_state_dicts).
        plan_ngpus = len(model_state_dicts)
        # gather model states
        full_model_state_dict = CubeModule.merge_model_state_dicts(model_state_dicts, fullmaps[0: plan_ngpus])
        _logger.info('finish merge model states')
        if optim_state_dicts is None:
            return full_model_state_dict, None

        # gather optimizer states
        full_optim_state_dict: Dict[str, Any] = {}  # param_id -> Dict[state_name, value]

        # build parameter order to match with the optimizer state order
        # NOTE: the param IDs in optimizer typically follow the same order of
        # local `model.parameters()`. However, `state_dict.keys()` contains
        # both parameters and buffers, we need to remove the buffers from the list.
        # More details refer to the implementation:
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module._save_to_state_dict
        origin_parameter_names: List[str] = CubeModule.get_origin_parameter_names(fullmaps)

        # handle 'state' in optimizer state dict
        # NOTE: each rank may have its local optimizer state working on a sub-set
        # of parameters of the full model. So the param IDs in each local optimizer
        # state is a sub-sequence of global parameter ordering.

        # we follow the order of in origin parameter names to find each (partitioned)
        # parameter in the local model state, and assign the slice to the position.
        full_optim_state_dict['state'] = {}
        full_states = full_optim_state_dict['state']

        def _check_state_size(opt_state_keys, bucket_state):
            """
            Check that all the keys except the scalar step for a
            parameter in optimizer states have the same shaped tensor.

            For example, exp_avg, exp_avg_sq in Adam.
            """
            if len(opt_state_keys) <= 1:
                return True
            return all(bucket_state[key].shape == bucket_state[opt_state_keys[0]].shape
                        for key in opt_state_keys)

        def _retrieve_param_opt_state(bucket_states, pstart, pend, pshape, bucket_size, zero_version):
            assert bucket_size % len(bucket_states) == 0
            opt_state_keys = list(bucket_states[0].keys())
            if 'step' in bucket_states[0]:
                opt_state_keys.remove('step')
            assert _check_state_size(opt_state_keys, bucket_states[0]), f'the keys {opt_state_keys} have different shape'
            # NOTE: only support adam for now
            assert 'exp_avg' in opt_state_keys
            assert 'exp_avg_sq' in opt_state_keys

            opt_states, opt_states_1d = {}, {}
            for key in opt_state_keys:
                opt_states[key] = torch.zeros(pshape, dtype=bucket_states[0][key].dtype,
                                                device=bucket_states[0][key].device, requires_grad=False)
                opt_states_1d[key] = opt_states[key].view(-1)

            if zero_version == 1:
                chunk_size = bucket_size // len(bucket_states)
                start_rank_id, start_offset = pstart // chunk_size, pstart % chunk_size
                end_rank_id, end_offset = pend // chunk_size, pend % chunk_size
                if start_rank_id == end_rank_id:
                    for key in opt_state_keys:
                        opt_states_1d[key][:] = bucket_states[start_rank_id][key][start_offset:end_offset]
                else:
                    offset = chunk_size-start_offset
                    for key in opt_state_keys:
                        opt_states_1d[key][:offset] = bucket_states[start_rank_id][key][start_offset:]
                    for i in range(start_rank_id+1, end_rank_id):
                        for key in opt_state_keys:
                            opt_states_1d[key][offset:offset+chunk_size] = bucket_states[i][key][:]
                        offset += chunk_size
                    if end_offset:  # skip if end_offset == 0, because it is a no-op
                        for key in opt_state_keys:
                            opt_states_1d[key][offset:] = bucket_states[end_rank_id][key][:end_offset]
            else:  # zero_version == 3
                assert zero_version > 1, f'unsupported zero version {zero_version}'
                for key in opt_state_keys:
                    fill_start = 0
                    fill_len = pend - pstart
                    param_numel = opt_states_1d[key].numel()
                    for bstate in bucket_states:
                        if fill_start >= param_numel:
                            # from current implementation, code never goes here
                            # because we have used model_idx2opt_idx to filter out unnecessary ranks
                            # but let's keep the logic here for safety
                            fill_start = fill_start % param_numel
                            if fill_start + fill_len > param_numel:
                                fill_len = param_numel - fill_start
                            # check consistency for the already filled part
                            if not CubeModule._safe_tensor_equal(
                                opt_states_1d[key][fill_start: fill_start + fill_len],
                                bstate[key][pstart: pstart+fill_len]
                            ):
                                raise ValueError(f"Conflict in merging optimizer state for param with shape {pshape}")
                        else:
                            if fill_start + fill_len > param_numel:
                                fill_len = param_numel - fill_start
                            # remove padding part
                            opt_states_1d[key][fill_start: fill_start + fill_len] = bstate[key][pstart: pstart+fill_len]
                        fill_start += fill_len

            if 'step' in bucket_states[0]:
                # make sure all steps are different tensors (with same value)
                opt_states['step'] = bucket_states[0]['step'].clone()
            return opt_states

        def _merge_opt_zero(param_shape, worker_idx, param_idx):
            if len(zero_idx_maps[worker_idx]) == 3:
                model_idx2opt_idx, opt_idx2ranks, zero_version = zero_idx_maps[worker_idx]
            else:  # backward compatibility
                assert len(zero_idx_maps[worker_idx]) == 2
                model_idx2opt_idx, opt_idx2ranks = zero_idx_maps[worker_idx]
                zero_version = 1  # default to ZeRO-1
            opt_idx = model_idx2opt_idx[param_idx]
            if isinstance(opt_idx, int):
                # the param without reducer
                assert opt_idx2ranks[opt_idx] is None
                return optim_state_dicts[worker_idx]['state'][opt_idx]
            else:
                # the param in reducer bucket
                opt_idx, pstart, pend, pshape = opt_idx
                if zero_version == 1:
                    assert param_shape == pshape, f'param shape {param_shape} vs pshape {pshape}'
                ranks, bucket_size = opt_idx2ranks[opt_idx]
                # parameters in reducer come first, so we can directly use opt_idx to index.
                bucket_states = [optim_state_dicts[rank]['state'][opt_idx] for rank in ranks]
                return _retrieve_param_opt_state(
                    bucket_states,
                    pstart,
                    pend,
                    param_shape,
                    bucket_size,
                    zero_version
                )

        # full_index: param IDs in the full optimizer state
        for full_index, param_name in enumerate(origin_parameter_names):
            _logger.info(f'start to handle optimizer state for param {param_name} with full_index {full_index}')
            # zero_done_track is used to avoid re-merging the same parameter
            # in the optimizer state
            # zero_done_track_id: slicers
            # Here we expand slice to (start, step, stop) tuple,
            # because before python 3.12, slice object is not hashable
            zero_done_track: Set[Tuple[Tuple[Any, Any, Any], ...]] = set()
            # used to track the merging status of each parameter to avoid inconsistence.
            # key is slicers
            # please note this is only used for non-zero mode
            # becase re-merging the same parameter slice (via _merge_opt_zero) is avoided in zero mode
            state_merge_track: Set[Tuple[Tuple[Any, Any, Any], ...]] = set()

            # There is this for loop because a parameter may be sharded due to TP,
            # consequently, the parameter's optimizer state is also sharded.
            # This for loop is for merging the sharded parameter's optimizer state
            # into its original full state (i.e., the non-partitioned one).
            for work_idx, (optim_state, fullmap) in enumerate(zip(optim_state_dicts[0 : plan_ngpus], fullmaps[0 : plan_ngpus])):
                if 'state' not in optim_state: continue
                # adam-like optimizers have optim_state['state']={} before any optimizer.step()
                if not optim_state['state']: continue
                # filter out non-param attributes as they don't appear in the optimizer state
                param_fullmap = [meta for meta in fullmap.values() if meta.is_param]
                # local index: param IDs in the local optimizer state, we assume
                # it aligns with the order of local `model.parameters()`
                for local_index, meta in enumerate(param_fullmap):
                    if meta.orig_name != param_name: continue
                    full_states.setdefault(full_index, {})

                    # TODO: support customized param groups, where each parameter has IDs
                    # specified from its own param_group
                    track_id = tuple((i.start, i.step, i.stop) for i in meta.slicers)
                    if zero_idx_maps is None:
                        states: Dict[str, torch.Tensor] = optim_state['state'][local_index]
                    else:
                        if track_id not in zero_done_track:
                            # As ZeRO is applied, the optimizer state of this parameter (a shard)
                            # may not be stored locally in its optimizer state.
                            # _merge_opt_zero is for recovering the optimizer state corresponding to this parameter shard.
                            states: Dict[str, torch.Tensor] = _merge_opt_zero(meta.sub_shape, work_idx, local_index)
                            zero_done_track.add(track_id)
                        else:
                            _logger.debug(f'rank {work_idx}: skip merging duplicated optimizer state for param {full_index} with slicers {meta.slicers}')
                            continue

                    for state_name in states.keys():
                        value = states[state_name]
                        # special handle for step: scalar tensor type
                        if state_name == 'step':
                            if state_name in full_states[full_index]:
                                if not CubeModule._safe_tensor_equal(full_states[full_index][state_name], value):
                                    raise ValueError(f"Conflict in merging {param_name}.{state_name} from rank {work_idx}")
                            else:
                                full_states[full_index][state_name] = value
                            continue

                        # for non-tensor states
                        if not isinstance(value, torch.Tensor):
                            if state_name in full_states[full_index]:
                                if full_states[full_index][state_name] != value:
                                    raise ValueError(f"Conflict in merging {param_name}.{state_name} from rank {work_idx}")
                            else:
                                full_states[full_index][state_name] = value
                                _logger.debug(f'non-tensor state {state_name} is merged for {full_index}')
                        # for tensor states, like 'exp_avg'
                        else:
                            # create optimizer state tensor
                            if state_name not in full_states[full_index]:
                                full_states[full_index][state_name] = torch.empty(meta.shape, dtype=value.dtype)

                            if track_id in state_merge_track:
                                if not CubeModule._safe_tensor_equal(full_states[full_index][state_name][meta.slicers], value):
                                    raise ValueError(f"Conflict in merging {param_name}.{state_name} from rank {work_idx}")
                            else:
                                # assign with partial tensor
                                full_states[full_index][state_name][meta.slicers] = value

                    state_merge_track.add(track_id)

        # handle additional state dict keys
        for optim_state_dict in optim_state_dicts[0 : plan_ngpus]:
            for key in optim_state_dict.keys():
                if key != 'state':
                    if key in full_optim_state_dict:
                        _logger.info(f'overwrite optimizer state key {key}')
                    else:
                        _logger.info(f'inherit optimizer state key {key}')
                    full_optim_state_dict[key] = optim_state_dict[key]

        # reset the param_groups params to the full parameter list
        if 'param_groups' in full_optim_state_dict:  # for backward compatibility
            full_optim_state_dict['param_groups'][0]['params'] = list(range(len(origin_parameter_names)))

        return full_model_state_dict, full_optim_state_dict

    @staticmethod
    def merge_checkpoints(filename_prefix='dist_checkpoint'):
        ckpts = {}
        for rank in range(DeviceGroup().world_size):
            filename = f"{filename_prefix}-{rank}.ckpt"
            ckpts[rank] = torch.load(filename, weights_only=False)
        _logger.info(f'checkpoints = {ckpts}')

        state_dicts = []
        for ckpt in ckpts.values():
            model_state_dict = ckpt['state_dict']
            dist_param_map = ckpt['dist_param_map']
            param_area_map = ckpt['param_area_map']
            optimizer_state_dict = ckpt['optim_state_dict']
            state_dicts.push(model_state_dict, optimizer_state_dict, dist_param_map, param_area_map, )

        merged_model_state_dict, merged_optimizer_state_dict = CubeModule.merge_partial_states(state_dicts)

        # dump to ckpt
        torch.save({'state_dict': merged_model_state_dict,
                    'optim_state_dict': merged_optimizer_state_dict
                    }, filename_prefix + '.full.ckpt')

    def sleep(self):
        """
        Move attributes (buffer and param) to cpu and release contiguous buffer in reducers. Different from
        nn.Module's cpu() method, references to attributes are unchanged.
        """
        for name, param in self.named_parameters():
            assert param.grad is None, f'expect {name} with shape {param.shape} has no grad'

        for reducer in self._reducers:
            reducer.zero_grad()

        # we want attribute references are unchanged, so super().cpu() is not used here
        cpu = torch.device('cpu')
        for buffer in self.buffers():
            buffer.data = buffer.data.to(cpu)

        for param in self.parameters():
            param.data = param.data.to(cpu)

        for reducer in self._reducers:
            reducer.sleep()

        gc.collect()
        torch.cuda.empty_cache()
        return self

    def wake_up(self, device: Optional[Union[int, device]] = None) -> Self:
        """
        Move attributes (buffer and param) back to gpu and reallocate memories in reducers. It is a reverse
        operation of `self.sleep()`.
        """
        gpu = torch.cuda.current_device()
        if device is not None:
            if isinstance(device, int):
                index = device
            elif isinstance(device, torch.device):
                index = device.index
            else:
                raise RuntimeError(f'unexpected device type {type(device)}')
            assert gpu == index, f'nnscaler module does not support cross gpu transport, expect {gpu} but got {index}'

        for name, param in self.named_parameters():
            assert param.grad is None, f'expect {name} with shape {param.shape} has no grad'

        # we want attribute references are unchanged, so super().gpu() is not used here
        for buffer in self.buffers():
            buffer.data = buffer.data.to(gpu)

        for param in self.parameters():
            param.data = param.data.to(gpu)

        for reducer in self._reducers:
            reducer.wake_up()

        gc.collect()
        torch.cuda.empty_cache()
        return self

    def to(self, *args, **kwargs):
        """
        Override nn.Module's to function, currently we only allow transfer data from host and device

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if dtype is not None:
            raise ValueError(f'nnscaler does not support passing dtype {dtype} to to()')
        if convert_to_format is not None:
            raise ValueError(f'nnscaler does not support passing convert_to_format {convert_to_format} to to()')
        if non_blocking is not None:
            warnings.warn(f'nnscaler moves tensors in a blocking approach currently')

        # after _parse_to `device` must in type of torch.device
        if device.type == 'cpu':
            return self.cpu()
        elif device.type == 'cuda':
            return self.cuda(device)
        else:
            raise ValueError(f'unsupported device type {device}')


@dataclass
class OriginModuleMetadata:
    origin_state_dict_names: List[str]       # used for merging module state dict
    origin_param_names: List[str]            # used for merging optimizer state dict
    origin_shared_param_names: List[List[str]]# used for merging module state dict


@dataclass
class ZeroMetadata:
    # a mapping from the index of the parameter in the model
    # to (optimizer_index, the start and end in the bucket, the shape of the parameter)
    model_idx2opt_idx: Optional[Dict] = None
    # a mapping from optimizer_index to the related bucket information (sub_ranks, bucket_size)
    opt_idx2ranks: Optional[Dict] = None
    # the level of zero optimization
    # 0: no zero optimization
    # 1: zero1
    # > 1: zero3
    zero: int = 0


@dataclass
class ParallelModuleConfig:
    rank: int
    compute_config: 'ComputeConfig'
    # the dist_param_map of ParallelModule
    dist_param_map: Dict[str, str]
    # the fullmap of ParallelModule
    param_area_map: Dict[str, AttrMeta]
    # the parameter names of ParallelModule
    cube_param_names: List[str]

    def __post_init__(self):
        if isinstance(self.compute_config, dict):
            from nnscaler.parallel import ComputeConfig
            self.compute_config = ComputeConfig(**self.compute_config)
        self.param_area_map = {
            k: AttrMeta(**v) if isinstance(v, dict) else v
            for k, v in self.param_area_map.items()
        }


@dataclass
class ExtraState(ZeroMetadata, OriginModuleMetadata, ParallelModuleConfig):
    pass


class ParallelModule(CubeModule):
    COMPUTE_CONFIG_FILE = 'compute_config.pt'
    ORIGIN_MODULE_METADATA_FILE = 'origin_module_metadata.pt'
    EXTRA_STATE_KEY = 'CUBE_EXTRA_STATE'
    ATTR_META_FILE_PREFIX = 'attr_meta'
    ATTR_META_FILE_TEMPLATE = ATTR_META_FILE_PREFIX + '{}.pkl'  # 'attr_meta{}.pkl'

    # the rank of the module, will be assigned in the generated subclasses
    rank: int
    # the world size to run this module, will be assigned in the generated subclasses
    world_size: int
    # the runtime version of the module when it is generated, will be assigned in the generated subclasses
    runtime_version: str
    # mapping from the name of local attribute tensor
    # to its corresponding fulltensor meta for all ranks.
    # it is a list of dictionaries mapping from attribute names to their metadata
    # and it is a replacement of `CubeModule.fullmap`
    attr_meta_maps: list[dict[str, AttrMeta]]
    # the directory of the module located
    module_dir: Path
    # The map is a dict mapping from the new parameter name (without tid suffix) in parallel module
    # to the parameter name in original module.
    dist_param_map: dict[str, str]
    compute_config: 'ComputeConfig'
    origin_module_metadata: OriginModuleMetadata

    def __init__(self):
        if self.__class__  == ParallelModule:  # not init via super().__init__()
            raise RuntimeError(f"ParallelModule should not be initialized directly. Please derive it first")

        rv = getattr(self, 'runtime_version', None)
        if rv != runtime_version:
            _logger.warning(
                f"Runtime version mismatch: {rv} vs {runtime_version}. "
            )
        super().__init__()
        # this is used to allow multiple sync_grad() calls
        self._sync_grad_required = False
        # save the param replicas info for calculating gradient norm
        # it is a dict mapping from number_of_replicas to a list of local params.
        # For example, _nreplicas2localparams[2] contains all the parameters that have replicated 2 times.
        # this is a lazy initialization,
        # which will be initialized in the first call of `clip_gnorm`
        self._nreplicas2localparams: Optional[Dict[int, List[torch.nn.Parameter]]] = None
        # track whether all the parames (especially the non-persistent buffers) have been initialized
        self._non_presistent_buffers_inited = False
        # track the params that have been prefetched in backward
        # this is only used for zero3
        # The reason is the eviction of prefetched params in backward
        # relies on the input.requires_grad flag to be True
        # If all the inputs do not require grad,
        # the eviction logic will not be triggered
        # In that case, we will delay the eviction until next backward hook.
        self._backward_prefetched_params: dict[torch.nn.Parameter, int] = {}
        # the params that have been prefetched in forward
        self._forward_prefetched_params: set[torch.nn.Parameter] = set()

    def __init_subclass__(cls, skip_init=False, **kwargs):
        # special case when we just fake a ParallelModule class
        # In this case, you should also use object.__new__ instead of __init__
        if skip_init:
            return

        from nnscaler.parallel import ComputeConfig

        super().__init_subclass__(**kwargs)
        cls.attr_meta_maps = []
        cls.module_dir = Path(sys.modules[cls.__module__].__file__).parent

        for rank in range(cls.world_size):
            attr_map_file = cls.module_dir / cls.ATTR_META_FILE_TEMPLATE.format(rank)
            with open(attr_map_file, 'rb') as f:
                attr_meta_map = pickle.load(f)
                attr_meta_map = {attr: AttrMeta(**meta) for attr, meta in attr_meta_map.items()}
                cls.attr_meta_maps.append(attr_meta_map)

        cls.dist_param_map = torch.load(cls.module_dir / FxModuleParser.ATTR_MAP_FILE, weights_only=False)
        cls.compute_config = ComputeConfig.safe_load_from_file(
            cls.module_dir / cls.COMPUTE_CONFIG_FILE,
            return_none_on_error=False
        )
        cls.origin_module_metadata = torch.load(cls.module_dir / cls.ORIGIN_MODULE_METADATA_FILE, weights_only=False)

    @property
    def non_presistent_buffers_inited(self):
        return self._non_presistent_buffers_inited

    def mark_non_persistent_buffers_inited(self):
        self._non_presistent_buffers_inited = True

    def _warn_uninitialized_non_persistent_buffers(self, raise_error = False):
        _non_persistent_buffers_load_warning = (
            "Non-persistent buffers cannot be initialized with load_[/merged/dedupped]state_dict. "
            "Please be sure to you will initialize them manually. "
        )
        _non_persistent_buffers_load_error = (
            "Non-persistent buffers haven't been initialized."
        )
        if not self._non_presistent_buffers_inited:
            if raise_error:
                raise RuntimeError(_non_persistent_buffers_load_error)
            else:
                _logger.warning(_non_persistent_buffers_load_warning)

    def _post_init(self, init_params=True, build_buckets=True):
        """
        This is post init function to further initialize the model. Should be called by subclass's __init__().

        Args:
            init_params (bool): whether to load model init parameters. Default True.
            build_buckets (bool): whether to build buckets for the model. Default True.
                If it is False, you must manually call `build_buckets()` later before use this module.
        """
        # Here we check the rank to load the module file name
        # Current we don't check rank when we are not in distributed mode
        # to facilitate local debugging
        # TODO: re-enable this check
        # if dist.is_initialized() and self.rank != dist.get_rank():
        #     raise RuntimeError(f"The rank to load this module file name is expected to be {self._rank}, but got {dist.get_rank()}")

        self._non_presistent_buffers_inited = init_params or not self._non_persistent_buffers_set
        module_file = Path(sys.modules[self.__module__].__file__)
        if init_params:
            self.load_attr_content(str(module_file.with_name(f"{FxModuleParser.ATTR_CONTENT_FILE_STEM}")))

        self._warn_uninitialized_non_persistent_buffers()

        # add state_dict hook to save extra state
        # Please note extra_state is only used for merging, not for loading
        # so we can safely remove it in load_state_dict pre hook
        self._register_state_dict_hook(ParallelModule._post_state_dict_hook)
        # add load_state_dict pre hook to pop extra state to prevent warning
        self._register_load_state_dict_pre_hook(ParallelModule._pre_load_state_dict_hook, with_module=True)

        if build_buckets:
            self.build_buckets()

    def build_buckets(self, param_clss: Optional[dict[torch.nn.Parameter, Any]]=None):
        """
        Build buckets for the model reducers.

        You should call this method exactly once before using this module.
        Typically this will be called when building optimizer when multiple optimizers/param groups are used.
        And we will put parameters with different optimizer or different param groups into different buckets.

        Currently we have done an optimization to make sure this is only called once even for hybrid optimizers
        by
        1. setting `build_buckets=False` when calling constructor in `nnscaler.parallelize`.
        2. manually calling `build_buckets()` later in `nnscaler.build_optimizer`
        """
        # needs all parameters to be in cuda memory before building buckets
        self.cuda()
        self._param_reducer_map: dict[torch.nn.Parameter, int] = {}
        model_params = {p: n for n, p in self.named_parameters()}
        # key: attr name of the parameter
        # value: Zero3AttrMeta
        self._zero3_param_metadata: dict[str, Zero3AttrMeta] = {}
        for idx, reducer in enumerate(self.reducers):
            reducer.build_buckets(param_clss)
            for param in reducer.params:
                self._param_reducer_map[param] = idx
                attr_name = model_params[param]
                param_attr = self._fullmap[attr_name]
                zero3_info = reducer.get_z3_info(param)
                self._zero3_param_metadata[attr_name] = Zero3AttrMeta(
                    attr_name=attr_name,
                    orig_name=param_attr.orig_name,
                    start = zero3_info.start,
                    end = zero3_info.end,
                    chunk_size=zero3_info.numel_with_padding(),
                ) if zero3_info is not None else None

        self._zero_metadata = self._get_zero_metadata()

    def get_zero3_attr_meta(self, attr_name: str) -> Optional[Zero3AttrMeta]:
        """
        Get the Zero3AttrMeta for the given attribute name.

        Args:
            attr_name (str): the attribute name of the parameter
        Returns:
            Optional[Zero3AttrMeta]: the Zero3AttrMeta for the given attribute name
        """
        return self._zero3_param_metadata.get(attr_name, None)

    @torch.no_grad()
    def prefetch_param(self, param: torch.nn.Parameter):
        """
        Gather the full parameter tensor for FSDP.

        Args:
            param (torch.nn.Parameter): the local parameter to gather
        """
        reducer = self._reducers[self._param_reducer_map[param]]
        reducer.prefetch_param(param)
        self._forward_prefetched_params.add(param)

    @torch.no_grad()
    def postevict_param(self, param: torch.nn.Parameter):
        """
        Release the full parameter tensor for zero3.

        Args:
            param (torch.nn.Parameter): the local parameter
        """
        reducer = self._reducers[self._param_reducer_map[param]]
        reducer.postevict_param(param)
        self._forward_prefetched_params.discard(param)

    def _backward_evict_leftover_params(self, order: int):
        for p in [p for p, o in self._backward_prefetched_params.items() if o > order]:
            self.postevict_param(p)
            self._backward_prefetched_params.pop(p, None)

    def backward_postevict_param(self, input: torch.Tensor, param: torch.nn.Parameter, order: int):
        """
        Here we need an input tensor to register the backward hook.
        """
        if not input.requires_grad:
            # if input does not require grad, we cannot register backward hook on it
            return input

        @torch.no_grad()
        def _postevict_param(param): # pragma: no cover
            self.postevict_param(param)
            self._backward_prefetched_params.pop(param, None)
            self._backward_evict_leftover_params(order)

        return insert_backward_hook(input, functools.partial(_postevict_param, param))

    def backward_prefetch_param(self, activation: torch.Tensor, param: torch.nn.Parameter, order: int):
        """
        Here we need an activation tensor to register the backward hook.
        """
        if not activation.requires_grad:
            # if activation does not require grad, we cannot register backward hook on it
            return activation

        @torch.no_grad()
        def _prefetch_param(param): # pragma: no cover
            self.prefetch_param(param)
            self._backward_prefetched_params[param] = order
            self._backward_evict_leftover_params(order)

        return insert_backward_hook(activation, functools.partial(_prefetch_param, param))

    def save_params_hooks(self) -> saved_tensors_hooks:
        """
        A hook to save tensors during forward pass.
        This is used to avoid parameters being saved for activation checkpointing.

        Returns:
            saved_tensors_hooks: the saved tensors hooks
        """
        def pack(x: torch.Tensor):
            for param in self._forward_prefetched_params:
                if x.untyped_storage() == param.untyped_storage():
                    return (param, x.shape, x.stride(), x.storage_offset())
            return x

        def unpack(x):
            if isinstance(x, tuple) and len(x) == 4:
                return torch.as_strided(x[0], x[1], x[2], x[3])
            return x

        return saved_tensors_hooks(pack, unpack)

    @classmethod
    def get_attr_meta_map(cls, rank=None):
        """
        Get the attribute meta map for the given rank.
        If rank is None, return the attribute map for the current rank.

        This function is preferred over accessing `CubeModule.fullmap` in most cases,
        since it doesn't need to instantiate the module.
        """
        if rank is None:
            rank = cls.rank
        if rank < 0 or rank >= cls.world_size:
            raise ValueError(f"Rank {rank} is out of range [0, {cls.world_size})")
        return cls.attr_meta_maps[rank]

    def forward(self, *args, **kwargs):
        self._warn_uninitialized_non_persistent_buffers(raise_error=True)
        if self.training:
            self._sync_grad_required = True  # mark sync_grad() can be called again
        # all prefetched params should have been evicted
        # please note the param can be evicted in Reducer,
        # which is not tracked in self._backward_prefetched_params
        # so we just check the shape to make sure the param is evicted
        for param in self._backward_prefetched_params.keys():
            old_shape = param.shape
            self.postevict_param(param)
            assert param.shape == old_shape, \
                f'Param {param} is not properly evicted in backward'
        self._backward_prefetched_params.clear()

        ret = self._forward_impl(*args, **kwargs)

        assert not self._forward_prefetched_params, \
            f'All forward prefetched params should have been evicted in forward'
        return ret

    def _forward_impl(self, *args, **kwargs):
        """
        forward implementation. Should be implemented by subclass
        """
        raise NotImplementedError

    def sync_grad(self):
        """
        synchronize gradients using allreduce (non-zero) or reduce-scatter (zero)
        """
        if self._sync_grad_required:
            self._sync_grad_required = False  # mark sync_grad() has been called
            for reducer in self._reducers:
                reducer.sync_grads()

    def _train_step(self, dataloader) -> Union[List[Any], Any]:
        """
        This function is assigned automatically when loading end2end module class
        Returns:
            Union[List[Any], Any]: the output of the training step,
                In Pipeline mode, it should return a list of outputs for each sample
                Otherwise, it should return a single output
        """
        raise NotImplementedError

    def _infer_step(self, dataloader) -> Union[List[Any], Any]:
        """
        This function is assigned automatically when loading end2end module class
        Returns:
            Union[List[Any], Any]: the output of the training step,
                In Pipeline mode, it should return a list of outputs for each sample
                Otherwise, it should return a single output
        """
        raise NotImplementedError

    def _scale_loss(self, is_dummy_batch: Optional[List[bool]], scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        """Setup cube backward hook for loss scale and dummy batch.

        If the batch is a dummy batch, the loss will be 0 to make the
        gradient 0.

        Args:
            is_dummy_batch (List[bool]): indicate whether the each micro-batch is dummy
            scale_fn (Callable[[torch.Tensor], torch.Tensor]): the function to scale the loss
        """

        # clear the previous hook
        Executor.register_backward_pre_hook(None)

        if not is_dummy_batch and not scale_fn:
            return

        accum_idx = 0
        def cube_scale(ins, outs, grads):
            nonlocal accum_idx
            if is_dummy_batch and accum_idx >= len(is_dummy_batch):
                raise RuntimeError(
                    f"Expected {len(is_dummy_batch)} number of micro-batches, but got more than it."
            )
            mul_coef = 0.0 if is_dummy_batch and is_dummy_batch[accum_idx] else 1.0
            # find loss
            for idx in range(len(outs)):
                # loss always requires to be a scalar, and its gradient should be None
                if grads[idx] is None:
                    assert idx == 0, "Loss must be the first output."
                    if outs[idx].size() != torch.Size([]):
                        raise ValueError(f"Expected scalar loss, but got {outs[idx].size()}.")
                    if scale_fn:
                        outs[idx] = mul_coef * scale_fn(outs[idx])
                    else:
                        outs[idx] = mul_coef * outs[idx]
                    break
            accum_idx += 1
            return ins, outs, grads

        Executor.register_backward_pre_hook(cube_scale)

    def train_step(self,
        samples: List[Any],
        is_dummy_batch: Optional[List[bool]] = None,
        scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> List[Any]:
        """
        The training step function. It should be called in the training loop.
        Please note:
            1. This function is only supported in end2end mode.
            2. Gradient accumulation is done inside this function.
                You shouldn't do gradient accumulation outside this function,
                because the gradients will be cleared in the beginning of this function
        Args:
            samples (List[Any]): a list of samples.
                if pipeline is used, it must have the same length as configured to pas policy
            is_dummy_batch (Optional[List[bool]]): indicates whether the each micro-batch is dummy
            scale_fn (Optional[Callable[[torch.Tensor], torch.Tensor]]): the function to scale the loss
        Results:
            List[Any]: a list of outputs for each sample
        """
        self._warn_uninitialized_non_persistent_buffers(raise_error=True)

        if not self.compute_config.use_end2end:
            raise RuntimeError("train_step() is only supported in end2end mode")
        if is_dummy_batch and len(samples) != len(is_dummy_batch):
            raise ValueError("The length of samples and is_dummy_batch should be the same")

        self._scale_loss(is_dummy_batch, scale_fn)

        # sync_grad will be done in _train_step
        # so we never need to call it manually
        self._sync_grad_required = False
        sample_count = len(samples)
        dataloader = microbatches(samples, cycle=False)

        if self.use_scheduler:
            if len(samples) != self.nmicros_per_scheduler_step:
                raise ValueError(f"Expected {self.nmicros_per_scheduler_step} samples, but got {sample_count}")
            # only one step, so begin/end are both True
            with accum_mode(begin=True, end=True):
                return self._train_step(dataloader)
        else:
            outputs = []
            for idx in range(sample_count):
                with accum_mode(begin=(idx==0), end=(idx==sample_count-1)):
                    output = self._train_step(dataloader)
                outputs.append(output)
            return outputs

    def infer_step(self, samples: List[Any]) -> List[Any]:
        """
        The inference step function. It should be called in the inference loop.
        Please note this function is only supported in end2end mode.

        Args:
            samples (List[Any]): a list of samples.
                if pipeline is used, it must have the same length as configured to pas policy
        Results:
            List[Any]: a list of outputs for each sample
        """
        self._warn_uninitialized_non_persistent_buffers(raise_error=True)

        if not self.compute_config.use_end2end:
            raise RuntimeError("infer_step() is only supported in end2end mode")

        sample_count = len(samples)
        dataloader = microbatches(samples, cycle=False)
        if self.use_scheduler:
            if len(samples) != self.nmicros_per_scheduler_step:
                raise ValueError(f"Expected {self.nmicros_per_scheduler_step} samples, but got {sample_count}")
            return self._infer_step(dataloader)
        else:
            outputs = []
            for _ in range(sample_count):
                output = self._infer_step(dataloader)
                outputs.append(output)
            return outputs

    def clip_gnorm(self, max_norm: Optional[float] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Calculate the gradient norm and clip gradients.

        Args:
            max_norm (Optional[float]): max norm value. If None or <= 0, no clipping will be performed.

        Returns:
            Tuple of The gradient norm and the list of gradients.
        """
        from nnscaler.runtime.gnorm import prepare_for_grad_clip, clip_gnorm
        if self._nreplicas2localparams is None:
            self._nreplicas2localparams = prepare_for_grad_clip(self, self.compute_config.use_zero)

        # make sure the gradients are synchronized
        self.sync_grad()

        return clip_gnorm(self._nreplicas2localparams, max_norm)

    def _get_zero_metadata(self) -> ZeroMetadata:
        """
        Get zero related metadata for checkpointing.

        In this function, we have a mocked optimizer index representing the combined flattened index of (reducer_index, bucket_index)

        Note:
        Parameters can be in one bucket or not in any bucket.
        When we need to reduce(sume) the gradient of a parameter across ranks,
        the parameters will be added in one reducer based on the rank group.
        There are two types of reducing: cross scale unit or intra scale unit.

        So when num of scale unit > 1, the parameters have to be reduced across scale units,
        so they will be in a reducer.

        When the num of scale unit == 1, the parameters can still need to be reduced inside the scale unit,
        when a parameter is replicated due to its operator's partition (i.e., through graph.partition)
        (when the paremeter is used by multiple ops,
        but some of ops are partitioned and some of ops are replicated,
        In that case, the parameter will not be in a reudcer.
        We will use mutliref, and insert identity-allreduce in generated code to reduce the parameter instead of using a reducer.
        )

        Returns:
            ZeroMetadata: the zero related metadata
        """
        if not self.compute_config.use_zero:
            return ZeroMetadata()

        model_params = self.parameters_for_optimizer()
        opt_idx = 0  # the combined flattened index of (reducer_index, bucket_index)
        # key: the index of the parameter in the model
        # value: (opt_idx, param_start, param_end, param_shape)
        #   where param_start and param_end are the start and end index of the parameter in the bucket
        model_idx2opt_idx: Dict[int, Tuple[int, int, int, torch.Size]] = {}
        # key: opt_idx
        # value: (sub_ranks, bucket_size), If value is None, then the parameter is not in a bucket
        opt_idx2ranks: Dict[int, Optional[Tuple[List[int], int]]] = {}
        model_params_id = [id(param) for param in self.parameters()]

        for reducer in self.reducers:
            _, sub_ranks = self._get_zero_subranks(reducer)
            for bucket in reducer.buckets:
                pstart, pend = 0, 0
                for param in bucket.params:
                    pstart = pend
                    pend = pstart + bucket.get_aligned_numel(param)
                    pend_without_padding = pstart + param.numel()
                    model_idx = model_params_id.index(id(param))
                    model_idx2opt_idx[model_idx] = (opt_idx, pstart, pend_without_padding, param.shape)
                assert len(bucket._contiguous_params.shape) == 1
                opt_idx2ranks[opt_idx] = (sub_ranks, bucket._contiguous_params.shape[0])
                opt_idx += 1

        assert len(model_params) >= opt_idx
        # The remaining parameters are not in any bucket
        # we assign them to the next available opt_idx
        # and set the opt_idx2ranks to None
        for param in model_params[opt_idx:]:
            model_idx = model_params_id.index(id(param))
            model_idx2opt_idx[model_idx] = opt_idx
            opt_idx2ranks[opt_idx] = None
            opt_idx += 1

        assert len(model_params) == opt_idx

        return ZeroMetadata(
            model_idx2opt_idx=model_idx2opt_idx,
            opt_idx2ranks=opt_idx2ranks,
            zero=self.compute_config.use_zero
        )

    def _get_zero_subranks(self, reducer: Reducer) -> Tuple[int, List[int]]:
        """
        Get the index in the zero subgroup the reducer belongs to, and the ranks of the subgroup.

        Args:
            reducer (nnscaler.runtime.adapter.Reducer): a reducer of cube model

        Returns:
            rank_idx (int): the index of current rank in sub_ranks
            sub_ranks (list): the ranks of ZeRO subgroup the current rank belongs to
        """
        cf = self.compute_config
        if not cf.use_zero:
            raise RuntimeError('ZERO is not enabled, cannot get the zero subgroup info')

        rank_idx = reducer.ranks.index(self.rank)
        if cf.zero_ngroups > 1:
            assert len(reducer.ranks) % cf.zero_ngroups == 0, \
                f'reducer.ranks {reducer.ranks} should be divisible by ZERO_NUM_GROUPS {cf.zero_ngroups}'
            zgroup_sz = len(reducer.ranks) // cf.zero_ngroups
            group_idx = rank_idx // zgroup_sz
            sub_ranks = reducer.ranks[group_idx * zgroup_sz : (group_idx + 1) * zgroup_sz]
            new_rank_idx = sub_ranks.index(self.rank)
            return new_rank_idx, sub_ranks
        else:
            assert cf.zero_ngroups == 1
            return rank_idx, reducer.ranks

    def _add_extra_state(self, state_dict, prefix) -> None:
        state_dict[f'{prefix}{self.EXTRA_STATE_KEY}'] = asdict(
            ExtraState(
                rank=self.rank,
                compute_config=self.compute_config,
                dist_param_map=self.dist_param_map,
                param_area_map=self._fullmap,
                cube_param_names=[name for name, _ in self.named_parameters()],
                **asdict(self.origin_module_metadata),
                **asdict(self._zero_metadata),
            )
        )

    def _remove_extra_state(self, state_dict, prefix) -> None:
        state_dict.pop(f'{prefix}{self.EXTRA_STATE_KEY}', None)

    def _post_state_dict_hook(self, state_dict, prefix, local_metadata) -> None:
        self._add_extra_state(state_dict, prefix)

    def _pre_load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        self._remove_extra_state(state_dict, prefix)
        # Both load_state_dict and load_deduped_state_dict will trigger this hook
        self._warn_uninitialized_non_persistent_buffers()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Override the default load_from_state_dict to support loading from a merged state dict.
        Please note
         1. we have to trigger the pre_load_state_dict_hook explicitly when loading merged state dict,
         2. post_load_state_dict_hook will be triggered in `super().load_state_dict`.
        """
        if f'{prefix}{self.EXTRA_STATE_KEY}' in state_dict:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs
            )
        else:
            for hook in self._load_state_dict_pre_hooks.values():
                hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            new_missing_keys = self.load_merged_state_dict(state_dict, prefix, strict=False)
            if strict:
                missing_keys.extend(new_missing_keys)

    @classproperty
    def module_dedup_group_size(cls) -> int:
        """
        Get the size of the deduplication group of the model state dict, which is `plan_ngpus`.
        """
        return cls.compute_config.module_dedup_group_size

    @classproperty
    def optimizer_dedup_group_size(cls) -> int:
        """
        Get the size of the deduplication group of the optimizer state dict.
        """
        return cls.compute_config.optimizer_dedup_group_size

    def _list_fullmodel_files(self) -> List[Path]:
        legacy_fullmodel_path = self.module_dir / FxModuleParser.ATTR_CONTENT_FILE_STEM
        files = []
        if not legacy_fullmodel_path.is_file():
            file_idx = 0
            while True:
                filepath = self.module_dir / FxModuleParser.ATTR_CONTENT_FILE_FORMAT.format(stem=FxModuleParser.ATTR_CONTENT_FILE_STEM, idx=file_idx)
                if not filepath.is_file():
                    break
                files.append(filepath)
                file_idx += 1
        else:
            files.append(legacy_fullmodel_path)

        return files

    def load_merged_state_dict(self, state_dict: Dict[str, Any], prefix: str = '', strict: bool = True):
        """
        Load the model from a merged state dict.

        Args:
            state_dict (Dict[str, Any]): the merged state dict
            prefix (str): the prefix of the model state dict in the merged state dict
            strict (bool, optional): whether to strictly enforce that state_dict has have all the parameters of the module
                Note: unlike `torch.nn.Module.load_state_dict`,
                we only make sure no missing keys. Unexpected keys are not checked.
                Default: `True`
        Returns:
            list[str]: a list of missing keys in the state_dict.
            Please note currently we don't check unexpected keys.
        Raises:
            RuntimeError: if strict=True and there are missing keys.
        """
        non_persistent_buffers = self.get_non_persistent_buffers()

        with torch.no_grad():
            # avoid checking the non-persistent buffers
            attr_names = set([attr for attr in self._fullmap.keys() if attr not in non_persistent_buffers])

            for prefix_attr, content in self.trim_merged_state_dict(state_dict, prefix).items():
                attr = prefix_attr[len(prefix):]
                tensor: torch.Tensor = getattr(self, attr)
                tensor.copy_(content)
                attr_names.remove(attr)

            missing_keys = [prefix + self._fullmap[attr].orig_name for attr in attr_names]
            if len(attr_names) != 0:
                erro_msg = f'Missing key(s) in state_dict: {missing_keys}.'
                if strict:
                    raise RuntimeError(erro_msg)
                else:
                    _logger.warning(erro_msg)

        self._warn_uninitialized_non_persistent_buffers()
        return missing_keys

    def trim_merged_state_dict(
            self,
            state_dict: Dict[str, Any],
            prefix: str = '',
            *,
            device=None,
    ) -> Dict[str, Any]:
        """
        Trim the merged state dict to only keep the parameters needed for the module.
        Please note we don't check missing/unexpected keys.

        Args:
            state_dict (Dict[str, Any]): the merged state dict
            prefix (str): the prefix of the model state dict in the merged state dict

        Returns:
            Dict[str, Any]: the trimmed state dict
        """
        device = device or torch.cuda.current_device()
        trimmed_state_dict = {}

        dist2param = self.dist_param_map
        orig_param_names = list(dist2param.values())  # param names in original module (without prefix)
        attr_meta_map = self.get_attr_meta_map(self.rank)
        with torch.no_grad():
            # avoid checking the non-persistent buffers
            origname_tid_map = {meta.orig_name: meta.tid for meta in attr_meta_map.values()}
            tid_info = defaultdict(list)
            for attr, meta in attr_meta_map.items():
                tid_info[meta.tid].append((attr, meta.slicers, meta.val_chunks))  # multiple params may share the same tid

            for orig_param_name in orig_param_names:
                if orig_param_name not in origname_tid_map:
                    # in pipeline mode, the parameter may not be in this rank
                    continue
                orig_param_name_with_prefix = prefix + orig_param_name
                if orig_param_name_with_prefix not in state_dict:
                    continue
                param_value = state_dict[orig_param_name_with_prefix]
                tid = origname_tid_map[orig_param_name]
                for attr, slicer, nchunks in tid_info[tid]:
                    content: torch.Tensor = param_value[slicer]
                    if nchunks != 1:
                        content = content / nchunks
                    if self.compute_config.use_zero <= 1 or self._zero3_param_metadata.get(attr, None) is None:
                        trimmed_state_dict[prefix + attr] = content.to(device)
                    else:
                        z3_info = self._zero3_param_metadata[attr]
                        start, end, chunk_size = z3_info.start, z3_info.end, z3_info.chunk_size
                        if end - start < chunk_size:
                            # need padding
                            padding = chunk_size - (end - start)
                            trimmed_state_dict[prefix + attr] = torch.cat([
                                content.view(-1)[start:end],
                                torch.zeros(padding, dtype=content.dtype, device=content.device)
                            ], dim=0).to(device)
                        else:
                            trimmed_state_dict[prefix + attr] = content.reshape(-1)[start:end].to(device)

        return trimmed_state_dict

    def _pack(
        self,
    ):
        """
        Get a packed information of the ParallelModule, so it can be sent to other ranks.
        """
        param_map: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        for p in self.parameters():
            param_map[p] = torch.nn.Parameter(
                torch.empty_like(p, device='meta')) if p is not None else None
        for b in self.buffers():
            param_map[b] = torch.empty_like(
                b, device='meta') if b is not None else None
        state = {}
        fields = unchecked_fields(self)
        state[fields._parameters] = {n: param_map[p] for n, p in self._parameters.items()}
        state[fields._buffers] = {n: param_map[b] for n, b in self._buffers.items()}
        state[fields._reducers] = [reducer._pack(param_map) for reducer in self._reducers]
        state[fields._zero_metadata] = self._zero_metadata
        state[fields._fullmap] = self._fullmap
        state[fields._param_reducer_map] = {
            param_map[p]: rid for p, rid in self._param_reducer_map.items()
        }
        state[fields._zero3_param_metadata] = self._zero3_param_metadata

        for cv in ParallelModule.__annotations__:
            state[cv] = getattr(self, cv)
        return state

    @classmethod
    def _unpack(cls, state: dict):
        """
        Unpack the information and return a fake ParallelModule that carries the same information.
        """
        class GenModelX(ParallelModule, skip_init=True):
            pass
        pm = object.__new__(GenModelX)
        fields = unchecked_fields(pm)
        object.__setattr__(pm, fields._parameters, state[fields._parameters])
        object.__setattr__(pm, fields._buffers, state[fields._buffers])
        object.__setattr__(pm, fields._reducers, [Reducer._unpack(reducer) for reducer in state[fields._reducers]])
        object.__setattr__(pm, fields._zero_metadata, state[fields._zero_metadata])
        object.__setattr__(pm, fields._fullmap, state[fields._fullmap])
        object.__setattr__(pm, fields._param_reducer_map, state[fields._param_reducer_map])
        object.__setattr__(pm, fields._zero3_param_metadata, state[fields._zero3_param_metadata])

        def named_parameters(
            prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
        ):
            assert prefix == "" and recurse is True, "Only support default arguments"
            return pm._parameters.items()

        pm.named_parameters = named_parameters

        for cv in ParallelModule.__annotations__:
            setattr(GenModelX, cv, state[cv])
        return pm
