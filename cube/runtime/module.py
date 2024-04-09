from typing import List, Set, Dict, Tuple, Optional, TYPE_CHECKING, Any
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.distributed as dist

from cube.graph.parser.fx.parser import FxModuleParser
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.gnorm import ParamsInfo
from cube.flags import CompileFlag

if TYPE_CHECKING:
    from cube.parallel import ComputeConfig


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

class CubeModule(torch.nn.Module):
    """
    The module is responsible for parameter synchronization
    before training
    """

    def __init__(self):
        super().__init__()
        self._reducers: List[Reducer] = list()
        # self._fullmap is mapping from the name of local attribute tensor
        # to its corresponding fulltensor meta
        # please note there can be multiple entries with same tid
        self._fullmap : Dict[str, AttrMeta] = dict()

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

        If the function is under the context of `with cube.accum_mode()`, the zero of gradients
        will be skipped.
        """
        for reducer in self._reducers:
            reducer.zero_grad()

    def parameters_for_optimizer(self) -> List[torch.nn.Parameter]:
        """Get parameter list for optimizer"""
        params = []
        reducer_pids = set()
        for reducer in self._reducers:
            params += reducer.parameters_for_optimizer()
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                params.append(param)
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
        meta = AttrMeta(tid, is_param, orig_name, shape, slicers, val_chunks)
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
        dist_param_map = getattr(self, '_dist_param_map', None)
        if not dist_param_map:
            module_file = Path(sys.modules[self.__module__].__file__)
            # load from the same directory as the module file
            dist_param_map = torch.load(module_file.with_name(FxModuleParser.ATTR_MAP_FILE))
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

    @staticmethod
    def merge_model_state_dicts(
        state_dicts: List[Dict],
        fullmaps: List[Dict[str, AttrMeta]]
    ):
        """Merge model states from multiple shard into a single-model state.

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
        # gather param/buffer full tensor
        for model_state_dict, local_fullmap in zip(state_dicts, fullmaps):
            for local_name, meta in local_fullmap.items():
                # create full tensor on cpu
                partial_tensor = model_state_dict[local_name]
                if meta.orig_name not in full_model_state_dict:
                    full_model_state_dict[meta.orig_name] = torch.empty(
                        meta.shape, dtype=partial_tensor.dtype)
                # assign partial tensor
                if meta.val_chunks > 1:
                    raise NotImplementedError("Not support of partitioning parameter / buffer at value dimension")
                full_model_state_dict[meta.orig_name][meta.slicers] = partial_tensor
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

        # at first, merge the partitioned optimizer states due to zero to the zero-disabled format
        if zero_idx_maps is not None:
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

            def _retrieve_param_opt_state(bucket_states, pstart, pend, pshape, bucket_size):
                assert bucket_size % len(bucket_states) == 0
                opt_state_keys = list(bucket_states[0].keys())
                if 'step' in bucket_states[0]:
                    opt_state_keys.remove('step')
                assert _check_state_size(opt_state_keys, bucket_states[0]), f'the keys {opt_state_keys} have different shape'
                # NOTE: only support adam for now
                assert 'exp_avg' in opt_state_keys
                assert 'exp_avg_sq' in opt_state_keys
                chunk_size = bucket_size // len(bucket_states)
                start_rank_id, start_offset = pstart // chunk_size, pstart % chunk_size
                end_rank_id, end_offset = pend // chunk_size, pend % chunk_size
                opt_states, opt_states_1d = {}, {}
                for key in opt_state_keys:
                    opt_states[key] = torch.zeros(pshape, dtype=bucket_states[0][key].dtype,
                                                    device=bucket_states[0][key].device, requires_grad=False)
                    opt_states_1d[key] = opt_states[key].view(-1)

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

                if 'step' in bucket_states[0]:
                    opt_states['step'] = bucket_states[0]['step']
                return opt_states

            # Parameters are partitioned inside a scale unit composed of plan_ngpus GPUs.
            # When ZeRO-1 is enabled, optimizer states (like exp_avg and exp_avg_sq) are partitioned within
            # each ZeRO group. Since the training is done in a synchronized way, the optimizer states are
            # identical across each ZeRO group.
            # As a result, we can retrieve and merge the optimizer states in other scale units following the
            # information stored in zero_idx_maps ONLY for the first scale unit.
            opt_state_list = []
            for work_idx in range(plan_ngpus):
                model_idx2opt_idx, opt_idx2ranks = zero_idx_maps[work_idx]
                opt_state = {}
                for model_idx, opt_idx in model_idx2opt_idx.items():
                    if isinstance(opt_idx, int):
                        # the param without reducer
                        assert opt_idx2ranks[opt_idx] is None
                        opt_state[model_idx] = optim_state_dicts[work_idx]['state'][opt_idx]
                    else:
                        # the param in reducer bucket
                        opt_idx, pstart, pend, pshape = opt_idx
                        ranks, bucket_size = opt_idx2ranks[opt_idx]
                        bucket_states = [optim_state_dicts[rank]['state'][opt_idx] for rank in ranks]
                        opt_state[model_idx] = _retrieve_param_opt_state(
                            bucket_states,
                            pstart,
                            pend,
                            pshape,
                            bucket_size)
                _logger.info(f'finish handle optimizer state for worker {work_idx}')
                opt_state_list.append(opt_state)
                assert len(optim_state_dicts[work_idx]['param_groups']) == 1, 'only support param_groups to be one group'

            # assign opt_state to state_dicts, cannot be assigned in the above loop
            opt_state_len = len(opt_state_list[0])
            for work_idx in range(plan_ngpus):
                optim_state_dicts[work_idx]['state'] = opt_state_list[work_idx]
                optim_state_dicts[work_idx]['param_groups'][0]['params'] = sorted(opt_state_list[work_idx].keys())
                _logger.info(f'finish assign optimizer state for worker {work_idx}')
                assert len(opt_state_list[work_idx]) == opt_state_len

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
        # full_index: param IDs in the full optimizer state
        for full_index, param_name in enumerate(origin_parameter_names):
            _logger.info(f'start to handle optimizer state for param {param_name} with full_index {full_index}')
            for optim_state, fullmap in zip(optim_state_dicts[0 : plan_ngpus], fullmaps[0 : plan_ngpus]):
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
                    states: Dict[str, torch.Tensor] = optim_state['state'][local_index]
                    for state_name in states.keys():
                        value = states[state_name]
                        # special handle for step: scalar tensor type
                        if state_name == 'step':
                            full_states[full_index][state_name] = value
                            continue
                        # for non-tensor states
                        if not isinstance(value, torch.Tensor):
                            full_states[full_index][state_name] = value
                        # for tensor states, like 'exp_avg'
                        else:
                            # create optimizer state tensor
                            if state_name not in full_states[full_index]:
                                full_states[full_index][state_name] = torch.empty(meta.shape, dtype=value.dtype)
                            # assign with partial tensor
                            full_states[full_index][state_name][meta.slicers] = value

        # handle additional state dict keys
        for optim_state_dict in optim_state_dicts[0 : plan_ngpus]:
            for key in optim_state_dict.keys():
                if key != 'state':
                    if key in full_optim_state_dict:
                        _logger.info(f'overwrite optimizer state key {key}')
                    else:
                        _logger.info(f'inherit optimizer state key {key}')
                    full_optim_state_dict[key] = optim_state_dict[key]

        return full_model_state_dict, full_optim_state_dict

    @staticmethod
    def merge_checkpoints(filename_prefix='dist_checkpoint'):
        ckpts = {}
        for rank in range(DeviceGroup().world_size):
            filename = f"{filename_prefix}-{rank}.ckpt"
            ckpts[rank] = torch.load(filename)
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
            from cube.parallel import ComputeConfig
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
    # the rank of the module, will be assigned in the generated subclasses
    rank: int

    def __init__(self):
        if self.__class__  == ParallelModule:  # not init via super().__init__()
            raise RuntimeError(f"ParallelModule should not be initialized directly. Please derive it first")

        super().__init__()
        # this is used to allow multiple sync_grad() calls
        self._sync_grad_required = False
        # save the param replicas info for calculating gradient norm
        # it is a dict mapping from number_of_replicas to a list of local params.
        # For example, _nreplicas2localparams[2] contains all the parameters that have replicated 2 times.
        # this is a lazy initialization,
        # which will be initialized in the first call of `clip_gnorm`
        self._nreplicas2localparams: Optional[Dict[int, List[torch.nn.Parameter]]] = None

    def _post_init(self, init_params=True):
        """
        This is post init function to further initialize the model. Should be called by subclass's __init__().

        Args:
            init_params (bool): whether to load model init parameters. Default True.
        """
        # Here we check the rank to load the module file name
        # Current we don't check rank when we are not in distributed mode
        # to facilitate local debugging
        # TODO: re-enable this check
        # if dist.is_initialized() and self.rank != dist.get_rank():
        #     raise RuntimeError(f"The rank to load this module file name is expected to be {self._rank}, but got {dist.get_rank()}")

        module_file = Path(sys.modules[self.__module__].__file__)
        self.module_dir = module_file.parent
        if init_params:
            self.load_attr_content(str(module_file.with_name(f"{FxModuleParser.ATTR_CONTENT_FILE_STEM}")))
        self._dist_param_map = torch.load(module_file.with_name(f"{FxModuleParser.ATTR_MAP_FILE}"))
        self._compute_config: 'ComputeConfig' = torch.load(module_file.with_name(f"{self.COMPUTE_CONFIG_FILE}"))
        self._orign_module_metadata: OriginModuleMetadata = torch.load(module_file.with_name(f"{self.ORIGIN_MODULE_METADATA_FILE}"))

        for reducer in self.reducers:
            reducer.build_buckets()

        self._zero_metadata = self._get_zero_metadata()

        # add state_dict hook to save extra state
        # Please note extra_state is only used for merging, not for loading
        # so we can safely remove it in load_state_dict pre hook
        self._register_state_dict_hook(ParallelModule._post_state_dict_hook)
        # add load_state_dict pre hook to pop extra state to prevent warning
        self._register_load_state_dict_pre_hook(ParallelModule._pre_load_state_dict_hook, with_module=True)

    def forward(self, *args, **kwargs):
        if self.training:
            self._sync_grad_required = True  # mark sync_grad() can be called again
        return self._forward_impl(*args, **kwargs)

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

    @property
    def dist_param_map(self) -> Dict[str, str]:
        """
        Get the parameter map of the model.
        The map is a dict mapping from the new parameter name (without tid suffix) in parallel module
            to the parameter name in original module.
        """
        return self._dist_param_map

    @property
    def compute_config(self) -> 'ComputeConfig':
        return self._compute_config

    def clip_gnorm(self, max_norm: Optional[float] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Calculate the gradient norm and clip gradients.

        Args:
            max_norm (Optional[float]): max norm value. If None or <= 0, no clipping will be performed.

        Returns:
            Tuple of The gradient norm and the list of gradients.
        """
        from cube.runtime.gnorm import prepare_for_grad_clip, clip_gnorm
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
                    pend += param.numel()
                    model_idx = model_params_id.index(id(param))
                    model_idx2opt_idx[model_idx] = (opt_idx, pstart, pend, param.shape)
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
        )

    def _get_zero_subranks(self, reducer: Reducer) -> Tuple[int, List[int]]:
        """
        Get the index in the zero subgroup the reducer belongs to, and the ranks of the subgroup.

        Args:
            reducer (cube.runtime.adapter.Reducer): a reducer of cube model

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
                compute_config=self._compute_config,
                dist_param_map=self._dist_param_map,
                param_area_map=self._fullmap,
                cube_param_names=[name for name, _ in self.named_parameters()],
                **asdict(self._orign_module_metadata),
                **asdict(self._zero_metadata),
            )
        )

    def _remove_extra_state(self, state_dict, prefix) -> None:
        state_dict.pop(f'{prefix}{self.EXTRA_STATE_KEY}', None)

    def _post_state_dict_hook(self, state_dict, prefix, local_metadata) -> None:
        self._add_extra_state(state_dict, prefix)

    def _pre_load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        self._remove_extra_state(state_dict, prefix)

    @property
    def module_dedup_group_size(self) -> int:
        """
        Get the size of the deduplication group of the model state dict, which is `plan_ngpus`.
        """
        return self.compute_config.module_dedup_group_size

    @property
    def optimizer_dedup_group_size(self) -> int:
        """
        Get the size of the deduplication group of the optimizer state dict.
        """
        return self.compute_config.optimizer_dedup_group_size

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
            None
        Raises:
            RuntimeError: if strict=True and there are missing keys.
        """

        dist2param = self.dist_param_map
        orig_param_names = list(dist2param.values())  # param names in original module (without prefix)

        with torch.no_grad():
            attr_names = set(self._fullmap.keys())

            origname_tid_map = {meta.orig_name: meta.tid for meta in self._fullmap.values()}
            tid_info = defaultdict(list)
            for attr, meta in self._fullmap.items():
                tid_info[meta.tid].append((attr, meta.slicers, meta.val_chunks))  # multiple params may share the same tid

            for orig_param_name in orig_param_names:
                orig_param_name_with_prefix = prefix + orig_param_name
                if orig_param_name_with_prefix not in state_dict:
                    continue
                param_value = state_dict[orig_param_name_with_prefix]
                tid = origname_tid_map[orig_param_name]
                for attr, slicer, nchunks in tid_info[tid]:
                    tensor: torch.Tensor = getattr(self, attr)
                    content = param_value[slicer]
                    if nchunks != 1:
                        content = content / nchunks
                    tensor.copy_(content)
                    attr_names.remove(attr)

            if len(attr_names) != 0:
                erro_msg = f'Missing key(s) in state_dict: {[prefix + self._fullmap[attr].orig_name for attr in attr_names]}.'
                if strict:
                    raise RuntimeError(erro_msg)
                else:
                    _logger.warning(erro_msg)
