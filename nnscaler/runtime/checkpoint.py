#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING

import torch

from nnscaler.utils import first_or, get_member_by_name
from nnscaler.runtime.module import AttrMeta, ExtraState, ParallelModule

if TYPE_CHECKING:
    from nnscaler.parallel import ParallelOptimizer


class _LazyStateDict(Mapping[str, Any]):
    """Read values on demand without retaining reconstructed full tensors."""

    def __init__(self, getters: Dict[str, Callable[[], Any]]):
        self.getters = getters

    def __getitem__(self, key: str) -> Any:
        return self.getters[key]()

    def __iter__(self) -> Iterator[str]:
        return iter(self.getters)

    def __len__(self) -> int:
        return len(self.getters)

    def __contains__(self, key: object) -> bool:
        return key in self.getters


class _ModuleStateReader:
    """Reconstruct one original attribute from a parallel module's checkpoint."""

    def __init__(
        self,
        extra_states: List[ExtraState],
        model_states: List[Dict[str, Any]],
        optimizer_states: Optional[List[Dict[str, Any]]] = None,
    ):
        self.extra_states = extra_states
        self.model_states = model_states
        self.optimizer_states = optimizer_states
        # Entries contain model-to-optimizer indices, bucket ranks, ZeRO version,
        # and per-parameter ZeRO-3 intervals.
        self.zero_maps = [
            (extra.model_idx2opt_idx, extra.opt_idx2ranks, extra.zero, extra.zero3_param_metadata)
            for extra in extra_states
        ] if any(extra.zero for extra in extra_states) else None
        # Entries contain rank, local attribute name, model parameter index, and metadata.
        # Note here param_index is consistent with `model_idx2opt_idx.keys()`
        self.entries: Dict[str, List[Tuple[int, str, Optional[int], AttrMeta]]] = defaultdict(list)
        for rank, extra in enumerate(extra_states):
            param_index = 0
            for name, meta in extra.param_area_map.items():
                self.entries[meta.orig_name].append(
                    (rank, name, param_index if meta.is_param else None, meta)
                )
                param_index += int(meta.is_param)

        # Name mapping
        # by default, the name is not changed
        # but if there are shared parameter names, we map them to the canonical names
        self.model_names = {
            name: name for name, entries in self.entries.items()
            if any(rank < len(model_states) and local_name in model_states[rank]
                   for rank, local_name, _, _ in entries)
        }
        for names in extra_states[0].origin_shared_param_names:
            canonical = first_or(names, lambda name: name in self.model_names, default=None)
            if canonical is not None:
                for name in names:
                    self.model_names.setdefault(name, canonical)

    def read_model(self, name: str) -> torch.Tensor:
        name = self.model_names[name]
        fullmaps = [{} for _ in self.model_states]
        states = [{} for _ in self.model_states]
        for rank, local_name, _, meta in self.entries[name]:
            if rank < len(states) and local_name in self.model_states[rank]:
                fullmaps[rank][local_name] = meta
                states[rank][local_name] = self.model_states[rank][local_name]
        return ParallelModule.merge_model_state_dicts(states, fullmaps, self.zero_maps)[name]

    def has_optimizer(self, name: str) -> bool:
        if self.optimizer_states is None:
            return False
        for rank, _, index, _ in self.entries[name]:
            if index is None or rank >= len(self.optimizer_states):
                continue
            extra = self.extra_states[rank]
            opt_index = extra.model_idx2opt_idx[index] if extra.zero else index
            if isinstance(opt_index, tuple):
                opt_index = opt_index[0]
            if self.optimizer_states[rank]['state'].get(opt_index):
                return True
        return False

    def read_optimizer(self, name: str) -> Dict[str, Any]:
        assert self.optimizer_states is not None
        fullmaps = [{} for _ in self.extra_states]
        # value: a list of parameter indices for each rank
        # Note: there is only one param index per rank for each parameter
        #       in most cases.(see `CubeModule._fullmap`)
        indices = [[] for _ in self.extra_states]
        for rank, local_name, param_index, meta in self.entries[name]:
            if param_index is not None:
                indices[rank].append(param_index)
                fullmaps[rank][local_name] = meta

        zero_maps = None
        if self.zero_maps is not None:
            # Filtering fullmaps renumbers model indices, but bucket indices stay unchanged.
            zero_maps = [
                (
                    {new: extra.model_idx2opt_idx[old] for new, old in enumerate(rank_indices)},
                    extra.opt_idx2ranks, extra.zero, extra.zero3_param_metadata,
                )
                for extra, rank_indices in zip(self.extra_states, indices)
            ]
            # The merge helper looks up `state[opt_idx]` with the original bucket indices, so keep
            # each rank's full state dict (by reference, nothing is copied). Only the requested
            # parameter's [pstart:pend] range of its bucket is read and materialized.
            states = [{'state': state['state']} for state in self.optimizer_states]
        else:
            states = [
                {'state': {
                    new: state['state'][old] for new, old in enumerate(rank_indices)
                    if old in state['state']
                }}
                for state, rank_indices in zip(self.optimizer_states, indices)
            ]
        # Do not pass param_groups: the merge helper rewrites their parameter indices.
        state = ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)['state'][0]
        # Adam keeps step on CPU, where optimizer.load_state_dict may retain its storage.
        if isinstance(state.get('step'), torch.Tensor):
            state['step'] = state['step'].clone()
        return state


@torch.no_grad()
def load_from_rank_state_dicts(
    module: torch.nn.Module,
    module_state_dicts: List[Dict[str, Any]],
    optimizer: Optional[Union[torch.optim.Optimizer, 'ParallelOptimizer']] = None,
    optimizer_state_dicts: Optional[List[Dict[str, Any]]] = None,
    *,
    device: Union[str, torch.device] = None,
    strict: bool = True,
) -> None:
    """
    Load per-rank checkpoints without constructing full merged state dicts.

    All source checkpoint ranks must be accessible in this process, including their
    nnscaler metadata. Tensor payloads may be deduplicated as in merge_state_dicts.
    Source and target TP, PP, ZeRO and bucket layouts may differ.
    Only original parameters needed by the target rank are reconstructed, one at a
    time, on CPU. Target-local model and optimizer state dicts are then loaded using
    their standard load_state_dict methods. No checkpoint communication is performed.

    Args:
        module: Target module.
        module_state_dicts: Model state dicts from the source checkpoint ranks.
        optimizer: Target optimizer created by build_optimizer, if loading its state.
        optimizer_state_dicts: Optimizer state dicts from the same source checkpoint.
            Only single-group Adam/Muon-like optimizers are supported, as in merge_state_dicts.
        device: Target device; defaults to torch.cuda.current_device().
        strict: Require all target model parameters and persistent buffers. Source
            keys not needed by the target rank are ignored.
    """
    # nnscaler.parallel imports this module, so its helpers are imported lazily here.
    # TODO: will move all checkpoint related functions from nnscaler.parallel to this module in the future.
    from nnscaler.parallel import (
        _NON_PARALLEL_MODULE_ATTR_NAME, OptimizerExtraState, _construct_optim_state,
        _get_optimizer_state_dict_info, _get_parallel_module_state_dict_info,
        _is_supported_optimizer, _sort_state_dicts,
    )

    if not module_state_dicts:
        raise ValueError("module_state_dicts should not be empty.")
    target_opt_extra: Optional[OptimizerExtraState] = getattr(optimizer, '_extra_state', None)
    if optimizer_state_dicts is not None:
        if target_opt_extra is None:
            raise ValueError("The target optimizer must be created by build_optimizer.")
        if not _is_supported_optimizer(target_opt_extra.name):
            raise ValueError("Only Adam-like or Muon-like optimizers are supported.")
        if not optimizer:
            raise ValueError("The target optimizer must be provided.")
        if type(optimizer).__name__ != target_opt_extra.name:
            raise ValueError(f"The target optimizer must be of type {target_opt_extra.name}.")
        if len(optimizer.param_groups) != 1:
            raise ValueError("Only a single optimizer parameter group is supported.")

    module_state_dicts = _sort_state_dicts(module_state_dicts)
    extra_states, model_states, non_parallel_states = _get_parallel_module_state_dict_info(module_state_dicts)
    opt_extra_states, opt_states, opt_metadata = None, {}, None
    if optimizer_state_dicts is not None:
        optimizer_state_dicts = _sort_state_dicts(optimizer_state_dicts)
        opt_extra_states, opt_states, opt_metadata = _get_optimizer_state_dict_info(optimizer_state_dicts)

    # non-parallel modules
    model_getters = {
        name: partial(non_parallel_states.__getitem__, name) for name in non_parallel_states
    }
    optimizer_getters = {}
    if opt_extra_states is not None:
        if _NON_PARALLEL_MODULE_ATTR_NAME in opt_states:  # zero 1
            reader = _ModuleStateReader(
                [extra.non_parallel_extra_state for extra in opt_extra_states],
                [], opt_states[_NON_PARALLEL_MODULE_ATTR_NAME],
            )
            for name in reader.entries:
                if reader.has_optimizer(name):
                    optimizer_getters[name] = partial(reader.read_optimizer, name)
        else: # no zero
            # Non-parallel parameters outside ZeRO buckets retain their registration order.
            # we will read the non-parallel optimizer state from rank 0
            target_param_names = {name for name, param in module.named_parameters() if param.requires_grad}
            non_parallel_names = [name for name in non_parallel_states if name in target_param_names]
            locations = opt_extra_states[0].parallel_module_locs.values()
            parallel_indices = {index for loc in locations for index in range(loc.offset, loc.offset + loc.count)}
            source_indices = [
                index for index in optimizer_state_dicts[0]['param_groups'][0]['params']
                if index not in parallel_indices
            ]
            if len(non_parallel_names) != len(source_indices):
                raise ValueError("Cannot match non-parallel optimizer parameters to the target model.")
            for name, index in zip(non_parallel_names, source_indices):
                if optimizer_state_dicts[0]['state'].get(index):
                    optimizer_getters[name] = partial(copy.deepcopy, optimizer_state_dicts[0]['state'][index])

    # parallel modules
    for module_path, extras in extra_states.items():
        if any(extra is None for extra in extras):
            raise ValueError(f"Missing checkpoint metadata for parallel module {'.'.join(module_path)}.")
        prefix = '.'.join(module_path)
        reader = _ModuleStateReader(extras, model_states.get(module_path, []), opt_states.get(prefix))
        prefix = prefix + '.' if prefix else ''
        for name in reader.model_names:
            model_getters[prefix + name] = partial(reader.read_model, name)
        for name in reader.entries:
            if reader.has_optimizer(name):
                optimizer_getters[prefix + name] = partial(reader.read_optimizer, name)

    device = device if device is not None else torch.cuda.current_device()
    module.to(device)
    model_reader = _LazyStateDict(model_getters)
    local_model_state = {}
    for prefix, child in module.named_modules():
        prefix = prefix + '.' if prefix else ''
        if isinstance(child, ParallelModule):
            local_model_state.update(child.trim_merged_state_dict(model_reader, prefix, device='cpu', copy=True))
            child._add_extra_state(local_model_state, prefix)
        else:
            own_state = {}
            child._save_to_state_dict(own_state, prefix, False)
            for name in own_state:
                if name in model_reader:
                    local_model_state[name] = model_reader[name]
    module.load_state_dict(local_model_state, strict=strict)
    del local_model_state

    if optimizer_state_dicts is None:
        return

    local_optimizer_state = {}
    # `parallel_module_locs` offsets index `optimizer.param_groups[0]['params']`, the parameter
    # sequence handed to the optimizer by build_optimizer. Under ZeRO that sequence holds the
    # reducers' flat bucket parameters instead of the module's own parameters.
    parallel_indices = set()
    for prefix, loc in target_opt_extra.parallel_module_locs.items():
        child = get_member_by_name(module, prefix)
        parallel_indices.update(range(loc.offset, loc.offset + loc.count))
        full_prefix = prefix + '.' if prefix and prefix != _NON_PARALLEL_MODULE_ATTR_NAME else ''
        orig_states = _LazyStateDict({
            name: optimizer_getters[full_prefix + name]
            for name in child.origin_module_metadata.origin_param_names
            if full_prefix + name in optimizer_getters
        })
        states = _construct_optim_state(child, orig_states)
        local_optimizer_state.update({loc.offset + index: state for index, state in states.items()})

    # Remaining indices are non-parallel parameters outside ZeRO buckets. Check the index first:
    # flat bucket parameters inside `parallel_indices` are not in `module.named_parameters()`.
    param_names = {param: name for name, param in module.named_parameters()}
    for index, param in enumerate(optimizer.param_groups[0]['params']):
        if index not in parallel_indices and param_names[param] in optimizer_getters:
            local_optimizer_state[index] = optimizer_getters[param_names[param]]()
    opt_metadata['state'] = local_optimizer_state
    opt_metadata['param_groups'][0]['params'] = list(range(len(optimizer.param_groups[0]['params'])))
    optimizer.load_state_dict(opt_metadata)
