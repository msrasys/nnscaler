#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import math
from dataclasses import dataclass
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


@dataclass
class _Shard:
    """
    One shard of an original tensor: full tensor -> box (TP) -> optional flat range
    (ZeRO-3 model shard or ZeRO optimizer bucket chunk) -> data, i.e.
    ``full[box].reshape(-1)[flat[0]:flat[1]] == data``, or ``full[box] == data`` when `flat` is None.
    """
    # the region of the full tensor covered by the TP shard, one slice per dimension
    box: Tuple[slice, ...]
    # the range of the flattened box this shard holds; None means the whole box
    flat: Optional[Tuple[int, int]]
    # box shaped when `flat` is None, otherwise 1D holding at least `flat[1] - flat[0]` elements
    data: torch.Tensor


@dataclass
class _OptimizerSource:
    """
    An optimizer state holding part of a parameter, i.e. for each state key
    ``full[box].reshape(-1)[flat[0]:flat[1]] == state[key][data_slice[0]:data_slice[1]]``.
    """
    # the region of the full parameter covered by the TP shard, one slice per dimension
    box: Tuple[slice, ...]
    # the range of the flattened box held by this source; None means the whole box
    flat: Optional[Tuple[int, int]]
    # the rank's optimizer state for this parameter, or for the whole bucket containing it
    state: Dict[str, Any]
    # where `flat` lives inside each state tensor of a bucket; None when the state tensors are box shaped
    data_slice: Optional[Tuple[int, int]]

    def shard(self, key: str) -> _Shard:
        data = self.state[key]
        if self.data_slice is not None:
            data = data[self.data_slice[0]:self.data_slice[1]]
        return _Shard(self.box, self.flat, data)


def _gather_region(target: Tuple[slice, ...], shards: List[_Shard]) -> torch.Tensor:
    """Assemble `full[target]` from the shards overlapping it without materializing `full`."""
    out = torch.empty([t.stop - t.start for t in target], dtype=shards[0].data.dtype, device='cpu')
    # elements written by some shard
    # TODO: Consider using a more efficient data structure for covered regions if needed.
    covered = torch.zeros(out.shape, dtype=torch.bool, device='cpu')
    for shard in shards:
        box, flat, data = shard.box, shard.flat, shard.data
        inter = [slice(max(t.start, b.start), min(t.stop, b.stop)) for t, b in zip(target, box)]
        if any(i.start >= i.stop for i in inter):
            continue
        dst_index = tuple(slice(i.start - t.start, i.stop - t.start) for i, t in zip(inter, target))
        dst, dst_covered = out[dst_index], covered[dst_index]
        src_index = [slice(i.start - b.start, i.stop - b.start) for i, b in zip(inter, box)]
        if flat is None:
            dst.copy_(data[tuple(src_index)])
            dst_covered.fill_(True)
            continue
        # shard = full[box].reshape(-1)[start:end]
        start, end = flat
        box_shape = [b.stop - b.start for b in box]
        if src_index and dst.is_contiguous() and all(r.stop - r.start == n for r, n in zip(src_index[1:], box_shape[1:])):
            # The intersection is a run of leading-dim rows, contiguous in the flattened box.
            row = math.prod(box_shape[1:])
            # lo/hi: the range in data
            # src_index[0].start/stop: the unflattened range (superset of data)
            lo, hi = max(src_index[0].start * row, start), min(src_index[0].stop * row, end)
            if lo < hi:
                run = slice(lo - src_index[0].start * row, hi - src_index[0].start * row)
                dst.view(-1)[run] = data[lo - start:hi - start]
                dst_covered.view(-1)[run] = True
            continue

        # General case: positions of the intersection inside the flattened box.
        # idx will hold the flattened indices of the intersection within the flattened box.
        idx = torch.zeros([r.stop - r.start for r in src_index], dtype=torch.int64, device='cpu')
        stride = 1
        for dim in reversed(range(len(src_index))):
            shape = [1] * len(src_index)
            shape[dim] = src_index[dim].stop - src_index[dim].start
            idx += (torch.arange(src_index[dim].start, src_index[dim].stop, device='cpu') * stride).view(shape)
            stride *= box_shape[dim]
        mask = (idx >= start) & (idx < end)
        dst[mask] = data[(idx[mask] - start).to(data.device)].to(dst.device)
        dst_covered[mask] = True

    if not covered.all():
        raise ValueError(
            f"Incomplete checkpoint shards: {int((~covered).sum())} of {out.numel()} elements "
            f"in region {target} are not covered by any shard.")
    return out


class _LazyRegion:
    """Stands in for one full tensor: indexing with slicers assembles only that region."""

    def __init__(self, read_region: Callable[[Tuple[slice, ...]], torch.Tensor]):
        self.read_region = read_region

    def __getitem__(self, slicers: Tuple[slice, ...]) -> torch.Tensor:
        return self.read_region(slicers)


class _ModuleStateReader:
    """Assemble regions of original tensors from a parallel module's per-rank checkpoint."""

    def __init__(
        self,
        extra_states: List[ExtraState],
        model_states: List[Dict[str, Any]],
        optimizer_states: Optional[List[Dict[str, Any]]] = None,
    ):
        self.extra_states = extra_states
        self.model_states = model_states
        self.optimizer_states = optimizer_states
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

    def read_model(self, name: str) -> _LazyRegion:
        return _LazyRegion(partial(self._read_model_region, self.model_names[name]))

    def _read_model_region(self, name: str, target: Tuple[slice, ...]) -> torch.Tensor:
        shards: Dict[Tuple, _Shard] = {}  # deduplicate replicas by (slicers, flat range)
        for rank, local_name, _, meta in self.entries[name]:
            if rank >= len(self.model_states) or local_name not in self.model_states[rank]:
                continue
            if meta.val_chunks != 1:
                raise NotImplementedError("Value-partitioned parameters are not supported.")
            data = self.model_states[rank][local_name]
            flat = None
            zero3 = (self.extra_states[rank].zero3_param_metadata or {}).get(local_name)
            if zero3 is not None and tuple(data.shape) != tuple(meta.sub_shape):
                flat = (int(zero3.start), int(zero3.end))
            shards.setdefault((_track_id(meta.slicers), flat), _Shard(meta.slicers, flat, data))
        if not shards:
            raise ValueError(
                f"No checkpoint shard found for {name}: entries "
                f"{[(rank, local_name) for rank, local_name, _, _ in self.entries[name]]}, "
                f"model states of {len(self.model_states)} ranks with keys "
                f"{[sorted(state)[:6] for state in self.model_states]}")
        return _gather_region(target, list(shards.values()))

    def _optimizer_sources(self, name: str) -> List[_OptimizerSource]:
        """Optimizer states holding parts of `name`, one per distinct shard."""
        if self.optimizer_states is None:
            return []
        sources, seen = [], set()
        for rank, _, index, meta in self.entries[name]:
            track = _track_id(meta.slicers)
            if index is None or rank >= len(self.optimizer_states) or track in seen:
                continue
            seen.add(track)
            extra = self.extra_states[rank]
            opt_idx = extra.model_idx2opt_idx[index] if extra.zero else index
            if isinstance(opt_idx, int):
                state = self.optimizer_states[rank]['state'].get(opt_idx)
                if state:
                    sources.append(_OptimizerSource(meta.slicers, None, state, None))
                continue
            # The parameter lives in [pstart, pend) of a reducer bucket that is spread over `ranks`.
            opt_idx, pstart, pend, _ = opt_idx
            ranks, bucket_size = extra.opt_idx2ranks[opt_idx]
            slice_len, total = pend - pstart, math.prod(meta.sub_shape)
            for i, r in enumerate(ranks):
                state = self.optimizer_states[r]['state'].get(opt_idx)
                if not state:
                    continue
                if extra.zero == 1:
                    # Each rank holds one equal chunk of the whole bucket.
                    chunk = bucket_size // len(ranks)
                    lo, hi = max(pstart, i * chunk), min(pend, (i + 1) * chunk)
                    if lo < hi:
                        # full_tensor[meta.slicers].view(-1)[lo - pstart:hi - pstart]
                        #  = state[(lo - i * chunk):(hi - i * chunk)]
                        sources.append(_OptimizerSource(meta.slicers, (lo - pstart, hi - pstart), state, (lo - i * chunk, hi - i * chunk)))
                else:
                    # ZeRO-3: each rank holds the i-th slice of length `slice_len` of the parameter.
                    lo = i * slice_len
                    if lo >= total:
                        break
                    hi = min(lo + slice_len, total)
                    # full_tensor[meta.slicers].view(-1)[lo:hi]
                    #  = state[pstart:pstart + hi - lo]
                    sources.append(_OptimizerSource(meta.slicers, (lo, hi), state, (pstart, pstart + hi - lo)))
        return sources

    def has_optimizer(self, name: str) -> bool:
        return bool(self._optimizer_sources(name))

    def read_optimizer(self, name: str) -> Dict[str, Any]:
        sources = self._optimizer_sources(name)
        state = {
            key: _LazyRegion(partial(self._read_optimizer_region, sources, key))
            for key in sources[0].state if key != 'step'
        }
        if 'step' in sources[0].state:
            step = sources[0].state['step']
            # Adam keeps step on CPU, where optimizer.load_state_dict may retain its storage.
            state['step'] = step.clone() if isinstance(step, torch.Tensor) else step
        return state

    @staticmethod
    def _read_optimizer_region(sources: List[_OptimizerSource], key: str, target: Tuple[slice, ...]) -> torch.Tensor:
        return _gather_region(target, [source.shard(key) for source in sources])


def _track_id(slicers: Tuple[slice, ...]) -> Tuple[Tuple[Any, Any, Any], ...]:
    # slice objects are not hashable before Python 3.12
    return tuple((s.start, s.step, s.stop) for s in slicers)


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
    Only the regions of original parameters needed by the target rank are assembled,
    one at a time, on CPU. Target-local model and optimizer state dicts are then loaded using
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
            local_model_state.update(child.trim_merged_state_dict(model_reader, prefix, device='cpu'))
            child._add_extra_state(local_model_state, prefix)
        else:
            own_state = {}
            child._save_to_state_dict(own_state, prefix, False)
            for name, tensor in own_state.items():
                if name in model_reader:
                    value = model_reader[name]
                    if isinstance(value, _LazyRegion):  # a parallel module's parameter, take it whole
                        value = value[tuple(slice(0, n) for n in tensor.shape)]
                    local_model_state[name] = value
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
            whole = tuple(slice(0, n) for n in param.shape)
            local_optimizer_state[index] = {
                key: value[whole] if isinstance(value, _LazyRegion) else value
                for key, value in optimizer_getters[param_names[param]]().items()
            }
    opt_metadata['state'] = local_optimizer_state
    opt_metadata['param_groups'][0]['params'] = list(range(len(optimizer.param_groups[0]['params'])))
    optimizer.load_state_dict(opt_metadata)
