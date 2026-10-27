#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import math
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Union, TYPE_CHECKING, runtime_checkable

import torch

from nnscaler.utils import first, first_or, get_member_by_name
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
class _Entry:
    """One local attribute of a rank's parallel module holding part of an original tensor."""
    rank: int
    # attribute name in the generated module
    local_name: str
    # index among the module's parameters, consistent with `ZeroMetadata.model_idx2opt_idx` keys;
    # None for buffers
    param_index: Optional[int]
    meta: AttrMeta

    def in_model_states(self, model_states: List[Dict[str, Any]]) -> bool:
        """Whether the (possibly deduplicated) model states hold this attribute's tensor."""
        return self.rank < len(model_states) and self.local_name in model_states[self.rank]


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


def _flat_positions(index: Tuple[slice, ...], box_shape: List[int]) -> torch.Tensor:
    """Row-major positions of the sub-box `index` inside the flattened box of shape `box_shape`."""
    positions = torch.zeros([i.stop - i.start for i in index], dtype=torch.int64, device='cpu')
    stride = 1
    for dim in reversed(range(len(index))):
        shape = [1] * len(index)
        shape[dim] = index[dim].stop - index[dim].start
        positions += (torch.arange(index[dim].start, index[dim].stop, device='cpu') * stride).view(shape)
        stride *= box_shape[dim]
    return positions


def _gather_region(
    target: Tuple[slice, ...], shards: List[_Shard], flat: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Assemble `full[target]`, or `full[target].reshape(-1)[flat[0]:flat[1]]` when `flat` is given
    (a ZeRO-3 chunk of the target box), from the shards overlapping it without materializing `full`.
    """
    target_shape = [t.stop - t.start for t in target]
    # Treat every side as a range of its flattened box; a whole box is the range (0, numel).
    whole_target = flat is None
    flat = flat or (0, math.prod(target_shape))
    out = torch.empty(flat[1] - flat[0], dtype=shards[0].data.dtype, device='cpu')
    covered = torch.zeros(out.shape, dtype=torch.bool, device='cpu')  # elements written by some shard
    for shard in shards:
        box, box_shape = shard.box, [b.stop - b.start for b in shard.box]
        inter = [slice(max(t.start, b.start), min(t.stop, b.stop)) for t, b in zip(target, box)]
        if any(i.start >= i.stop for i in inter):
            continue
        # dst_index is offset relative to the target box
        # src_index is offset relative to the shard box
        dst_index = tuple(slice(i.start - t.start, i.stop - t.start) for i, t in zip(inter, target))
        src_index = tuple(slice(i.start - b.start, i.stop - b.start) for i, b in zip(inter, box))
        if whole_target and shard.flat is None:
            # two plain boxes: a strided copy of the intersection
            out.view(target_shape)[dst_index].copy_(shard.data[src_index])
            covered.view(target_shape)[dst_index] = True
            continue
        # The remaining paths work in the flattened boxes; a whole box is the range (0, numel).
        src_flat = shard.flat or (0, math.prod(box_shape))
        data = shard.data if shard.flat else shard.data.reshape(-1)  # a view for contiguous data
        if dst_index and all(i.stop - i.start == m == n for i, m, n in zip(inter[1:], target_shape[1:], box_shape[1:])):
            # The intersection is a run of leading-dim rows, contiguous in both flattened
            # spaces with `source position = target position + shift`.
            row = math.prod(target_shape[1:])
            # src_index and dst_index have been aligned
            # shard.data[src_index] corresponds to out[dst_index]
            # so when we compare positions in the flattened space, we need to account for the shift.
            shift = (src_index[0].start - dst_index[0].start) * row
            lo = max(dst_index[0].start * row, flat[0], src_flat[0] - shift)
            hi = min(dst_index[0].stop * row, flat[1], src_flat[1] - shift)
            if lo < hi:
                # lo/hi are index of full[target].reshape(-1) in the flattened space
                # so we need to subtract flat[0] to get the index relative to the flattened target box.
                out[lo - flat[0]:hi - flat[0]] = data[lo + shift - src_flat[0]:hi + shift - src_flat[0]]
                covered[lo - flat[0]:hi - flat[0]] = True
            continue
        # General case: positions of the intersection elements in both flattened boxes.
        if dst_index:
            # Keep only the leading-dim rows touched by both flattened ranges, so the position
            # grids scale with the requested chunk rather than with the whole intersection.
            # A flattened range covers whole rows except at its two ends, so the grids exceed
            # the chunk by at most two rows (each `prod(shape[1:])` elements). The mask below
            # still trims those partial end rows to the exact ranges.
            shift = src_index[0].start - dst_index[0].start  # source row = target row + shift
            row_t, row_s = math.prod(target_shape[1:]), math.prod(box_shape[1:])
            lo = max(dst_index[0].start, flat[0] // row_t, src_flat[0] // row_s - shift)
            hi = min(dst_index[0].stop, -(-flat[1] // row_t), -(-src_flat[1] // row_s) - shift)
            if lo >= hi:
                continue
            dst_index = (slice(lo, hi),) + dst_index[1:]
            src_index = (slice(lo + shift, hi + shift),) + src_index[1:]
        target_positions = _flat_positions(dst_index, target_shape)
        source_positions = _flat_positions(src_index, box_shape)
        mask = (target_positions >= flat[0]) & (target_positions < flat[1]) \
            & (source_positions >= src_flat[0]) & (source_positions < src_flat[1])
        positions = target_positions[mask] - flat[0]
        out[positions] = data[(source_positions[mask] - src_flat[0]).to(data.device)].to('cpu')
        covered[positions] = True

    if not covered.all():
        raise ValueError(
            f"Incomplete checkpoint shards: {int((~covered).sum())} of {out.numel()} elements "
            f"in region {target}{'' if whole_target else f' flat range {flat}'} are not covered by any shard.")
    return out.view(target_shape) if whole_target else out


@runtime_checkable
class LazyTensor(Protocol):
    """
    Stands in for one full tensor in a state dict: indexing with slicers, or `slice_tensor`,
    assembles only the requested region instead of materializing the full tensor.
    """

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the full tensor."""

    def slice_tensor(self, slicers: Tuple[slice, ...], flat: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """`full[slicers]`, or `full[slicers].reshape(-1)[flat[0]:flat[1]]` when `flat` is given."""

    def __getitem__(self, slicers: Tuple[slice, ...]) -> torch.Tensor: ...


def slice_tensor(value: Union[torch.Tensor, LazyTensor], slicers: Tuple[slice, ...], flat: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    `value[slicers]`, or `value[slicers].reshape(-1)[flat[0]:flat[1]]` when `flat` is given,
    for a tensor or a `LazyTensor`, which assembles only that region.
    """
    if isinstance(value, LazyTensor):
        return value.slice_tensor(slicers, flat)
    value = value[slicers]
    return value if flat is None else value.reshape(-1)[flat[0]:flat[1]]


class _LazyRegion:
    """A `LazyTensor` backed by a region reader of a checkpoint."""

    def __init__(
        self,
        read_region: Callable[[Tuple[slice, ...], Optional[Tuple[int, int]]], torch.Tensor],
        dtype: torch.dtype,
        shape: Tuple[int, ...],
    ):
        self.read_region = read_region
        self.dtype = dtype
        self.shape = tuple(shape)

    def slice_tensor(self, slicers: Tuple[slice, ...], flat: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        return self.read_region(slicers, flat)

    def __getitem__(self, slicers: Tuple[slice, ...]) -> torch.Tensor:
        return self.slice_tensor(slicers)


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
        # original name -> its local attributes on every rank
        self.entries: Dict[str, List[_Entry]] = defaultdict(list)
        for rank, extra in enumerate(extra_states):
            param_index = 0
            for name, meta in extra.param_area_map.items():
                self.entries[meta.orig_name].append(
                    _Entry(rank, name, param_index if meta.is_param else None, meta)
                )
                param_index += int(meta.is_param)

        # Name mapping
        # by default, the name is not changed
        # but if there are shared parameter names, we map them to the canonical names
        self.model_names = {
            name: name for name, entries in self.entries.items()
            if any(entry.in_model_states(model_states) for entry in entries)
        }
        for names in extra_states[0].origin_shared_param_names:
            canonical = first_or(names, lambda name: name in self.model_names, default=None)
            if canonical is not None:
                for name in names:
                    self.model_names.setdefault(name, canonical)

    def read_model(self, name: str) -> _LazyRegion:
        name = self.model_names[name]
        # `model_names` only lists names with at least one shard present, so this cannot fail
        entry = first(self.entries[name], lambda entry: entry.in_model_states(self.model_states))
        return _LazyRegion(
            partial(self._read_model_region, name),
            self.model_states[entry.rank][entry.local_name].dtype, self.entries[name][0].meta.shape)

    def _read_model_region(self, name: str, target: Tuple[slice, ...], flat: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        shards: Dict[Tuple, _Shard] = {}  # deduplicate replicas by (slicers, flat range)
        for entry in self.entries[name]:
            if not entry.in_model_states(self.model_states):
                continue
            rank, local_name, meta = entry.rank, entry.local_name, entry.meta
            if meta.val_chunks != 1:
                raise NotImplementedError("Value-partitioned parameters are not supported.")
            data = self.model_states[rank][local_name]
            shard_flat = None
            zero3 = (self.extra_states[rank].zero3_param_metadata or {}).get(local_name)
            if zero3 is not None and tuple(data.shape) != tuple(meta.sub_shape):
                shard_flat = (int(zero3.start), int(zero3.end))
            shards.setdefault((_track_id(meta.slicers), shard_flat), _Shard(meta.slicers, shard_flat, data))
        if not shards:
            raise ValueError(
                f"No checkpoint shard found for {name}: entries "
                f"{[(entry.rank, entry.local_name) for entry in self.entries[name]]}, "
                f"model states of {len(self.model_states)} ranks with keys "
                f"{[sorted(state)[:6] for state in self.model_states]}")
        return _gather_region(target, list(shards.values()), flat)

    def _optimizer_sources(self, name: str) -> List[_OptimizerSource]:
        """Optimizer states holding parts of `name`, one per distinct shard."""
        if self.optimizer_states is None:
            return []
        sources, seen = [], set()
        for entry in self.entries[name]:
            rank, index, meta = entry.rank, entry.param_index, entry.meta
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
            key: _LazyRegion(
                partial(
                    self._read_optimizer_region, sources, key),
                    sources[0].state[key].dtype,
                    self.entries[name][0].meta.shape
                )
            for key in sources[0].state if key != 'step'
        }
        if 'step' in sources[0].state:
            step = sources[0].state['step']
            # Adam keeps step on CPU, where optimizer.load_state_dict may retain its storage.
            state['step'] = step.clone() if isinstance(step, torch.Tensor) else step
        return state

    @staticmethod
    def _read_optimizer_region(
        sources: List[_OptimizerSource], key: str, target: Tuple[slice, ...], flat: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return _gather_region(target, [source.shard(key) for source in sources], flat)


def _non_parallel_optimizer_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A non-parallel parameter's optimizer state, taken from the checkpoint by reference like
    merge_state_dicts does.

    `optimizer.load_state_dict` moves state tensors to the parameter's device, which copies them
    for GPU training, but keeps `step` on CPU as it is. Without a clone the optimizer would then
    share `step` with the checkpoint dict and its in-place `step += 1` would modify the caller's
    input. The other tensors are not cloned: they only alias the input for CPU training, which is
    not supported, and cloning them here would copy every state on each read.
    """
    return {
        key: value.clone() if key == 'step' and isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


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
        if optimizer is None:
            raise ValueError("The target optimizer must be provided.")
        if target_opt_extra is None:
            raise ValueError("The target optimizer must be created by build_optimizer.")
        if not _is_supported_optimizer(target_opt_extra.name):
            raise ValueError("Only Adam-like or Muon-like optimizers are supported.")
        if type(optimizer).__name__ != target_opt_extra.name:
            raise ValueError(f"The target optimizer must be of type {target_opt_extra.name}.")

    module_state_dicts = _sort_state_dicts(module_state_dicts)
    extra_states, model_states, non_parallel_states = _get_parallel_module_state_dict_info(module_state_dicts)
    opt_extra_states, opt_states, opt_metadata = None, {}, None
    if optimizer_state_dicts is not None:
        optimizer_state_dicts = _sort_state_dicts(optimizer_state_dicts)
        opt_extra_states, opt_states, opt_metadata = _get_optimizer_state_dict_info(optimizer_state_dicts)
        # HybridOptimizer packs its children into one group in the state dict, as merge_state_dicts expects.
        if len(opt_metadata['param_groups']) != 1:
            raise ValueError("Only a single optimizer parameter group is supported.")

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
                    optimizer_getters[name] = partial(_non_parallel_optimizer_state, optimizer_state_dicts[0]['state'][index])

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
    # a submodule registered under several names needs state under every prefix
    for prefix, child in module.named_modules(remove_duplicate=False):
        prefix = prefix + '.' if prefix else ''
        if isinstance(child, ParallelModule):
            local_model_state.update(child.trim_merged_state_dict(model_reader, prefix, device='cpu'))
            child._add_extra_state(local_model_state, prefix)
        else:
            own_state = {}
            child._save_to_state_dict(own_state, prefix, False)
            for name, _ in own_state.items():
                if name in model_reader:
                    value = model_reader[name]
                    if isinstance(value, LazyTensor):
                        # a parallel module's parameter: take the whole source tensor, so that
                        # load_state_dict still reports a shape mismatch with this target
                        value = value[tuple(slice(0, n) for n in value.shape)]
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

    # The remaining indices belong to the trainable non-parallel parameters outside ZeRO buckets,
    # which build_optimizer hands to the optimizer in registration order. Derive them from the
    # module rather than `optimizer.param_groups`, whose entries may be wrapped (e.g. the FP32
    # copies of MixedPrecisionAdam or the concatenated child groups of HybridOptimizer).
    parallel_params = {
        id(param) for child in module.modules() if isinstance(child, ParallelModule) for param in child.parameters()
    }
    non_parallel = [
        (name, param) for name, param in module.named_parameters()
        if param.requires_grad and id(param) not in parallel_params
    ] if _NON_PARALLEL_MODULE_ATTR_NAME not in target_opt_extra.parallel_module_locs else []
    total = len(parallel_indices) + len(non_parallel)
    non_parallel_indices = [index for index in range(total) if index not in parallel_indices]
    if len(non_parallel_indices) != len(non_parallel):  # locs reach beyond the parameters handed to the optimizer
        raise ValueError("The optimizer's parameter layout does not match the module.")
    for index, (name, _) in zip(non_parallel_indices, non_parallel):
        if name in optimizer_getters:
            local_optimizer_state[index] = {
                key: value[tuple(slice(0, n) for n in value.shape)] if isinstance(value, LazyTensor) else value
                for key, value in optimizer_getters[name]().items()
            }
    opt_metadata['state'] = local_optimizer_state
    opt_metadata['param_groups'][0]['params'] = list(range(total))
    optimizer.load_state_dict(opt_metadata)
