"""
MoE FWD-BWD overlap scheduling engine.

Provides ScheduleNode (CUDA stream/event scheduling primitive) and
MergedScheduler (merged forward-backward with communication/computation overlap).

4-Phase Overlap (MoE layers):
  Phase 1: f_attn_router(COMP) || b_combine(COMM)
  Phase 2: b_expert (COMP) || f_dispatch   (COMM)
  Phase 3: f_expert (COMP) || b_dispatch(COMM)
  Phase 4: b_attn   (COMP) || f_combine    (COMM)

2-Phase Overlap (dense layers):
  Phase 1: body_bwd(prev)[COMP] || attn_fwd(next)[COMP]
  Phase 2: attn_bwd(prev)[COMP] || body_fwd(next)[COMP]

MoE layers use 4 ScheduleNodes (attn_router, dispatch, expert, combine)
alternating COMP/COMM streams for true communication/computation overlap.
Dense layers use 2 ScheduleNodes (attn, ffn) both on COMP stream.
"""

import logging
import os
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.autograd import Variable

import nnscaler


_logger = logging.getLogger(__name__)

_COMP_STREAM = None   # computation stream (non-default)
_COMM_STREAM = None   # communication stream (non-default)
_CREATION_STREAM_ATTR = '_nnscaler_creation_stream'
_MEM_HISTORY_STARTED = False
_MEM_THRESHOLD_DUMPS = 0


class TransformerLayerState:
    """Per-layer state shared by the four MoE schedule nodes."""

    pass


def _env_true(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ('0', 'false', 'no', '')


def _debug_rank():
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))


def _mem_debug_enabled():
    if not _env_true('NNSCALER_MEM_DEBUG', False):
        return False
    ranks = os.environ.get('NNSCALER_MEM_DEBUG_RANKS', '0')
    if ranks.lower() in ('all', '*'):
        return True
    return str(_debug_rank()) in {r.strip() for r in ranks.split(',')}


def _gb(value):
    return float(value) / 1024 / 1024 / 1024


def _cuda_storage_nbytes(obj):
    total = 0
    seen = set()
    for tensor in _iter_cuda_tensors(obj):
        key = _storage_key(tensor)
        if key is None or key in seen:
            continue
        seen.add(key)
        try:
            total += tensor.untyped_storage().nbytes()
        except Exception:
            pass
    return total


def _tensor_storage_byte_range(tensor):
    """Return the byte range of *tensor* inside its backing storage."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return None
    if tensor.numel() == 0:
        return (0, 0)
    try:
        start = tensor.storage_offset() * tensor.element_size()
        span = 1
        for size, stride in zip(tensor.size(), tensor.stride()):
            if size == 0:
                return (start, start)
            span += (size - 1) * abs(stride)
        end = start + span * tensor.element_size()
        nbytes = tensor.untyped_storage().nbytes()
    except Exception:
        return None
    return (max(0, start), min(end, nbytes))


def _covered_storage_bytes(ranges):
    merged = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _release_debug_log_enabled():
    if not _env_true('NNSCALER_RELEASE_DEBUG_LOG', False):
        return False
    ranks = os.environ.get('NNSCALER_RELEASE_DEBUG_LOG_RANKS', '0')
    if ranks.lower() in ('all', '*'):
        return True
    return str(_debug_rank()) in {r.strip() for r in ranks.split(',')}


def _release_debug_log(tag, message):
    if not _release_debug_log_enabled():
        return
    print(f"[RELEASE_DEBUG][rank={_debug_rank()}][{tag}] {message}", flush=True)


def _safe_tag(tag):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', tag).strip('_')


def _maybe_start_memory_history():
    global _MEM_HISTORY_STARTED
    if _MEM_HISTORY_STARTED:
        return
    if not (_mem_debug_enabled() or _env_true('NNSCALER_MEM_SNAPSHOT_ON_THRESHOLD', False)):
        return
    ranks = os.environ.get(
        'NNSCALER_MEM_HISTORY_RANKS',
        os.environ.get('NNSCALER_MEM_SNAPSHOT_RANKS', '0'))
    if ranks.lower() not in ('all', '*') and str(_debug_rank()) not in {r.strip() for r in ranks.split(',')}:
        return
    if not (
        _env_true('NNSCALER_MEM_HISTORY', False)
        or _env_true('NNSCALER_MEM_SNAPSHOT_HISTORY', False)
    ):
        return
    max_entries = int(os.environ.get('NNSCALER_MEM_HISTORY_MAX_ENTRIES', '200000'))
    torch.cuda.memory._record_memory_history(
        enabled='all', context='all', stacks='all', max_entries=max_entries)
    _MEM_HISTORY_STARTED = True
    print(f"[MEMDBG][rank={_debug_rank()}] memory history enabled max_entries={max_entries}", flush=True)


def _maybe_dump_threshold_snapshot(tag, reserved, active, allocated):
    global _MEM_THRESHOLD_DUMPS
    if not _env_true('NNSCALER_MEM_SNAPSHOT_ON_THRESHOLD', False):
        return
    ranks = os.environ.get('NNSCALER_MEM_SNAPSHOT_RANKS', '0')
    if ranks.lower() not in ('all', '*') and str(_debug_rank()) not in {r.strip() for r in ranks.split(',')}:
        return

    tag_filters = [
        item.strip() for item in os.environ.get(
            'NNSCALER_MEM_SNAPSHOT_TAG_SUBSTRINGS', '').split(',')
        if item.strip()
    ]
    if tag_filters and not any(item in tag for item in tag_filters):
        return

    min_reserved_gb = float(os.environ.get('NNSCALER_MEM_SNAPSHOT_MIN_RESERVED_GB', '0'))
    if _gb(reserved) < min_reserved_gb:
        return

    max_dumps = int(os.environ.get('NNSCALER_MEM_SNAPSHOT_MAX_DUMPS', '1'))
    if _MEM_THRESHOLD_DUMPS >= max_dumps:
        return

    dump_dir = os.environ.get('NNSCALER_MEM_SNAPSHOT_DIR')
    if not dump_dir:
        return

    _maybe_start_memory_history()
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    _MEM_THRESHOLD_DUMPS += 1
    path = pathlib.Path(dump_dir) / (
        f"rank{_debug_rank()}_dump{_MEM_THRESHOLD_DUMPS}_"
        f"{_safe_tag(tag)}_reserved{_gb(reserved):.1f}G.pickle")
    torch.cuda.memory._dump_snapshot(str(path))
    print(
        f"[MEMDBG][rank={_debug_rank()}][{tag}] threshold_snapshot={path} "
        f"allocated={_gb(allocated):.3f}G active={_gb(active):.3f}G "
        f"reserved={_gb(reserved):.3f}G",
        flush=True,
    )


def _mem_probe(tag, dump=False):
    if _env_true('NNSCALER_MEM_PROBE_TO_PHASE_LOG', False):
        _phase_mem_probe(f'mem:{tag}')

    if not _mem_debug_enabled() or not torch.cuda.is_available():
        return
    if _env_true('NNSCALER_MEM_DEBUG_SYNC', False):
        torch.cuda.synchronize()

    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    active = stats.get('active_bytes.all.current', 0)
    inactive_split = stats.get('inactive_split_bytes.all.current', 0)
    inactive = stats.get('inactive_bytes.all.current', 0)
    requested = stats.get('requested_bytes.all.current', 0)
    seg_current = stats.get('segment.all.current', 0)
    oversize_segments = stats.get('oversize_segments.current', 0)
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    print(
        f"[MEMDBG][rank={_debug_rank()}][{tag}] "
        f"alloc={_gb(allocated):.3f}G active={_gb(active):.3f}G "
        f"requested={_gb(requested):.3f}G reserved={_gb(reserved):.3f}G "
        f"inactive={_gb(inactive):.3f}G inactive_split={_gb(inactive_split):.3f}G "
        f"segments={seg_current} oversize_segments={oversize_segments} "
        f"max_alloc={_gb(max_allocated):.3f}G max_reserved={_gb(max_reserved):.3f}G",
        flush=True,
    )

    dump_dir = os.environ.get('NNSCALER_MEM_DEBUG_DUMP_DIR')
    dump_tags = {t.strip() for t in os.environ.get('NNSCALER_MEM_DEBUG_DUMP_TAGS', '').split(',') if t.strip()}
    if dump_dir and (dump or tag in dump_tags):
        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(dump_dir) / f"rank{_debug_rank()}_{_safe_tag(tag)}.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        print(f"[MEMDBG][rank={_debug_rank()}][{tag}] snapshot={path}", flush=True)

    _maybe_dump_threshold_snapshot(tag, reserved, active, allocated)


def _phase_mem_probe(tag):
    if not _env_true('NNSCALER_PHASE_MEM_LOG', False) or not torch.cuda.is_available():
        return
    ranks = os.environ.get('NNSCALER_PHASE_MEM_LOG_RANKS', '0')
    if ranks.lower() not in ('all', '*') and str(_debug_rank()) not in {r.strip() for r in ranks.split(',')}:
        return
    if _env_true('NNSCALER_PHASE_MEM_LOG_SYNC', False):
        torch.cuda.synchronize()

    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    active = stats.get('active_bytes.all.current', 0)
    requested = stats.get('requested_bytes.all.current', 0)
    allocated_stat = stats.get('allocated_bytes.all.current', 0)
    active_allocated = stats.get('active_bytes.all.allocated', 0)
    active_freed = stats.get('active_bytes.all.freed', 0)
    reserved_large = stats.get('reserved_bytes.large_pool.current', 0)
    active_large = stats.get('active_bytes.large_pool.current', 0)
    allocated_large = stats.get('allocated_bytes.large_pool.current', 0)
    requested_large = stats.get('requested_bytes.large_pool.current', 0)
    inactive_split = stats.get('inactive_split_bytes.all.current', 0)
    inactive = stats.get('inactive_bytes.all.current', 0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    pending_or_active_cache = max(active - allocated, 0)
    cached_reusable = max(reserved - active, 0)

    line = (
        f"[PHASE_MEM][rank={_debug_rank()}][{tag}] "
        f"alloc_live={_gb(allocated):.3f}G "
        f"alloc_stat={_gb(allocated_stat):.3f}G "
        f"active_allocator={_gb(active):.3f}G "
        f"pending_or_active_cache={_gb(pending_or_active_cache):.3f}G "
        f"requested={_gb(requested):.3f}G "
        f"reserved={_gb(reserved):.3f}G "
        f"cached_reusable={_gb(cached_reusable):.3f}G "
        f"active_delta={_gb(active_allocated - active_freed):.3f}G "
        f"large_alloc={_gb(allocated_large):.3f}G "
        f"large_active={_gb(active_large):.3f}G "
        f"large_requested={_gb(requested_large):.3f}G "
        f"large_reserved={_gb(reserved_large):.3f}G "
        f"inactive={_gb(inactive):.3f}G "
        f"inactive_split={_gb(inactive_split):.3f}G "
        f"cuda_free={_gb(free_bytes):.3f}G "
        f"cuda_total={_gb(total_bytes):.3f}G "
        f"max_alloc={_gb(torch.cuda.max_memory_allocated()):.3f}G "
        f"max_reserved={_gb(torch.cuda.max_memory_reserved()):.3f}G\n"
    )

    path = os.environ.get('NNSCALER_PHASE_MEM_LOG_FILE')
    if path:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
    else:
        print(line, end='', flush=True)

    _maybe_dump_threshold_snapshot(tag, reserved, active, allocated)


def set_streams():
    """Initialize global COMP/COMM streams.
    BOTH streams are non-default so that CUDA's legacy
    default-stream implicit sync does not serialise them.
    """
    global _COMP_STREAM, _COMM_STREAM
    if _env_true('NNSCALER_EP_OVERLAP_MEGATRON_STYLE_STREAMS', False):
        # Megatron's EP overlap uses the current/default stream for compute and
        # only creates a persistent side stream for communication.
        _COMP_STREAM = torch.cuda.current_stream()
    else:
        _COMP_STREAM = torch.cuda.Stream()
    _COMM_STREAM = torch.cuda.Stream()
    _maybe_start_memory_history()
    _mem_probe('set_streams')
    # Merged scheduling intentionally accumulates gradients on non-default
    # streams. Suppress this warning only when the merged scheduler streams
    # are initialized, rather than at module import time.
    if hasattr(torch.autograd.graph, 'set_warn_on_accumulate_grad_stream_mismatch'):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def get_comp_stream():
    assert _COMP_STREAM is not None, "call set_streams() first"
    return _COMP_STREAM


def get_comm_stream():
    assert _COMM_STREAM is not None, "call set_streams() first"
    return _COMM_STREAM


def manual_sync_grads(parallel_module):
    """Manually trigger synchronous allreduce after all backward calls.

    The merged scheduler uses skip_reducer=True, so hooks copy grads to the
    contiguous buffer but skip counting/triggering allreduce. This function
    performs the allreduce and sets param.grad from the buffer.
    """
    pm = parallel_module
    if hasattr(pm, 'backbone'):
        pm = pm.backbone
    if not hasattr(pm, '_reducers'):
        _logger.warning("No _reducers found on parallel module, skipping manual sync")
        return

    for reducer in pm._reducers:
        for bucket in reducer._buckets:
            old_async = bucket._async
            bucket._async = False
            try:
                bucket.sync_grads()
            finally:
                bucket._async = old_async
            bucket.reset()


def _kernel_make_viewless_tensor(tensor, requires_grad):
    """Create a viewless tensor sharing the same data, matching Megatron."""
    out = torch.empty(
        (1,), dtype=tensor.dtype, device=tensor.device,
        requires_grad=requires_grad)
    out.data = tensor.data
    return out


class _MakeViewlessTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, requires_grad):
        return _kernel_make_viewless_tensor(tensor, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _make_viewless(tensor, keep_graph=True):
    """Return a non-view tensor without cloning storage.

    Megatron uses this to avoid holding a view's base tensor alive.  Cloning
    would allocate new activation storage and defeats the memory-lifetime goal.
    """
    if not isinstance(tensor, torch.Tensor) or tensor._base is None:
        return tensor
    if keep_graph:
        return _MakeViewlessTensor.apply(tensor, tensor.requires_grad)
    return _kernel_make_viewless_tensor(tensor, tensor.requires_grad)


def _iter_cuda_tensors(obj):
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.is_cuda and id(value) not in seen:
                seen.add(id(value))
                yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from visit(item)

    yield from visit(obj)


def _stream_key(stream):
    return (stream.device, stream.cuda_stream)


def _same_stream(left, right):
    return _stream_key(left) == _stream_key(right)


def _tensor_creation_stream(tensor):
    stream = getattr(tensor, _CREATION_STREAM_ATTR, None)
    if stream is not None:
        return stream
    if tensor.is_cuda:
        return torch.cuda.default_stream(tensor.device)
    return None


def _set_tensor_creation_stream(tensor, stream):
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        try:
            setattr(tensor, _CREATION_STREAM_ATTR, stream)
        except Exception:
            pass


def _mark_creation_stream(obj, stream):
    for tensor in _iter_cuda_tensors(obj):
        _set_tensor_creation_stream(tensor, stream)
    return obj


def _storage_key(tensor):
    try:
        return (tensor.device, tensor.untyped_storage().data_ptr())
    except Exception:
        return None


def _mark_outputs_creation_stream(obj, stream, inputs=()):
    input_streams = {}
    for tensor in _iter_cuda_tensors(inputs):
        key = _storage_key(tensor)
        if key is not None:
            input_streams[key] = _tensor_creation_stream(tensor)

    for tensor in _iter_cuda_tensors(obj):
        origin = None
        key = _storage_key(tensor)
        if key is not None:
            origin = input_streams.get(key)
        base = getattr(tensor, '_base', None)
        if origin is None and isinstance(base, torch.Tensor):
            origin = _tensor_creation_stream(base)
        _set_tensor_creation_stream(tensor, origin or stream)
    return obj


def _copy_creation_stream(dst, src, fallback_stream):
    stream = _tensor_creation_stream(src) if isinstance(src, torch.Tensor) else None
    _set_tensor_creation_stream(dst, stream or fallback_stream)


def _sync_lifetime_to_creation_stream(obj, consumer_stream):
    """Sync tensor lifetime back to the stream that owns its allocation."""
    creators = {}
    storages = set()
    for tensor in _iter_cuda_tensors(obj):
        storage = _storage_key(tensor)
        if storage is not None:
            if storage in storages:
                continue
            storages.add(storage)
        creator = _tensor_creation_stream(tensor)
        if creator is not None and not _same_stream(creator, consumer_stream):
            creators[_stream_key(creator)] = creator

    if not creators:
        return

    event = torch.cuda.Event()
    event.record(consumer_stream)
    for creator in creators.values():
        creator.wait_event(event)


def _release_owned_storage_after_last_use(
    tag,
    obj,
    stream,
    exclude_obj=(),
    env_name='NNSCALER_EP_OVERLAP_RELEASE_NODE_OUTPUT_STORAGE',
    default=False,
    exact_storage_only=False,
):
    """Resize storages once a scheduler node has consumed all known users.

    This is a storage-level liveness check, not a tensor-shape ownership check:
    views and tensors backed by larger storages are safe to release when the
    storage does not also appear in an explicitly live object such as node
    inputs. Resizing the storage invalidates every alias, so the exclude set is
    the guardrail.
    """
    if not _env_true(env_name, default):
        return 0

    excluded = set()
    for tensor in _iter_cuda_tensors(exclude_obj):
        key = _storage_key(tensor)
        if key is not None:
            excluded.add(key)

    _sync_lifetime_to_creation_stream(obj, stream)

    storage_groups = {}
    for tensor in _iter_cuda_tensors(obj):
        key = _storage_key(tensor)
        if key is None or key in excluded:
            continue
        try:
            nbytes = tensor.untyped_storage().nbytes()
        except Exception:
            continue
        if nbytes == 0:
            continue
        group = storage_groups.setdefault(
            key, {'tensor': tensor, 'nbytes': nbytes, 'ranges': []})
        byte_range = _tensor_storage_byte_range(tensor)
        if byte_range is not None:
            group['ranges'].append(byte_range)

    released = 0
    for group in storage_groups.values():
        tensor = group['tensor']
        nbytes = group['nbytes']
        logical_nbytes = tensor.numel() * tensor.element_size()
        is_exact = (
            getattr(tensor, '_base', None) is None
            and tensor.storage_offset() == 0
            and nbytes == logical_nbytes
        )
        if exact_storage_only and not is_exact:
            _release_debug_log(
                tag,
                f'skip_non_owned_storage={_gb(nbytes):.6f}G '
                f'logical={_gb(logical_nbytes):.6f}G '
                f'offset={tensor.storage_offset()} '
                f'is_view={getattr(tensor, "_base", None) is not None}')
            continue

        force_alias_release = (
            not exact_storage_only
            and _env_true('NNSCALER_EP_OVERLAP_RELEASE_PARTIAL_ALIAS_STORAGE', False)
        )

        if not exact_storage_only and not is_exact and not force_alias_release:
            covered = _covered_storage_bytes(group['ranges'])
            if covered < nbytes:
                _release_debug_log(
                    tag,
                    f'skip_partial_alias_storage={_gb(nbytes):.6f}G '
                    f'covered={_gb(covered):.6f}G '
                    f'logical={_gb(logical_nbytes):.6f}G '
                    f'offset={tensor.storage_offset()} '
                    f'is_view={getattr(tensor, "_base", None) is not None}')
                continue
        elif force_alias_release and not is_exact:
            _release_debug_log(
                tag,
                f'force_release_alias_storage={_gb(nbytes):.6f}G '
                f'logical={_gb(logical_nbytes):.6f}G '
                f'offset={tensor.storage_offset()} '
                f'is_view={getattr(tensor, "_base", None) is not None}')

        try:
            tensor.record_stream(stream)
            tensor.untyped_storage().resize_(0)
        except RuntimeError as exc:
            _release_debug_log(tag, f'skip_resize_error={exc}')
            continue
        released += nbytes
        _release_debug_log(
            tag,
            f'resize_storage={_gb(nbytes):.6f}G '
            f'logical={_gb(logical_nbytes):.6f}G '
            f'offset={tensor.storage_offset()} '
            f'is_view={getattr(tensor, "_base", None) is not None}')
    return released


def _release_consumed_storage_after_last_use(tag, obj, stream, exclude_obj=()):
    """Release gradient/input storage after a backward consumer has finished.

    This is deliberately narrower than globally releasing every node output_grad:
    callers pass the specific tensor tuple that was just consumed and an explicit
    live exclude set for gradients that still feed another node in the same layer.
    """
    return _release_owned_storage_after_last_use(
        tag, obj, stream,
        exclude_obj=exclude_obj,
        env_name='NNSCALER_EP_OVERLAP_RELEASE_CONSUMED_GRAD_STORAGE',
        default=False,
        exact_storage_only=True,
    )


def _release_layer_dead_storage_after_backward(tag, obj, stream, exclude_obj=()):
    """Release all dead storage once a layer backward is fully complete.

    At this point tensor-shape ownership is not the right criterion anymore: if
    an activation/grad view is part of the just-finished layer and no live object
    explicitly excludes its storage, the whole storage can be resized. This is
    intentionally more aggressive than per-node consumed-grad release.
    """
    if _env_true('NNSCALER_EP_OVERLAP_SYNC_BEFORE_LAYER_DEAD_RELEASE', False):
        stream.synchronize()
    return _release_owned_storage_after_last_use(
        tag, obj, stream,
        exclude_obj=exclude_obj,
        env_name='NNSCALER_EP_OVERLAP_RELEASE_LAYER_DEAD_ALIAS_STORAGE',
        default=False,
        exact_storage_only=False,
    )


class ScheduleNode:
    """A node that executes *forward_func* on *stream*, synchronized via *event*.

    The event protocol:
        event.wait(stream)      # wait for previous node on this mb
        <execute on stream>
        event.record(stream)    # signal completion to next node on this mb

    When checkpoint=True (activation checkpointing):
      - Forward runs with torch.no_grad(), saving only inputs (not intermediates).
      - Backward recomputes forward from saved inputs to rebuild the autograd
        graph, then runs backward through it.
      - Do NOT use checkpoint for nodes containing collective communication
        (dispatch/combine) as recomputation would deadlock.
    """

    def __init__(
        self,
        forward_func,
        stream,
        event,
        backward_func=None,
        free_input=False,
        name="schedule_node",
        checkpoint=False,
        pass_node=False,
        layer_state=None,
    ):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self._default_backward
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.checkpoint = checkpoint
        self.pass_node = pass_node
        self.layer_state = layer_state
        self.inputs = None
        self.output = None
        self.before_detached = tuple()
        self.detached = tuple()
        self.loss_aux_tensors = tuple()
        self._skip_event = False  # When True, skip event wait/record in _stream_ctx

    def forward(self, inputs=()):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        with self._stream_ctx(f"{self.name} fwd"):
            self.inputs = []
            for inp in inputs:
                if inp is None:
                    self.inputs.append(None)
                elif isinstance(inp, torch.Tensor):
                    d = _make_viewless(inp).detach()
                    d.requires_grad = inp.requires_grad
                    _copy_creation_stream(d, inp, self.stream)
                    self.inputs.append(d)
                else:
                    self.inputs.append(inp)

            if self.checkpoint:
                with torch.no_grad():
                    if self.pass_node:
                        data = self.forward_func(self, *self.inputs)
                    else:
                        data = self.forward_func(*self.inputs)
                if not isinstance(data, tuple):
                    data = _make_viewless(data).detach().requires_grad_(True)
                else:
                    data = tuple(
                        _make_viewless(e).detach().requires_grad_(True)
                        if isinstance(e, torch.Tensor) else e
                        for e in data
                    )
            else:
                if self.pass_node:
                    data = self.forward_func(self, *self.inputs)
                else:
                    data = self.forward_func(*self.inputs)
                if not isinstance(data, tuple):
                    data = _make_viewless(data)
                else:
                    data = tuple(
                        _make_viewless(e) if isinstance(e, torch.Tensor) else e
                        for e in data
                    )

            self.output = data
            _mark_outputs_creation_stream(self.output, self.stream, self.inputs)

        if self.free_input:
            _sync_lifetime_to_creation_stream(inputs, self.stream)
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.is_cuda:
                    nbytes = _cuda_storage_nbytes(inp)
                    inp.record_stream(self.stream)
                    inp.untyped_storage().resize_(0)
                    if nbytes:
                        _release_debug_log(
                            f'{self.name}:free_input',
                            f'resize_storage={_gb(nbytes):.6f}G')

        return self.output

    def get_output(self):
        return self.output

    def detach_for_backward(self, tensor):
        """Detach a forward tensor for downstream use and reconnect its grad later."""
        if self.checkpoint:
            raise RuntimeError(
                f"{self.name}: detach_for_backward is unsupported with checkpoint=True")
        detached = _make_viewless(tensor).detach()
        detached.requires_grad = tensor.requires_grad
        _copy_creation_stream(detached, tensor, self.stream)
        self.before_detached = self.before_detached + (tensor,)
        self.detached = self.detached + (detached,)
        return detached

    def backward(self, output_grad, retain_graph=False):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad, retain_graph=retain_graph)

    def _backward(self, *output_grad, retain_graph=False):
        with self._stream_ctx(f"{self.name} bwd"):
            if self.checkpoint:
                if self.pass_node:
                    recomputed = self.forward_func(self, *self.inputs)
                else:
                    recomputed = self.forward_func(*self.inputs)
                if not isinstance(recomputed, tuple):
                    recomputed = (recomputed,)

                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(recomputed):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = output_grad[i] if i < len(output_grad) else None
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)
            else:
                outputs = self.output if isinstance(self.output, tuple) else (self.output,)
                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = output_grad[i] if i < len(output_grad) else None
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                for out, detached in zip(self.before_detached, self.detached):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        tensor_outputs.append(out)
                        tensor_grads.append(detached.grad)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)

        for grad in tensor_grads:
            if isinstance(grad, torch.Tensor) and grad.is_cuda:
                grad.record_stream(self.stream)

        grads = self.get_grad()
        self._release(output_grad)
        return grads

    def get_grad(self):
        grad = tuple(e.grad if e is not None else None for e in self.inputs)
        _mark_creation_stream(grad, self.stream)
        return grad[0] if len(grad) == 1 else grad

    @staticmethod
    def _default_backward(outputs, output_grad, retain_graph=False):
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=retain_graph,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )

    @contextmanager
    def _stream_ctx(self, name=None):
        if not self._skip_event:
            self.event.wait(self.stream)
        if name:
            torch.cuda.nvtx.range_push(name)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            if name:
                torch.cuda.nvtx.range_pop()
            if not self._skip_event:
                self.event.record(self.stream)

    def _release(self, extra_refs=()):
        _sync_lifetime_to_creation_stream(self.inputs, self.stream)
        _sync_lifetime_to_creation_stream(self.before_detached, self.stream)
        _sync_lifetime_to_creation_stream(self.detached, self.stream)
        _sync_lifetime_to_creation_stream(getattr(self, 'loss_aux_tensors', ()), self.stream)
        _sync_lifetime_to_creation_stream(extra_refs, self.stream)
        _release_owned_storage_after_last_use(
            f'{self.name}:node_output', self.output, self.stream,
            exclude_obj=self.inputs)
        _release_owned_storage_after_last_use(
            f'{self.name}:before_detached', self.before_detached, self.stream,
            exclude_obj=self.inputs)
        _release_owned_storage_after_last_use(
            f'{self.name}:detached', self.detached, self.stream,
            exclude_obj=self.inputs)
        _release_owned_storage_after_last_use(
            f'{self.name}:loss_aux_tensors', getattr(self, 'loss_aux_tensors', ()), self.stream,
            exclude_obj=self.inputs)
        _release_owned_storage_after_last_use(
            f'{self.name}:consumed_output_grad', extra_refs, self.stream,
            exclude_obj=(self.inputs, self.output, self.before_detached, self.detached),
            env_name='NNSCALER_EP_OVERLAP_RELEASE_NODE_GRAD_STORAGE',
            default=False)
        self.inputs = None
        self.output = None
        self.before_detached = tuple()
        self.detached = tuple()
        self.loss_aux_tensors = tuple()
        self.layer_state = None
        self.forward_func = None
        self.backward_func = None


@dataclass
class LayerCallables:
    """User-provided callable description for each layer step."""
    # MoE 4-node mode:
    attn_fn: Callable = None          # (h) -> (h, h_ln) for MoE, (h) -> h for dense
    dispatch_fn: Callable = None      # (h_ln) -> (sorted_tokens, sorted_probs)
    expert_fn: Callable = None        # (sorted_tokens, sorted_probs) -> expert_outs
    combine_fn: Callable = None       # (expert_outs, h, shared_expert_out) -> h_out

    # Dense-only:
    body_fn: Callable = None          # (h) -> h_out (wraps FFN)

    is_moe: bool = False
    step_data: dict = field(default_factory=dict)

    # Special steps (yoco_proj):
    is_special: bool = False
    special_forward: Callable = None  # (h) -> (h, special_data)
    special_backward: Callable = None  # (grad_h) -> grad_h


class MergedScheduler:
    """Communication/computation overlap merged forward-backward scheduler.

    MoE layers use 4-node scheduling with COMP/COMM stream alternation:
      - attn_router: attention + residual + gate + routing (COMP)
      - dispatch:    all2all dispatch (COMM)
      - expert:      fused FFN (COMP)
      - combine:     all2all combine + shared expert + residual (COMM)

    Dense layers use 2-node scheduling:
      - attn_node: attention + residual (COMP)
      - body_node: FFN (COMP)
    """

    def __init__(self, parallel_module, num_layers, *,
                 use_checkpoint=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self._use_4node = True

        # Async overlap: launch COMM and COMP ops from separate CPU threads
        # so their kernel launches can interleave, enabling true GPU overlap.
        # Controlled by ASYNC_4PHASE env var (default=1, set 0 to disable).
        _async_on = os.environ.get('ASYNC_4PHASE', '1') not in ('0', '')
        self._async_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='moe-overlap')
            if _async_on else None
        )
        self._early_attn_memory_release = _env_true(
            'NNSCALER_EP_OVERLAP_EARLY_ATTN_MEMORY_RELEASE', False)
        self._sync_4phase_between_phases = _env_true(
            'NNSCALER_EP_OVERLAP_SYNC_PHASES', False)
        self._sync_4phase_device = _env_true(
            'NNSCALER_EP_OVERLAP_SYNC_PHASES_DEVICE', False)
        self._aggressive_release_debug = _env_true(
            'NNSCALER_EP_OVERLAP_AGGRESSIVE_RELEASE', False)
        self._drop_forward_only_step_data = _env_true(
            'NNSCALER_EP_OVERLAP_DROP_STEP_DATA',
            self._aggressive_release_debug)
        self._sync_after_loss_backward = _env_true(
            'NNSCALER_EP_OVERLAP_SYNC_AFTER_LOSS_BWD', False)

        # Pre-allocate CUDA events for cross-stream synchronization.
        # Using separate events per barrier slot avoids re-recording while
        # a previous wait may still be pending (undefined behavior per CUDA spec).
        # 3 barrier slots in _merged_step_4phase, each needs comp + comm events
        self._barrier_comp_evts = [torch.cuda.Event() for _ in range(3)]
        self._barrier_comm_evts = [torch.cuda.Event() for _ in range(3)]
        # 2 uni-directional sync events (comp->comm and comm->comp)
        self._sync_c2m_evt = torch.cuda.Event()
        self._sync_m2c_evt = torch.cuda.Event()

    def shutdown(self):
        """Release scheduler-owned CPU worker resources."""
        pool = self._async_pool
        if pool is not None:
            self._async_pool = None
            pool.shutdown(wait=True)

    close = shutdown

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def run(self, samples, layer_callables_fn, embed_fn, loss_fn):
        """Execute merged FWD-BWD schedule.

        Args:
            samples: micro-batch list (>= 2)
            layer_callables_fn: (step_idx, sample) -> LayerCallables
            embed_fn: (sample) -> h_float32
            loss_fn: (h, sample, routing_maps, gate_scores_list) -> (loss_node, output_info)

        Returns: [output_info] per micro-batch
        """
        _mem_probe('run:start')
        num_mbs = len(samples)
        assert num_mbs >= 2, "merged FWD-BWD requires at least 2 micro-batches"

        # Ensure COMP/COMM streams see all prior default-stream work
        # (optimizer.step, zero_grad from the previous training step).
        # Non-blocking streams do NOT auto-synchronize with the default stream,
        # so without this explicit sync, the forward pass might read stale
        # parameters or see non-zeroed gradients.
        default_done = torch.cuda.Event()
        default_done.record(torch.cuda.default_stream())
        get_comp_stream().wait_event(default_done)
        get_comm_stream().wait_event(default_done)
        _mem_probe('run:after_default_wait_setup')

        num_steps = self.num_layers
        events = [torch.cuda.Event() for _ in range(num_mbs)]
        results = [None] * num_mbs

        _logger.debug("Warmup: forward mb0")
        with torch.cuda.stream(get_comp_stream()):
            h0 = embed_fn(samples[0])
            _mark_creation_stream(h0, get_comp_stream())
        _mem_probe('run:after_embed_mb0')
        _embed_h_list = [h0]  # Save embedding outputs for backward

        lc_list_0 = []
        for si in range(num_steps):
            lc_list_0.append(layer_callables_fn(si, samples[0]))

        h0, all_nodes_0, rmaps_0, eprobs_0 = self._forward_all_layers(
            h0, lc_list_0, events[0])
        _mem_probe('run:after_forward_all_layers_mb0')

        # Sync COMM→COMP: last forward node may be on COMM (MoE combine),
        # but loss_node runs on COMP with a fresh event (no wait).
        self._sync_comm_to_comp()

        loss_node_0, output_info_0 = loss_fn(h0, samples[0], rmaps_0, eprobs_0)
        loss_0 = loss_node_0.forward((h0,))
        results[0] = output_info_0['output_tuple']
        _mem_probe('run:after_loss_forward_mb0')

        del h0, rmaps_0, eprobs_0, output_info_0

        prev_all_nodes = all_nodes_0
        prev_loss_node = loss_node_0

        for mb_i in range(num_mbs - 1):
            fwd_sample = samples[mb_i + 1]
            fwd_event = events[mb_i + 1]

            _logger.debug(f"[MERGED] bwd(mb{mb_i}) + fwd(mb{mb_i+1})")

            # Launch embed on COMM — overlaps with loss_bwd on COMP.
            # embed_fn only reads sample data + embedding weight, independent of COMP.
            with torch.cuda.stream(get_comm_stream()):
                fwd_h = embed_fn(fwd_sample)
                _mark_creation_stream(fwd_h, get_comm_stream())
            _embed_h_list.append(fwd_h)
            _mem_probe(f'run:after_embed_mb{mb_i + 1}')

            # Loss backward on COMP — overlaps with embed on COMM
            with torch.cuda.stream(get_comp_stream()):
                loss_grad = torch.ones_like(prev_loss_node.get_output())
                _mark_creation_stream(loss_grad, get_comp_stream())
            with nnscaler.sync_grad_when(False):
                grad_h = prev_loss_node.backward(loss_grad)
            del loss_grad
            self._release_loss_wgrad_scratch()
            self._sync_after_loss_backward_for_debug(
                f'run:after_loss_backward_mb{mb_i}')
            _mem_probe(f'run:after_loss_backward_mb{mb_i}')

            prev_loss_node._release()

            # Create layer callables (CPU work, overlaps with GPU)
            fwd_lc_list = []
            for si in range(num_steps):
                fwd_lc_list.append(layer_callables_fn(si, fwd_sample))

            fwd_routing_maps = []
            fwd_expert_probs = []
            fwd_all_nodes = [None] * num_steps

            bwd_idx = num_steps - 1
            fwd_idx = 0

            # (4) Sync COMM→COMP: embed result available for forward layers
            self._sync_comm_to_comp()

            while fwd_idx < num_steps and bwd_idx >= 0:
                fwd_lc = fwd_lc_list[fwd_idx]

                if fwd_lc.is_special:
                    # Sync COMM→COMP: previous MoE layer's combine ran on COMM.
                    self._sync_comm_to_comp()
                    with torch.cuda.stream(get_comp_stream()):
                        special_input = fwd_h
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                        _mark_outputs_creation_stream(
                            (fwd_h, special_data), get_comp_stream(), (special_input,))
                    fwd_all_nodes[fwd_idx] = ('special', fwd_lc)
                    fwd_idx += 1
                    continue

                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    bwd_lc = bwd_entry[1]
                    with nnscaler.sync_grad_when(False):
                        with torch.cuda.stream(get_comp_stream()):
                            special_grad_input = grad_h
                            grad_h = bwd_lc.special_backward(grad_h)
                            _mark_outputs_creation_stream(
                                grad_h, get_comp_stream(), (special_grad_input,))
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue

                with nnscaler.sync_grad_when(False):
                    _mem_probe(f'run:before_merged mb{mb_i} f{fwd_idx} b{bwd_idx}')
                    fwd_h, grad_h, fwd_entry = self._merged_step_general(
                        bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)
                    _mem_probe(f'run:after_merged mb{mb_i} f{fwd_idx} b{bwd_idx}')

                prev_all_nodes[bwd_idx] = None

                fwd_all_nodes[fwd_idx] = fwd_entry

                if fwd_lc.is_moe:
                    fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                    fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))

                fwd_idx += 1
                bwd_idx -= 1

            while fwd_idx < num_steps:
                fwd_lc = fwd_lc_list[fwd_idx]
                if fwd_lc.is_special:
                    # Sync COMM→COMP: previous MoE layer's combine ran on COMM.
                    self._sync_comm_to_comp()
                    with torch.cuda.stream(get_comp_stream()):
                        special_input = fwd_h
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                        _mark_outputs_creation_stream(
                            (fwd_h, special_data), get_comp_stream(), (special_input,))
                    fwd_all_nodes[fwd_idx] = ('special', fwd_lc)
                    fwd_idx += 1
                    continue

                if fwd_lc.is_moe and self._use_4node:
                    fwd_h, fwd_entry = self._forward_single_layer_4node(
                        fwd_h, fwd_lc, fwd_event)
                else:
                    fwd_h, fwd_entry = self._forward_single_layer(
                        fwd_h, fwd_lc, fwd_event)
                _mem_probe(f'run:after_forward_tail mb{mb_i + 1} f{fwd_idx}')
                fwd_all_nodes[fwd_idx] = fwd_entry
                if fwd_lc.is_moe:
                    fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                    fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))
                fwd_idx += 1

            while bwd_idx >= 0:
                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    bwd_lc = bwd_entry[1]
                    with nnscaler.sync_grad_when(False):
                        with torch.cuda.stream(get_comp_stream()):
                            special_grad_input = grad_h
                            grad_h = bwd_lc.special_backward(grad_h)
                            _mark_outputs_creation_stream(
                                grad_h, get_comp_stream(), (special_grad_input,))
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue
                with nnscaler.sync_grad_when(False):
                    grad_h = self._backward_entry(bwd_entry, grad_h)
                _mem_probe(f'run:after_backward_tail mb{mb_i} b{bwd_idx}')
                prev_all_nodes[bwd_idx] = None
                bwd_idx -= 1

            del prev_all_nodes
            # Propagate gradient through embedding graph to tok_embed weight
            with nnscaler.sync_grad_when(False):
                _embed_h_list[mb_i].backward(grad_h)
            _mem_probe(f'run:after_embed_backward_mb{mb_i}')

            # Sync COMM→COMP: last fwd node may be on COMM (MoE combine),
            # but loss_node runs on COMP with a fresh event.
            self._sync_comm_to_comp()

            fwd_loss_node, fwd_output_info = loss_fn(
                fwd_h, fwd_sample, fwd_routing_maps, fwd_expert_probs)
            fwd_loss = fwd_loss_node.forward((fwd_h,))
            results[mb_i + 1] = fwd_output_info['output_tuple']
            _mem_probe(f'run:after_loss_forward_mb{mb_i + 1}')

            prev_all_nodes = fwd_all_nodes
            prev_loss_node = fwd_loss_node

            if mb_i == 0:
                del all_nodes_0, loss_node_0, loss_0
            del fwd_h, fwd_loss, fwd_routing_maps, fwd_expert_probs
            del fwd_lc_list

        _logger.debug(f"Cooldown: backward mb{num_mbs-1}")
        with nnscaler.sync_grad_when(False):
            with torch.cuda.stream(get_comp_stream()):
                loss_grad = torch.ones_like(prev_loss_node.get_output())
                _mark_creation_stream(loss_grad, get_comp_stream())
            grad_h = prev_loss_node.backward(loss_grad)
            del loss_grad
            self._release_loss_wgrad_scratch()
            self._sync_after_loss_backward_for_debug('run:cooldown_after_loss_backward')
            _mem_probe('run:cooldown_after_loss_backward')
            for i in reversed(range(num_steps)):
                entry = prev_all_nodes[i]
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'special':
                    lc = entry[1]
                    with torch.cuda.stream(get_comp_stream()):
                        special_grad_input = grad_h
                        grad_h = lc.special_backward(grad_h)
                        _mark_outputs_creation_stream(
                            grad_h, get_comp_stream(), (special_grad_input,))
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry(entry, grad_h)
                _mem_probe(f'run:cooldown_after_backward b{i}')
            # Propagate gradient through embedding graph to tok_embed weight
            _embed_h_list[-1].backward(grad_h)
            _mem_probe('run:after_final_embed_backward')

        # Make default stream wait for COMP/COMM to finish without blocking host.
        comp_done = torch.cuda.Event()
        comm_done = torch.cuda.Event()
        comp_done.record(get_comp_stream())
        comm_done.record(get_comm_stream())
        torch.cuda.default_stream().wait_event(comp_done)
        torch.cuda.default_stream().wait_event(comm_done)
        _mem_probe('run:after_default_stream_wait')

        for i in range(len(results)):
            if results[i] is not None:
                results[i] = tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t
                    for t in results[i]
                )
        _mem_probe('run:after_detach_results')

        del prev_all_nodes, prev_loss_node
        _mem_probe('run:end', dump=True)
        _phase_mem_probe('run:end')
        if _env_true('NNSCALER_MEM_DEBUG_EMPTY_CACHE', False):
            torch.cuda.empty_cache()
            _mem_probe('run:after_empty_cache', dump=True)

        return results

    def _create_nodes(self, lc, event):
        """Create 2 ScheduleNodes for a layer (dense or MoE fallback)."""
        comp_stream = get_comp_stream()

        attn_node = ScheduleNode(
            lc.attn_fn, comp_stream, event,
            name="attn", checkpoint=self.use_checkpoint)

        if lc.is_moe:
            raise NotImplementedError(
                "2-node MoE fallback is unsupported. MoE layers must use the "
                "4-node dispatch/expert/combine schedule.")

        body_node = ScheduleNode(
            lc.body_fn, comp_stream, event,
            name="ffn", checkpoint=self.use_checkpoint)

        return (attn_node, body_node)

    def _create_nodes_4(self, lc, event):
        """Create 4 ScheduleNodes for MoE layer, alternating COMP/COMM streams."""
        comp_stream = get_comp_stream()
        comm_stream = get_comm_stream()
        layer_state = TransformerLayerState()
        free_combine_input = lc.step_data.get('_free_combine_input', False)
        free_dispatch_input = lc.step_data.get('_free_dispatch_input', False)
        if free_combine_input and not (
            lc.step_data.get('_use_state_residual', False) and
            lc.step_data.get('_use_state_shared_expert', False)
        ):
            raise RuntimeError(
                "combine input can be freed only when residual/shared expert use state")

        attn_node = ScheduleNode(
            lc.attn_fn, comp_stream, event,
            name="attn_router", checkpoint=self.use_checkpoint,
            pass_node=lc.step_data.get('_state_lifetime_in_callables', False),
            layer_state=layer_state)
        attn_node.uses_state_residual = lc.step_data.get('_use_state_residual', False)
        attn_node.uses_state_shared_expert = lc.step_data.get('_use_state_shared_expert', False)

        dispatch_node = ScheduleNode(
            lc.dispatch_fn, comm_stream, event,
            name="dispatch", checkpoint=False,
            free_input=free_dispatch_input,
            pass_node=lc.step_data.get('_state_lifetime_in_callables', False),
            layer_state=layer_state)
        dispatch_node.uses_state_dispatched_probs = lc.step_data.get(
            '_use_state_dispatched_probs', False)

        expert_node = ScheduleNode(
            lc.expert_fn, comp_stream, event,
            name="expert", checkpoint=self.use_checkpoint,
            free_input=lc.step_data.get('_free_expert_input', False),
            pass_node=lc.step_data.get('_state_lifetime_in_callables', False),
            layer_state=layer_state)
        expert_node.uses_state_dispatched_probs = lc.step_data.get(
            '_use_state_dispatched_probs', False)
        expert_node.uses_state_shared_expert = lc.step_data.get(
            '_use_state_shared_expert', False)
        expert_node.has_h_ln_input = not lc.step_data.get('_shared_expert_in_attn', False)

        combine_node = ScheduleNode(
            lc.combine_fn, comm_stream, event,
            name="combine", checkpoint=False,
            free_input=free_combine_input,
            pass_node=lc.step_data.get('_state_lifetime_in_callables', False),
            layer_state=layer_state)
        combine_node.uses_state_residual = lc.step_data.get('_use_state_residual', False)

        return (attn_node, dispatch_node, expert_node, combine_node)

    def _sync_comm_to_comp(self):
        self._sync_m2c_evt.record(get_comm_stream())
        get_comp_stream().wait_event(self._sync_m2c_evt)

    def _sync_comp_to_comm(self):
        self._sync_c2m_evt.record(get_comp_stream())
        get_comm_stream().wait_event(self._sync_c2m_evt)

    def _cross_stream_barrier(self, slot=0):
        """Bidirectional sync: each stream waits for the other's current position.

        Records events on both streams simultaneously, then makes each wait
        for the other. Uses pre-allocated events indexed by slot to avoid
        event creation overhead and re-recording conflicts.
        """
        comp_evt = self._barrier_comp_evts[slot]
        comm_evt = self._barrier_comm_evts[slot]
        comp_evt.record(get_comp_stream())
        comm_evt.record(get_comm_stream())
        get_comm_stream().wait_event(comp_evt)
        get_comp_stream().wait_event(comm_evt)

    def _sync_4phase_boundary_for_debug(self, tag):
        if not self._sync_4phase_between_phases:
            return
        if self._sync_4phase_device:
            torch.cuda.synchronize()
            _mem_probe(f'4phase:after_device_sync_{tag}')
            _phase_mem_probe(f'4phase:after_device_sync_{tag}')
            return
        get_comp_stream().synchronize()
        get_comm_stream().synchronize()
        _mem_probe(f'4phase:after_sync_{tag}')
        _phase_mem_probe(f'4phase:after_sync_{tag}')

    def _release_debug_probe(self, tag):
        if not self._aggressive_release_debug:
            return
        if self._sync_4phase_between_phases:
            get_comp_stream().synchronize()
            get_comm_stream().synchronize()
        _mem_probe(f'4phase:after_release_{tag}')
        _phase_mem_probe(f'4phase:after_release_{tag}')

    def _sync_after_loss_backward_for_debug(self, tag):
        if not self._sync_after_loss_backward:
            return
        if _env_true('NNSCALER_EP_OVERLAP_SYNC_AFTER_LOSS_BWD_DEVICE', False):
            torch.cuda.synchronize()
            _mem_probe(f'{tag}:after_sync_loss_bwd')
            return
        get_comp_stream().synchronize()
        if _env_true('NNSCALER_EP_OVERLAP_SYNC_AFTER_LOSS_BWD_BOTH_STREAMS', False):
            get_comm_stream().synchronize()
        _mem_probe(f'{tag}:after_sync_loss_bwd')

    def _release_loss_wgrad_scratch(self):
        if not _env_true('MSRALLM_CE_RELEASE_WGRAD_SCRATCH_AFTER_BWD', False):
            return
        try:
            from arch.linear import release_ce_wgrad_scratch
        except Exception:
            return
        with torch.cuda.stream(get_comp_stream()):
            release_ce_wgrad_scratch()

    def _drop_step_data_keys(self, lc, keys, tag):
        if not self._drop_forward_only_step_data:
            return
        dropped = []
        dropped_bytes = 0
        for key in keys:
            if key not in lc.step_data:
                continue
            value = lc.step_data.pop(key)
            nbytes = _cuda_storage_nbytes(value)
            dropped_bytes += nbytes
            dropped.append(f'{key}:{_gb(nbytes):.6f}G')
            del value
        if dropped:
            _release_debug_log(
                tag,
                f"dropped_step_data={','.join(dropped)} "
                f'total_tensor_storage={_gb(dropped_bytes):.6f}G')

    def _drop_after_dispatch_forward(self, lc, tag):
        # The private routing map is only needed to build dispatch metadata.
        # Keep public routing_map/gate_scores because the loss path consumes them.
        self._drop_step_data_keys(lc, ('_routing_map', '_deepep_token_indices'), tag)

    def _drop_after_combine_forward(self, lc, tag):
        # meta/precomputed are forward-only scheduler inputs. Autograd Functions
        # save what they need in their own ctx during forward.
        self._drop_step_data_keys(
            lc,
            ('meta', 'precomputed', 'aux_counts', '_loss_aux_tensors'),
            tag,
        )

    @staticmethod
    def _as_tuple(value):
        return value if isinstance(value, tuple) else (value,)

    @staticmethod
    def _prepare_state_residual(lc, attn_node, attn_out):
        attn_items = list(attn_out if isinstance(attn_out, tuple) else (attn_out,))
        if lc.step_data.get('_state_lifetime_in_callables', False):
            if len(attn_items) < 2:
                raise RuntimeError("MoE attention callable must return h_ln and routing_probs")
            attn_node.uses_state_residual = lc.step_data.get('_use_state_residual', False)
            attn_node.uses_state_shared_expert = lc.step_data.get(
                '_use_state_shared_expert', False)
            return None, attn_items[0], attn_items[1]

        h_residual, h_ln, routing_probs = attn_items[:3]
        tail = attn_items[3:]

        shared_in_attn = lc.step_data.get('_shared_expert_in_attn', False)
        if shared_in_attn:
            shared_expert_out = tail.pop() if tail else None
            if lc.step_data.get('_use_state_shared_expert', False):
                if shared_expert_out is not None:
                    lc.step_data['_state_shared_expert_out_for_combine'] = (
                        attn_node.detach_for_backward(shared_expert_out))
                attn_node.uses_state_shared_expert = True
            else:
                attn_node.uses_state_shared_expert = False
                tail.append(shared_expert_out)
        else:
            attn_node.uses_state_shared_expert = False

        if lc.step_data.get('_use_state_residual', False):
            lc.step_data['_state_residual_for_combine'] = (
                attn_node.detach_for_backward(h_residual))
            attn_node.output = tuple([h_ln, routing_probs] + tail)
            attn_node.uses_state_residual = True
            return None, h_ln, routing_probs

        attn_node.uses_state_residual = False
        attn_node.output = tuple([h_residual, h_ln, routing_probs] + tail)
        return h_residual, h_ln, routing_probs

    @staticmethod
    def _prepare_state_dispatched_probs(lc, dispatch_node, dispatch_out):
        dispatch_items = list(dispatch_out if isinstance(dispatch_out, tuple) else (dispatch_out,))
        if lc.step_data.get('_state_lifetime_in_callables', False):
            dispatch_node.uses_state_dispatched_probs = lc.step_data.get(
                '_use_state_dispatched_probs', False)
            if len(dispatch_items) != 1:
                raise RuntimeError(
                    "Megatron-style dispatch callable should return dispatched tokens only")
            return dispatch_items[0]

        if not lc.step_data.get('_use_state_dispatched_probs', False):
            dispatch_node.uses_state_dispatched_probs = False
            return dispatch_out

        if len(dispatch_items) < 2:
            raise RuntimeError("dispatch state requires dispatched tokens and probs")

        dispatched_tokens, dispatched_probs = dispatch_items[:2]
        lc.step_data['_state_dispatched_probs_for_expert'] = (
            dispatch_node.detach_for_backward(dispatched_probs))
        dispatch_node.output = (dispatched_tokens,)
        dispatch_node.uses_state_dispatched_probs = True
        return dispatched_tokens

    @staticmethod
    def _expert_forward_inputs(lc, expert_node, dispatch_out, h_ln):
        use_state_probs = lc.step_data.get('_use_state_dispatched_probs', False)
        shared_in_attn = lc.step_data.get('_shared_expert_in_attn', False)
        expert_node.uses_state_dispatched_probs = use_state_probs
        expert_node.has_h_ln_input = not shared_in_attn

        if use_state_probs:
            inputs = [dispatch_out]
        else:
            inputs = list(dispatch_out if isinstance(dispatch_out, tuple) else (dispatch_out,))
        if not shared_in_attn:
            inputs.append(h_ln)
        return tuple(inputs)

    @staticmethod
    def _prepare_state_shared_expert(lc, expert_node, expert_result):
        if lc.step_data.get('_shared_expert_in_attn', False):
            expert_node.uses_state_shared_expert = lc.step_data.get(
                '_use_state_shared_expert', False)
            return expert_result, None

        expert_out, shared_expert_out = expert_result
        use_state_shared = lc.step_data.get('_use_state_shared_expert', False)
        if use_state_shared:
            if shared_expert_out is not None:
                lc.step_data['_state_shared_expert_out_for_combine'] = (
                    expert_node.detach_for_backward(shared_expert_out))
            expert_node.output = (expert_out,)
            expert_node.uses_state_shared_expert = True
            return expert_out, None

        expert_node.uses_state_shared_expert = False
        return expert_out, shared_expert_out

    @staticmethod
    def _combine_forward_inputs(lc, expert_out, h_residual, shared_expert_out):
        use_state_residual = lc.step_data.get('_use_state_residual', False)
        use_state_shared = lc.step_data.get('_use_state_shared_expert', False)
        if use_state_residual and use_state_shared:
            return (expert_out,)
        if use_state_residual:
            return (expert_out, shared_expert_out)
        return (expert_out, h_residual, shared_expert_out)

    @staticmethod
    def _expert_backward_grads(expert_node, combine_node, combine_grads):
        combine_grads = MergedScheduler._as_tuple(combine_grads)
        use_state_residual = getattr(combine_node, 'uses_state_residual', False)
        use_state_shared = getattr(expert_node, 'uses_state_shared_expert', False)
        if use_state_shared:
            return (combine_grads[0],)
        if use_state_residual:
            return (combine_grads[0], combine_grads[1])
        return (combine_grads[0], combine_grads[2])

    @staticmethod
    def _dispatch_backward_grads(dispatch_node, expert_grads):
        expert_grads = MergedScheduler._as_tuple(expert_grads)
        if getattr(dispatch_node, 'uses_state_dispatched_probs', False):
            return expert_grads[0]
        return (expert_grads[0], expert_grads[1])

    @staticmethod
    def _expert_h_ln_grad(expert_node, expert_grads):
        if not getattr(expert_node, 'has_h_ln_input', False):
            return None
        expert_grads = MergedScheduler._as_tuple(expert_grads)
        idx = 1 if getattr(expert_node, 'uses_state_dispatched_probs', False) else 2
        return expert_grads[idx] if len(expert_grads) > idx else None

    @staticmethod
    def _add_optional_grad(left, right):
        if right is None:
            return left
        if left is None:
            return right
        return left + right

    @staticmethod
    def _residual_grad_for_attn(attn_node, combine_grads):
        if getattr(attn_node, 'uses_state_residual', False):
            return None
        return MergedScheduler._as_tuple(combine_grads)[1]

    @staticmethod
    def _attn_backward_grads(attn_node, grad_h, grad_h_ln, grad_routing):
        outputs = attn_node.get_output()
        if not isinstance(outputs, tuple):
            return grad_h

        if getattr(attn_node, 'uses_state_residual', False):
            grads = [grad_h_ln, grad_routing]
        else:
            grads = [grad_h, grad_h_ln, grad_routing]
        aux_tensors = getattr(attn_node, 'loss_aux_tensors', ())
        for tensor in aux_tensors:
            if isinstance(tensor, torch.Tensor):
                grads.append(tensor.grad)
            else:
                grads.append(None)
        while len(grads) < len(outputs):
            grads.append(None)
        return tuple(grads[:len(outputs)])

    def _forward_all_layers(self, h, lc_list, event):
        """Warmup: forward all layers, collect nodes and routing data."""
        all_nodes = []
        routing_maps = []
        expert_probs = []

        for si, lc in enumerate(lc_list):
            if lc.is_special:
                # Sync COMM→COMP: previous MoE layer's combine ran on COMM,
                # special_forward runs on COMP and needs the COMM-produced h.
                self._sync_comm_to_comp()
                with torch.cuda.stream(get_comp_stream()):
                    special_input = h
                    h, special_data = lc.special_forward(h)
                    _mark_outputs_creation_stream(
                        (h, special_data), get_comp_stream(), (special_input,))
                all_nodes.append(('special', lc))
                continue

            if lc.is_moe and self._use_4node:
                h, entry = self._forward_single_layer_4node(h, lc, event)
            else:
                nodes = self._create_nodes(lc, event)
                attn_n, body_n = nodes

                attn_out = attn_n.forward((h,))
                h = body_n.forward(attn_out)

                if attn_n.checkpoint:
                    attn_n.output = None
                if body_n.checkpoint:
                    body_n.output = None

                entry = ('layer2', nodes)

            if lc.is_moe:
                routing_maps.append(lc.step_data.get('routing_map'))
                expert_probs.append(lc.step_data.get('gate_scores'))

            all_nodes.append(entry)

        return h, all_nodes, routing_maps, expert_probs

    def _forward_single_layer(self, h, lc, event):
        """Forward a single dense layer."""
        nodes = self._create_nodes(lc, event)
        attn_n, body_n = nodes

        attn_out = attn_n.forward((h,))
        h = body_n.forward(attn_out)

        if attn_n.checkpoint:
            attn_n.output = None
        if body_n.checkpoint:
            body_n.output = None

        return h, ('layer2', nodes)

    def _forward_single_layer_4node(self, h, lc, event):
        """Forward a single MoE layer through 4 nodes.

        MoE auxiliary tensors may be carried through node state rather than as
        ordinary node inputs, matching Megatron's safe-release schedule.
        """
        nodes = self._create_nodes_4(lc, event)
        attn_n, dispatch_n, expert_n, combine_n = nodes

        attn_out = attn_n.forward((h,))
        h_residual, h_ln, routing_probs = self._prepare_state_residual(
            lc, attn_n, attn_out)
        loss_aux_tensors = lc.step_data.get('_loss_aux_tensors')
        if loss_aux_tensors is not None:
            attn_n.loss_aux_tensors = loss_aux_tensors
        dispatch_out = dispatch_n.forward((h_ln, routing_probs))
        dispatch_out = self._prepare_state_dispatched_probs(lc, dispatch_n, dispatch_out)
        self._drop_after_dispatch_forward(lc, 'forward_single:after_dispatch')
        expert_result = expert_n.forward(
            self._expert_forward_inputs(lc, expert_n, dispatch_out, h_ln))
        expert_out, shared_expert_out = self._prepare_state_shared_expert(
            lc, expert_n, expert_result)
        combine_n.uses_state_residual = lc.step_data.get('_use_state_residual', False)
        h_out = combine_n.forward(
            self._combine_forward_inputs(lc, expert_out, h_residual, shared_expert_out))
        self._drop_after_combine_forward(lc, 'forward_single:after_combine')

        for n in nodes:
            if n.checkpoint:
                n.output = None

        return h_out, ('layer4', nodes)

    def _backward_layer(self, nodes, grad_h):
        """Backward through a 2-node layer: body -> attn."""
        attn_n, body_n = nodes
        consumed_grad_h = grad_h
        layer_dead_refs = []
        # Ensure all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            body_grads = body_n.backward(grad_h)
            grad_x = attn_n.backward(body_grads)
            layer_dead_refs.extend((consumed_grad_h, body_grads))
            _release_consumed_storage_after_last_use(
                'layer2:consumed_output_grad', consumed_grad_h,
                get_comp_stream(), exclude_obj=(body_grads, grad_x))
            _release_consumed_storage_after_last_use(
                'layer2:consumed_body_grads', body_grads,
                get_comp_stream(), exclude_obj=grad_x)
            _release_layer_dead_storage_after_backward(
                'layer2:dead_after_layer_backward', layer_dead_refs,
                get_comp_stream(), exclude_obj=grad_x)
        return grad_x

    def _backward_layer_4node(self, nodes, grad_h):
        """Backward through a 4-node MoE layer: combine→expert→dispatch→attn.

        Dispatch probs and shared expert output may be state tensors. Their
        gradients are reconnected through the node that detached them.
        """
        attn_n, dispatch_n, expert_n, combine_n = nodes
        consumed_grad_h = grad_h
        layer_dead_refs = [consumed_grad_h]

        # Ensure all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            self._sync_comp_to_comm()

            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            combine_grads = combine_n.backward(grad_h)
            layer_dead_refs.append(combine_grads)
            _release_consumed_storage_after_last_use(
                'layer4:combine_consumed_output_grad', consumed_grad_h,
                get_comm_stream(), exclude_obj=combine_grads)

            # expert backward needs grads for both outputs: expert_outs and shared_expert_out
            expert_bwd_grads = self._expert_backward_grads(
                expert_n, combine_n, combine_grads)
            live_residual_grad = self._residual_grad_for_attn(attn_n, combine_grads)
            expert_grads = expert_n.backward(expert_bwd_grads)
            layer_dead_refs.extend((expert_bwd_grads, expert_grads))
            _release_consumed_storage_after_last_use(
                'layer4:expert_consumed_output_grads', expert_bwd_grads,
                get_comp_stream(), exclude_obj=(live_residual_grad, expert_grads))

            dispatch_bwd_grads = self._dispatch_backward_grads(dispatch_n, expert_grads)
            expert_h_ln_grad = self._expert_h_ln_grad(expert_n, expert_grads)
            dispatch_grads = dispatch_n.backward(dispatch_bwd_grads)
            layer_dead_refs.extend((dispatch_bwd_grads, dispatch_grads, expert_h_ln_grad))
            _release_consumed_storage_after_last_use(
                'layer4:dispatch_consumed_output_grads', dispatch_bwd_grads,
                get_comm_stream(), exclude_obj=(expert_h_ln_grad, dispatch_grads))

            self._sync_comm_to_comp()

            grad_h_ln_total = self._add_optional_grad(
                dispatch_grads[0], expert_h_ln_grad)

            attn_grads = self._attn_backward_grads(
                attn_n, live_residual_grad,
                grad_h_ln_total, dispatch_grads[1])
            grad_x = attn_n.backward(attn_grads)
            layer_dead_refs.extend((live_residual_grad, grad_h_ln_total, attn_grads))
            _release_consumed_storage_after_last_use(
                'layer4:attn_consumed_output_grads', attn_grads,
                get_comp_stream(), exclude_obj=grad_x)
            _release_layer_dead_storage_after_backward(
                'layer4:dead_after_layer_backward', layer_dead_refs,
                get_comp_stream(), exclude_obj=grad_x)

            _sync_lifetime_to_creation_stream((dispatch_grads[0],), get_comp_stream())
            del (combine_grads, expert_bwd_grads, expert_grads,
                 dispatch_bwd_grads, dispatch_grads, expert_h_ln_grad,
                 live_residual_grad, grad_h_ln_total, attn_grads,
                 layer_dead_refs)

        return grad_x

    def _backward_entry(self, entry, grad_h):
        """Dispatch backward by entry type."""
        tag, nodes = entry
        if tag == 'layer2':
            return self._backward_layer(nodes, grad_h)
        elif tag == 'layer4':
            return self._backward_layer_4node(nodes, grad_h)
        else:
            raise ValueError(f"Unknown entry tag: {tag}")

    def _merged_step(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h):
        """2-phase overlap for dense layers."""
        _, bwd_nodes = bwd_entry
        bwd_attn, bwd_body = bwd_nodes
        consumed_grad_h = grad_h
        layer_dead_refs = [consumed_grad_h]

        fwd_nodes = self._create_nodes(fwd_lc, fwd_event)
        fwd_attn, fwd_body = fwd_nodes

        # Ensure all intermediate ops run on COMP stream, not default stream.
        with torch.cuda.stream(get_comp_stream()):
            body_grads = bwd_body.backward(grad_h)
            fwd_attn_out = fwd_attn.forward((fwd_h,))

            grad_x = bwd_attn.backward(body_grads)
            layer_dead_refs.append(body_grads)
            _release_consumed_storage_after_last_use(
                '2phase:body_consumed_output_grad', consumed_grad_h,
                get_comp_stream(), exclude_obj=(body_grads, grad_x))
            _release_consumed_storage_after_last_use(
                '2phase:attn_consumed_body_grads', body_grads,
                get_comp_stream(), exclude_obj=grad_x)
            _release_layer_dead_storage_after_backward(
                '2phase:dead_after_layer_backward', layer_dead_refs,
                get_comp_stream(), exclude_obj=(grad_x, fwd_nodes, fwd_h, fwd_attn_out))
            fwd_h_out = fwd_body.forward(fwd_attn_out)

            if fwd_attn.checkpoint:
                fwd_attn.output = None
            if fwd_body.checkpoint:
                fwd_body.output = None

        return fwd_h_out, grad_x, ('layer2', fwd_nodes)

    def _merged_step_4phase(self, bwd_nodes, fwd_lc, fwd_event, grad_h, fwd_h):
        """4-phase overlap for MoE layers.

        Each phase interleaves COMP and COMM operations from different
        micro-batches. Within each phase, COMP and COMM run in PARALLEL on
        the GPU because we skip the per-node shared-event protocol (which
        would serialize all operations). Instead, we use explicit cross-stream
        barriers at phase boundaries to enforce data dependencies.

        Phase 1: f_attn_router(COMP) || b_combine(COMM)   [combine = pure comm]
        Phase 2: b_expert    (COMP) || f_dispatch (COMM)   [expert includes shared expert bwd]
        Phase 3: f_expert    (COMP) || b_dispatch (COMM)   [expert includes shared expert fwd]
        Phase 4: b_attn      (COMP) || f_combine  (COMM)

        Shared expert is merged into expert_fn (COMP stream), keeping
        combine_fn (COMM stream) as pure communication.
        """
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        pool = self._async_pool  # None in sequential mode

        # --- Skip per-node event protocol during merged step ---
        # The default shared event serializes all nodes (COMP waits for COMM
        # and vice versa), preventing within-phase parallelism. By setting
        # _skip_event=True, each node still switches to its own stream and
        # does not wait/record the shared event. Cross-stream dependencies are
        # enforced explicitly via CUDA events at phase boundaries.
        all_nodes = (*bwd_nodes, *fwd_nodes)
        for n in all_nodes:
            n._skip_event = True

        # Ensure grad addition runs on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        # ScheduleNode calls internally switch to their own stream and restore on exit.
        with torch.cuda.stream(get_comp_stream()):
            consumed_grad_h = grad_h
            layer_dead_refs = [consumed_grad_h]
            # Initial sync: COMP→COMM so COMM can read grad_h (from loss_bwd on COMP)
            self._sync_comp_to_comm()
            _mem_probe('4phase:start')

            # Phase 1: COMM(b_combine) || COMP(f_attn_router)
            # b_combine is pure communication (shared expert merged into expert).
            if pool is not None:
                fut_combine = pool.submit(bwd_combine.backward, grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
                combine_grads = fut_combine.result()
            else:
                combine_grads = bwd_combine.backward(grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
            _release_consumed_storage_after_last_use(
                '4phase:combine_consumed_output_grad', consumed_grad_h,
                get_comm_stream(), exclude_obj=combine_grads)
            layer_dead_refs.append(combine_grads)
            fwd_h_residual, fwd_h_ln, fwd_routing_probs = self._prepare_state_residual(
                fwd_lc, fwd_attn, fwd_attn_out)
            combine_grads_for_attn = combine_grads
            live_residual_grad = self._residual_grad_for_attn(
                bwd_attn, combine_grads_for_attn)
            loss_aux_tensors = fwd_lc.step_data.get('_loss_aux_tensors')
            if loss_aux_tensors is not None:
                fwd_attn.loss_aux_tensors = loss_aux_tensors
            if self._aggressive_release_debug:
                # These Python refs are redundant after the nodes have captured
                # their own forward/backward state. If active memory does not
                # move here, the storage is still owned by node/autograd state.
                fwd_attn_out = None
                fwd_h = None
                grad_h = None
                self._release_debug_probe('phase1_refs')
            _mem_probe('4phase:after_phase1')
            _phase_mem_probe('4phase:after_phase1')
            self._sync_4phase_boundary_for_debug('phase1')

            # Phase boundary: Phase 2 COMP needs COMM output (combine_grads),
            # Phase 2 COMM needs COMP output (attn_out + precomputed metadata)
            self._cross_stream_barrier(slot=0)
            _mem_probe('4phase:after_barrier0')

            # Phase 2: COMM(f_dispatch) || COMP(b_expert)
            # expert backward includes shared expert backward (both on COMP).
            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            expert_bwd_grads = self._expert_backward_grads(
                bwd_expert, bwd_combine, combine_grads)
            if pool is not None:
                fut_dispatch = pool.submit(fwd_dispatch.forward, (fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward(expert_bwd_grads)
                fwd_dispatch_out = fut_dispatch.result()
            else:
                fwd_dispatch_out = fwd_dispatch.forward((fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward(expert_bwd_grads)
            _release_consumed_storage_after_last_use(
                '4phase:expert_consumed_output_grads', expert_bwd_grads,
                get_comp_stream(), exclude_obj=(live_residual_grad, expert_grads))
            layer_dead_refs.extend((expert_bwd_grads, expert_grads))
            expert_bwd_grads = None
            if getattr(bwd_attn, 'uses_state_residual', False):
                combine_grads = None
                combine_grads_for_attn = None
            if self._aggressive_release_debug and getattr(bwd_attn, 'uses_state_residual', False):
                self._release_debug_probe('phase2_combine_grads')
            fwd_dispatch_out = self._prepare_state_dispatched_probs(
                fwd_lc, fwd_dispatch, fwd_dispatch_out)
            self._drop_after_dispatch_forward(fwd_lc, '4phase:after_dispatch_forward')
            _mem_probe('4phase:after_phase2')
            _phase_mem_probe('4phase:after_phase2')
            self._sync_4phase_boundary_for_debug('phase2')

            # Phase boundary: Phase 3 COMP needs COMM output (dispatch_out),
            # Phase 3 COMM needs COMP output (expert_grads)
            self._cross_stream_barrier(slot=1)
            _mem_probe('4phase:after_barrier1')

            if self._early_attn_memory_release:
                # Release previous attention activations before launching the next
                # expert forward, which is the next large allocation region.
                dispatch_bwd_grads = self._dispatch_backward_grads(
                    bwd_dispatch, expert_grads)
                expert_h_ln_grad = self._expert_h_ln_grad(bwd_expert, expert_grads)
                dispatch_grads = bwd_dispatch.backward(dispatch_bwd_grads)
                _release_consumed_storage_after_last_use(
                    '4phase:dispatch_consumed_output_grads', dispatch_bwd_grads,
                    get_comm_stream(), exclude_obj=(expert_h_ln_grad, dispatch_grads))
                layer_dead_refs.extend((dispatch_bwd_grads, dispatch_grads, expert_h_ln_grad))
                dispatch_bwd_grads = None
                if self._aggressive_release_debug:
                    expert_grads = None
                    self._release_debug_probe('early_dispatch_inputs')
                self._sync_comm_to_comp()
                _mem_probe('4phase:after_dispatch_bwd_early')

                grad_h_ln_total = self._add_optional_grad(
                    dispatch_grads[0], expert_h_ln_grad)
                attn_grads = self._attn_backward_grads(
                    bwd_attn,
                    live_residual_grad,
                    grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)
                _release_consumed_storage_after_last_use(
                    '4phase:attn_consumed_output_grads',
                    (attn_grads, dispatch_grads, expert_h_ln_grad, live_residual_grad),
                    get_comp_stream(), exclude_obj=grad_x)
                layer_dead_refs.extend((live_residual_grad, grad_h_ln_total, attn_grads))
                _release_layer_dead_storage_after_backward(
                    '4phase:dead_after_layer_backward', layer_dead_refs,
                    get_comp_stream(), exclude_obj=(
                        grad_x, fwd_nodes, fwd_lc.step_data,
                        fwd_h_residual, fwd_h_ln, fwd_routing_probs,
                        fwd_dispatch_out))
                if self._aggressive_release_debug:
                    dispatch_grads = None
                    grad_h_ln_total = None
                    expert_h_ln_grad = None
                    attn_grads = None
                    live_residual_grad = None
                    combine_grads_for_attn = None
                    self._release_debug_probe('early_attn_grads')
                _mem_probe('4phase:after_attn_bwd_early')

                fwd_expert_result = fwd_expert.forward(
                    self._expert_forward_inputs(
                        fwd_lc, fwd_expert, fwd_dispatch_out, fwd_h_ln))
                fwd_expert_out, fwd_shared_expert_out = self._prepare_state_shared_expert(
                    fwd_lc, fwd_expert, fwd_expert_result)
                fwd_expert_result = None
                if self._aggressive_release_debug:
                    fwd_dispatch_out = None
                    self._release_debug_probe('early_expert_inputs')
                fwd_combine.uses_state_residual = fwd_lc.step_data.get('_use_state_residual', False)
                _mem_probe('4phase:after_phase3')
                _phase_mem_probe('4phase:after_phase3')
                self._sync_4phase_boundary_for_debug('phase3')

                self._cross_stream_barrier(slot=2)
                _mem_probe('4phase:after_barrier2')

                fwd_h_out = fwd_combine.forward(
                    self._combine_forward_inputs(
                        fwd_lc, fwd_expert_out, fwd_h_residual, fwd_shared_expert_out))
                self._drop_after_combine_forward(fwd_lc, '4phase:after_combine_forward')
                if self._aggressive_release_debug:
                    fwd_expert_out = None
                    fwd_shared_expert_out = None
                    fwd_h_residual = None
                    self._release_debug_probe('early_combine_refs')
            else:
                # Phase 3: COMM(b_dispatch) || COMP(f_expert)
                # expert forward includes shared expert (takes h_ln, returns tuple).
                dispatch_bwd_grads = self._dispatch_backward_grads(
                    bwd_dispatch, expert_grads)
                expert_h_ln_grad = self._expert_h_ln_grad(bwd_expert, expert_grads)
                if pool is not None:
                    fut_dispatch_bwd = pool.submit(
                        bwd_dispatch.backward, dispatch_bwd_grads)
                    fwd_expert_result = fwd_expert.forward(
                        self._expert_forward_inputs(
                            fwd_lc, fwd_expert, fwd_dispatch_out, fwd_h_ln))
                    dispatch_grads = fut_dispatch_bwd.result()
                else:
                    dispatch_grads = bwd_dispatch.backward(dispatch_bwd_grads)
                    fwd_expert_result = fwd_expert.forward(
                        self._expert_forward_inputs(
                            fwd_lc, fwd_expert, fwd_dispatch_out, fwd_h_ln))
                _release_consumed_storage_after_last_use(
                    '4phase:dispatch_consumed_output_grads', dispatch_bwd_grads,
                    get_comm_stream(), exclude_obj=(expert_h_ln_grad, dispatch_grads))
                layer_dead_refs.extend((dispatch_bwd_grads, dispatch_grads, expert_h_ln_grad))
                dispatch_bwd_grads = None
                if self._aggressive_release_debug:
                    expert_grads = None
                    fwd_dispatch_out = None
                    self._release_debug_probe('phase3_dispatch_expert_inputs')
                fwd_expert_out, fwd_shared_expert_out = self._prepare_state_shared_expert(
                    fwd_lc, fwd_expert, fwd_expert_result)
                fwd_expert_result = None
                fwd_combine.uses_state_residual = fwd_lc.step_data.get('_use_state_residual', False)
                _mem_probe('4phase:after_phase3')
                _phase_mem_probe('4phase:after_phase3')
                self._sync_4phase_boundary_for_debug('phase3')

                # Phase boundary: Phase 4 COMP needs COMM output (dispatch_grads),
                # Phase 4 COMM needs COMP output (expert_out + shared_expert_out)
                self._cross_stream_barrier(slot=2)
                _mem_probe('4phase:after_barrier2')

                # Phase 4: COMM(f_combine) || COMP(b_attn)
                if pool is not None:
                    fut_combine_fwd = pool.submit(
                        fwd_combine.forward,
                        self._combine_forward_inputs(
                            fwd_lc, fwd_expert_out, fwd_h_residual, fwd_shared_expert_out))
                    grad_h_ln_total = self._add_optional_grad(
                        dispatch_grads[0], expert_h_ln_grad)
                    attn_grads = self._attn_backward_grads(
                        bwd_attn,
                        live_residual_grad,
                        grad_h_ln_total, dispatch_grads[1])
                    grad_x = bwd_attn.backward(attn_grads)
                    fwd_h_out = fut_combine_fwd.result()
                else:
                    fwd_h_out = fwd_combine.forward(
                        self._combine_forward_inputs(
                            fwd_lc, fwd_expert_out, fwd_h_residual, fwd_shared_expert_out))
                    grad_h_ln_total = self._add_optional_grad(
                        dispatch_grads[0], expert_h_ln_grad)
                    attn_grads = self._attn_backward_grads(
                        bwd_attn,
                        live_residual_grad,
                        grad_h_ln_total, dispatch_grads[1])
                    grad_x = bwd_attn.backward(attn_grads)
                _release_consumed_storage_after_last_use(
                    '4phase:attn_consumed_output_grads',
                    (attn_grads, dispatch_grads, expert_h_ln_grad, live_residual_grad),
                    get_comp_stream(), exclude_obj=grad_x)
                layer_dead_refs.extend((live_residual_grad, grad_h_ln_total, attn_grads))
                _release_layer_dead_storage_after_backward(
                    '4phase:dead_after_layer_backward', layer_dead_refs,
                    get_comp_stream(), exclude_obj=(
                        grad_x, fwd_nodes, fwd_lc.step_data,
                        fwd_h_residual, fwd_h_ln, fwd_routing_probs,
                        fwd_dispatch_out, fwd_expert_out,
                        fwd_shared_expert_out, fwd_h_out))
                self._drop_after_combine_forward(fwd_lc, '4phase:after_combine_forward')
                if self._aggressive_release_debug:
                    dispatch_grads = None
                    grad_h_ln_total = None
                    expert_h_ln_grad = None
                    attn_grads = None
                    live_residual_grad = None
                    combine_grads_for_attn = None
                    fwd_expert_out = None
                    fwd_shared_expert_out = None
                    fwd_h_residual = None
                    self._release_debug_probe('phase4_refs')
            _mem_probe('4phase:after_phase4')
            _phase_mem_probe('4phase:after_phase4')
            self._sync_4phase_boundary_for_debug('phase4')

            if dispatch_grads is not None:
                _sync_lifetime_to_creation_stream((dispatch_grads[0],), get_comp_stream())
            del (combine_grads, combine_grads_for_attn, expert_grads,
                 dispatch_bwd_grads, dispatch_grads, expert_h_ln_grad,
                 live_residual_grad, grad_h_ln_total, attn_grads,
                 layer_dead_refs)
            _mem_probe('4phase:after_del_grads')

            for n in fwd_nodes:
                if n.checkpoint:
                    n.output = None

            # Sync COMM→COMP: fwd_combine ran on COMM stream producing fwd_h_out.
            # The next merged step's fwd_attn will consume fwd_h_out on COMP stream,
            # so COMP must wait for COMM to finish.
            self._sync_comm_to_comp()
            _mem_probe('4phase:end')

        # Restore event protocol for sequential backward use (cooldown)
        for n in fwd_nodes:
            n._skip_event = False

        return fwd_h_out, grad_x, ('layer4', fwd_nodes)

    def _merged_step_general(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h):
        """Dispatch to 4-phase or 2-phase merged step based on layer types."""
        bwd_tag, bwd_nodes = bwd_entry

        if self._use_4node and bwd_tag == 'layer4' and fwd_lc.is_moe:
            return self._merged_step_4phase(bwd_nodes, fwd_lc, fwd_event, grad_h, fwd_h)
        elif bwd_tag == 'layer2':
            if fwd_lc.is_moe:
                raise NotImplementedError(
                    "Merging dense backward with MoE forward is unsupported. "
                    "The MoE forward must run through the 4-node schedule.")
            return self._merged_step(bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)
        else:
            with nnscaler.sync_grad_when(False):
                grad_x = self._backward_entry(bwd_entry, grad_h)
            if fwd_lc.is_moe and self._use_4node:
                fwd_h_out, fwd_entry = self._forward_single_layer_4node(fwd_h, fwd_lc, fwd_event)
            else:
                fwd_h_out, fwd_entry = self._forward_single_layer(fwd_h, fwd_lc, fwd_event)
            return fwd_h_out, grad_x, fwd_entry
