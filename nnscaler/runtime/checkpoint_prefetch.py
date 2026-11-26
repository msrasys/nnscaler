# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Eager checkpoint replay at an independent operation's launch boundary.

Uses the installed non-reentrant checkpoint hook contract. No work is
borrowed from another forward invocation. GPU users join a forward prefetch at
the end of its segment, and all cached values retain normal allocator tracking.
"""
from contextlib import contextmanager
from threading import local, Lock, RLock
import weakref

import torch
import torch.utils.checkpoint as ckpt
from torch.multiprocessing.reductions import StorageWeakRef

from .backward import BackwardLaunchPoint
from .stream import _get_module_stream, input_stream_lifetime, hold_owners_through_backward

_budgets = {}
_budgets_lock = RLock()


def device_budget(device, limit):
    with _budgets_lock:
        budget = _budgets.setdefault(device, Budget(limit))
        budget.poll()
        if budget.limit != limit:
            if budget.used:
                raise ValueError('Live checkpoint prefetches on a device must use the same budget')
            budget.limit = limit
        return budget


def tensors(values):
    pending, seen = [values], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
        elif type(value) is dict:
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)


def frame_from_hook(hook):
    for cell in getattr(hook, '__closure__', None) or ():
        value = cell.cell_contents
        if isinstance(value, ckpt._CheckpointFrame):
            return value
    raise RuntimeError('Prefetch requires the installed non-reentrant checkpoint hook')


class Budget:
    def __init__(self, limit):
        self.limit, self.used, self.peak = limit, 0, 0
        self.pending = []
        self.lock = RLock()

    def poll(self, wait=False):
        with self.lock:
            remaining = []
            for event, size in self.pending:
                if wait:
                    event.synchronize()
                if wait or event.query():
                    self.used -= size
                else:
                    remaining.append((event, size))
            self.pending = remaining

    def reserve(self, size):
        with self.lock:
            self.poll()
            if self.used + size > self.limit:
                return None
            self.used += size
            self.peak = max(self.peak, self.used)
            return Lease(self, size)


class Lease:
    def __init__(self, budget, size):
        self.budget, self.size, self.event = budget, size, None
        self.finalizer = None

    def release(self, event=None):
        with self.budget.lock:
            if self.finalizer is not None:
                self.finalizer.detach()
                self.finalizer = None
            if not self.size:
                return
            event = event or self.event
            if event is None:
                self.budget.used -= self.size
            else:
                self.budget.pending.append((event, self.size))
            self.size = 0

    def __del__(self):
        self.release()


class Prefetch(BackwardLaunchPoint):
    def __init__(self, stream, budget):
        super().__init__(())
        self.name, self.budget = stream, budget
        self.frame_ref = None
        self.token = object()
        self.prefetched = False
        self.cuda = False
        self.forward_done = self.done = self.lease = None
        self.replaying = False
        self.gids = set()
        self.validated_gids = set()
        self.active_leases = {}
        self.joined = set()
        self.opaque = []
        self.stats = dict(launched=0, budget_skips=0, reused=0)
        self.supported = True
        self.lock = Lock()
        self.prepared_args = None
        self.prepared_versions = ()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None and not self.captured:
            raise RuntimeError('The checkpoint prefetch backend did not capture its launch point')

    def contexts(self, inputs):
        if self.frame_ref is not None:
            raise RuntimeError('A checkpoint prefetch point belongs to one forward invocation')
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
        if hooks is not None:
            try:
                parent = frame_from_hook(hooks[0])
            except RuntimeError:
                parent = None
            if parent is not None:
                raise RuntimeError('Checkpoint prefetch does not support nested checkpoints')
        devices = {value.device for value in tensors(inputs) if value.is_cuda}
        if len(devices) > 1:
            raise ValueError('Prefetch requires one device')
        self.device = next(iter(devices)) if devices else torch.device('cpu')
        self.cuda = self.device.type == 'cuda'
        self.inputs = inputs
        if isinstance(self.budget, int):
            self.budget = device_budget(self.device, self.budget)
        if self.cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Checkpoint prefetch requires eager execution')
        return self.forward_context(), Reusable(self.recompute_context)

    def record_ready(self):
        if self.cuda:
            super().record_ready()
        else:
            self.ready = True

    @contextmanager
    def forward_context(self):
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
        if hooks is None:
            raise RuntimeError('Missing checkpoint hooks')
        pack, unpack = hooks
        frame = frame_from_hook(pack)
        self.frame_ref = weakref.ref(frame)
        self.estimate = 0
        input_storage = set()
        saved_storage = {}
        for value in tensors(self.inputs):
            try:
                input_storage.add(value.untyped_storage()._cdata)
            except (RuntimeError, NotImplementedError):
                self.supported = False
        self.input_storage = input_storage
        self.forward_stream = torch.cuda.current_stream(self.device) if self.cuda else None

        def capture(value):
            # Weak storage references preserve identity without pinning GPU
            # allocations. Pointer reuse during the original forward must not
            # cause distinct activations to be mistaken for one allocation.
            try:
                storage = value.untyped_storage()
                if value.device == self.device and storage._cdata not in input_storage:
                    key = storage._cdata
                    if key not in saved_storage:
                        saved_storage[key] = StorageWeakRef(storage)
                        self.estimate += storage.nbytes()
            except (RuntimeError, NotImplementedError):
                self.supported = False
            return pack(value)

        def consume(holder):
            if torch._C._current_graph_task_id() < 0:
                return unpack(holder)  # Ordinary debug inspection outside backward.
            gid = self.graph_id()
            if torch.is_grad_enabled():
                if self.prefetched:
                    self.clear(frame, self.token)
                    self.prefetched = False
                    self.lease.release(self.done)
                    self.lease = None
                raise RuntimeError('Checkpoint prefetch supports first-order gradients only')
            if self.gids and gid not in self.gids:
                raise RuntimeError('Concurrent backward of one prefetched checkpoint is unsupported')
            if (self.prefetched or gid in self.gids) and gid not in self.validated_gids:
                try:
                    self.validate_input_versions()
                    # Ordinary checkpoint replay checks these versions at its
                    # first backward unpack. Prefetch ran earlier, so retain
                    # that check when its cache is actually consumed.
                    frame.input_saver.grad_fn.saved_tensors
                except BaseException:
                    key = self.token if self.prefetched else gid
                    self.clear(frame, key)
                    if self.prefetched:
                        lease, self.lease = self.lease, None
                        self.prefetched = False
                    else:
                        lease = self.active_leases.pop(gid, None)
                        self.gids.discard(gid)
                    if lease is not None:
                        lease.release(self.done)
                    raise
                self.validated_gids.add(gid)
            if self.prefetched:
                self.adopt(gid)
            if gid in self.gids:
                if self.cuda:
                    caller = torch.cuda.current_stream(self.device)
                    if (gid, caller.cuda_stream) not in self.joined:
                        caller.wait_event(self.done)
                        self.joined.add((gid, caller.cuda_stream))
                value = unpack(holder)
                if self.cuda and value.is_cuda:
                    value.record_stream(torch.cuda.current_stream(self.device))
                return value
            return unpack(holder)

        with torch.autograd.graph.saved_tensors_hooks(capture, consume):
            yield
        saved_storage.clear()
        if self.cuda:
            self.forward_done = torch.cuda.Event()
            self.forward_done.record(self.forward_stream)
        ctx = frame.input_saver.grad_fn
        self.prepared_args = ctx.get_args(ctx.saved_tensors)
        self.prepared_versions = tuple((value, value._version) for value in tensors(self.inputs))

    @staticmethod
    def graph_id():
        group_type = getattr(ckpt, 'GraphExecGroup', None)
        group = group_type._get_current_group() if group_type is not None else None
        if group is not None:
            raise RuntimeError('Checkpoint prefetch does not support GraphExecGroup')
        return torch._C._current_graph_task_id()

    @contextmanager
    def recompute_context(self):
        if not self.cuda or not self.replaying:
            yield
            return
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
        pack, unpack = hooks
        caller = torch.cuda.current_stream(self.device)
        stream = caller if self.name == 'default' else _get_module_stream(self.device, self.name)
        stream.wait_event(self.forward_done)
        if self.replaying and self.ready is not None:
            stream.wait_event(self.ready)

        def tracked(value):
            if value.is_cuda:
                value.record_stream(caller)
            return pack(value)

        with torch.cuda.stream(stream), input_stream_lifetime(self.inputs, True) as owners:
            if torch._C._current_graph_task_id() >= 0:
                hold_owners_through_backward(owners, (caller, stream))
            else:
                self.opaque.extend(owners)
            try:
                with torch.autograd.graph.saved_tensors_hooks(tracked, unpack):
                    yield
            finally:
                event = torch.cuda.Event()
                event.record(stream)
                if self.replaying:
                    self.done = event
                else:
                    caller.wait_event(event)

    def prefetch(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Concurrent replay of one checkpoint is unsupported')
        try:
            return self._prefetch()
        finally:
            self.lock.release()

    def validate_input_versions(self):
        if any(value._version != version for value, version in self.prepared_versions):
            raise RuntimeError('Checkpoint inputs were modified by an inplace operation before replay')

    def _prefetch(self):
        frame = self.frame_ref() if self.frame_ref is not None else None
        if frame is None or not frame.weak_holders or self.prefetched or not self.supported:
            return False
        if not frame.forward_completed:
            raise RuntimeError('Cannot prefetch an incomplete checkpoint forward')
        gid = self.graph_id()
        backward = gid != -1
        if backward and torch.is_grad_enabled():
            raise RuntimeError('Checkpoint prefetch supports first-order gradients only')
        if self.gids and gid not in self.gids:
            raise RuntimeError('Concurrent backward of one prefetched checkpoint is unsupported')
        if backward and frame.is_recomputed[gid]:
            return False
        key = gid if backward else self.token
        lease = self.budget.reserve(self.estimate)
        if lease is None:
            self.stats['budget_skips'] += 1
            return False
        # The cache belongs to the checkpoint frame, even when a caller keeps
        # the launch point itself for inspection after abandoning the graph.
        lease.finalizer = weakref.finalize(frame, lease.release)
        lease.finalizer.atexit = False
        self.lease = lease
        self.replaying = True
        self.done = None
        if self.cuda and not backward:
            self.ready = torch.cuda.Event()
            self.ready.record()
        try:
            if self.prepared_args is None:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
            else:
                self.validate_input_versions()
                args, self.prepared_args = self.prepared_args, None
            try:
                with ckpt._recomputation_hook(weakref.ref(frame), key), torch.enable_grad():
                    run = getattr(ckpt, '_run_fn_with_dynamo_disabled', lambda fn, *args: fn(*args))
                    run(frame.recompute_fn, *args)
            except ckpt._StopRecomputationError:
                pass
            frame.is_recomputed[key] = True
            frame.check_recomputed_tensors_match(key)
            storages = {}
            for value in frame.recomputed[key].values():
                storage = value.untyped_storage()
                if value.device == self.device and storage._cdata not in self.input_storage:
                    storages[storage._cdata] = storage.nbytes()
            actual_bytes = sum(storages.values())
            if actual_bytes > lease.size:
                raise RuntimeError('Checkpoint replay grew its saved storage footprint; prefetch budget cannot be honored')
            with self.budget.lock:
                self.budget.used -= lease.size - actual_bytes
                lease.size = actual_bytes
            self.stats['retained_bytes'] = actual_bytes
            lease.event = self.done
            self.stats['launched'] += 1
            if backward:
                self.attach_cleanup(gid)
            else:
                self.prefetched = True
        except BaseException:
            self.clear(frame, key)
            if self.cuda and self.done is not None:
                torch.cuda.current_stream(self.device).wait_event(self.done)
            lease.release(self.done)
            self.lease = None
            raise
        finally:
            self.replaying = False
        return True

    def submitted(self):
        if not self.captured or self.ready is None:
            raise RuntimeError('The checkpoint prefetch backend did not record its input readiness')
        self.prefetch()

    def adopt(self, gid):
        if gid == -1:
            raise RuntimeError('Prefetched tensors must be consumed by backward')
        frame = self.frame_ref()
        if frame.is_recomputed[gid]:
            raise RuntimeError('Checkpoint was recomputed before adopting its prefetch')
        frame.recomputed[gid] = frame.recomputed.pop(self.token)
        frame.recomp_counter[gid] = frame.recomp_counter.pop(self.token)
        frame.is_recomputed[gid] = frame.is_recomputed.pop(self.token)
        for ref in frame.weak_holders:
            holder = ref()
            if holder is not None and self.token in holder.handles:
                holder.handles[gid] = holder.handles.pop(self.token)
        self.prefetched = False
        self.attach_cleanup(gid)

    def attach_cleanup(self, gid):
        self.gids.add(gid)
        self.stats['reused'] += 1
        lease = self.lease
        self.lease = None
        # Once backward adopts the cache, GPU consumers can outlive the
        # checkpoint frame. Only GraphTask completion may release this lease.
        if lease.finalizer is not None:
            lease.finalizer.detach()
            lease.finalizer = None
        self.active_leases[gid] = lease
        if self.cuda:
            hold_owners_through_backward(self.opaque, (torch.cuda.current_stream(self.device),
                                                       (torch.cuda.current_stream(self.device) if self.name == 'default' else
                                                        _get_module_stream(self.device, self.name))))
        self.opaque.clear()

        def cleanup():
            lease = self.active_leases.pop(gid, None)
            if lease is None:
                return
            frame = self.frame_ref()
            if frame is not None:
                self.clear(frame, gid)
            event = None
            if self.cuda:
                caller = torch.cuda.current_stream(self.device)
                caller.wait_event(self.done)
                event = torch.cuda.Event()
                event.record(caller)
            lease.release(event)
            self.gids.discard(gid)
            self.validated_gids.discard(gid)
            self.joined = {key for key in self.joined if key[0] != gid}

        torch.autograd.Variable._execution_engine.queue_callback(cleanup)

    @staticmethod
    def clear(frame, gid):
        frame.recomputed.pop(gid, None)
        frame.recomp_counter.pop(gid, None)
        frame.is_recomputed.pop(gid, None)
        for ref in frame.weak_holders:
            holder = ref()
            if holder is not None:
                holder.handles.pop(gid, None)

    def finish_forward(self):
        if self.cuda and self.done is not None:
            torch.cuda.current_stream(self.device).wait_event(self.done)


class Reusable:
    def __init__(self, factory):
        self.factory, self.local = factory, local()

    def __enter__(self):
        if not hasattr(self.local, 'stack'):
            self.local.stack = []
        context = self.factory()
        result = context.__enter__()
        self.local.stack.append(context)
        return result

    def __exit__(self, *exc):
        return self.local.stack.pop().__exit__(*exc)


@contextmanager
def prefetch_scopes():
    """Join this forward invocation's prefetched work, including exception exits."""
    points = []
    try:
        yield points
    finally:
        for point in points:
            point.finish_forward()
