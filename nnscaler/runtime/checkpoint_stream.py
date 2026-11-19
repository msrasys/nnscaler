# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run checkpoint recomputation without waiting for the incoming gradient.

The original forward's completion event protects saved inputs. Every backward
stream consuming a reconstructed saved tensor joins the replay and protects the
tensor's storage, including consumers other than the one that triggered replay.
"""

from contextlib import ExitStack, nullcontext
from threading import local

import torch

from .stream import _get_module_stream, input_stream_lifetime, hold_owners_through_backward


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


class _Forward:
    def __init__(self, state):
        self.state = state
        self.hooks = None

    def __enter__(self):
        self.state.forward_stream = torch.cuda.current_stream(self.state.device)
        top = getattr(torch._C._autograd, '_top_saved_tensors_default_hooks', None)
        hooks = top(False) if top is not None else None
        if hooks is None:
            raise RuntimeError('Checkpoint stream context must run inside a non-reentrant checkpoint')
        original_pack, original_unpack = hooks
        state = self.state  # Do not retain this context in its own hook closure.

        def unpack(packed):
            gid = torch._C._current_graph_task_id()
            tensor = original_unpack(packed)
            if tensor.device.type == 'cuda':
                state.consume(tensor, gid)
            return tensor

        self.hooks = torch.autograd.graph.saved_tensors_hooks(original_pack, unpack)
        return self.hooks.__enter__()

    def __exit__(self, *exc):
        try:
            return self.hooks.__exit__(*exc)
        finally:
            self.hooks = None
            self.state.ready = torch.cuda.Event()
            self.state.ready.record(self.state.forward_stream)


class _Recompute:
    def __init__(self, state):
        self.state = state
        self.local = local()

    def __enter__(self):
        top = getattr(torch._C._autograd, '_top_saved_tensors_default_hooks', None)
        if top is None:
            raise RuntimeError('Checkpoint streams require PyTorch saved-tensor hook inspection support')
        hooks = top(False)
        if hooks is None or self.state.ready is None:
            raise RuntimeError('Recompute stream context must run inside a non-reentrant checkpoint')
        caller = torch.cuda.current_stream(self.state.device)
        stream = _get_module_stream(caller.device, self.state.name)
        stream.wait_event(self.state.ready)
        if self.state.backward_ready is not None:
            if not self.state.backward_ready.captured:
                raise RuntimeError('The recompute readiness backend did not capture its launch point')
            self.state.backward_ready.wait(stream)
        original_pack, original_unpack = hooks

        def pack(tensor):
            # Do this before the checkpoint's pack hook, which can raise its
            # intentional early-stop exception after saving the final tensor.
            if tensor.device.type == 'cuda':
                tensor.record_stream(caller)
            return original_pack(tensor)

        stack = ExitStack()
        try:
            stack.enter_context(torch.cuda.stream(stream))
            owners = stack.enter_context(input_stream_lifetime(self.state.inputs, True))
            hold_owners_through_backward(owners, (caller, stream))
            stack.enter_context(torch.autograd.graph.saved_tensors_hooks(pack, original_unpack))
        except BaseException:
            stack.close()
            raise
        if not hasattr(self.local, 'entries'):
            self.local.entries = []
        self.local.entries.append((caller, stream, stack))

    def __exit__(self, *exc):
        caller, stream, stack = self.local.entries.pop()
        try:
            return stack.__exit__(*exc)
        finally:
            done = torch.cuda.Event()
            done.record(stream)
            caller.wait_event(done)
            self.state.replayed(torch._C._current_graph_task_id(), done, caller)


class _State:
    def __init__(self, name, inputs, device, backward_ready):
        self.name, self.inputs, self.device = name, inputs, device
        self.ready = None
        self.forward_stream = None
        self.backward_ready = backward_ready
        self.replays = {}
        self.cleanup_tasks = set()

    def replayed(self, gid, done, caller):
        self.replays[gid] = (done, caller, {caller.cuda_stream})
        if gid >= 0 and gid not in self.cleanup_tasks:
            self.cleanup_tasks.add(gid)

            def cleanup():
                self.replays.pop(gid, None)
                self.cleanup_tasks.discard(gid)

            torch.autograd.Variable._execution_engine.queue_callback(cleanup)

    def consume(self, tensor, gid):
        done, first, joined = self.replays[gid]
        caller = torch.cuda.current_stream(tensor.device)
        if caller.cuda_stream not in joined:
            caller.wait_event(done)
            joined.add(caller.cuda_stream)
        if caller != first:
            # Packing already protects the triggering stream. A different
            # consumer also needs allocator protection after its Python
            # backward returns while its CUDA reads are still queued.
            tensor.record_stream(caller)


def contexts(name, inputs, *, backward_ready=None):
    devices = {tensor.device for tensor in _tensors(inputs) if tensor.device.type == 'cuda'}
    if not devices:
        return nullcontext(), nullcontext()
    if len(devices) != 1:
        raise ValueError('Checkpoint streams support one CUDA device per region')
    device = next(iter(devices))
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Checkpoint stream contexts do not support CUDA graph capture')
    state = _State(name, inputs, device, backward_ready)
    return _Forward(state), _Recompute(state)
