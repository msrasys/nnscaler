# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run checkpoint recomputation without waiting for the incoming gradient.

The original forward's completion event protects saved inputs. Recomputed
tensors are recorded on the original backward stream as they are packed, and
that stream joins recomputation before the checkpoint supplies them to backward.
"""

from contextlib import ExitStack, nullcontext
from threading import local

import torch

from .stream import _get_module_stream, input_stream_lifetime


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

    def __enter__(self):
        self.state.forward_stream = torch.cuda.current_stream(self.state.device)

    def __exit__(self, *exc):
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
            stack.enter_context(input_stream_lifetime(self.state.inputs, True))
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


class _State:
    def __init__(self, name, inputs, device):
        self.name, self.inputs, self.device = name, inputs, device
        self.ready = None
        self.forward_stream = None


def contexts(name, inputs):
    devices = {tensor.device for tensor in _tensors(inputs) if tensor.device.type == 'cuda'}
    if not devices:
        return nullcontext(), nullcontext()
    if len(devices) != 1:
        raise ValueError('Checkpoint streams support one CUDA device per region')
    device = next(iter(devices))
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Checkpoint stream contexts do not support CUDA graph capture')
    state = _State(name, inputs, device)
    return _Forward(state), _Recompute(state)
