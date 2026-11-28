# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rebuild the saved tensors of a checkpoint's final pair of pure operators."""
from contextlib import nullcontext
from threading import local

import torch
from torch.utils.checkpoint import checkpoint


_tails = {}


def register_checkpoint_tail(producer, consumer, *, input_index=0, rebuild):
    """Register a saved-state implementation for the terminal edge A -> B.

    rebuild(a_args, a_kwargs, b_args, b_kwargs) receives None at B's positional
    input_index, where the unused replay output of A would normally go. It must
    restore all tensors saved by A and B in their original order, using the
    same values, and return True. A callback can decline an unsupported input
    by returning False *before* doing work or saving anything. The usual pair
    then executes. Both operators must be pure and use ordinary call emission.
    """
    if not all(callable(fn) for fn in (producer, consumer, rebuild)):
        raise TypeError('Checkpoint tail operators and rebuild must be callable')
    if type(input_index) is not int or input_index < 0:
        raise ValueError('input_index must be a nonnegative positional argument index')
    key = tuple(f'{op.__module__}.{op.__name__}' for op in (producer, consumer))
    existing = _tails.get(key)
    if existing is not None and existing != (input_index, rebuild):
        raise ValueError(f'Checkpoint tail already registered for {key}')
    _tails[key] = (input_index, rebuild)


def registered_checkpoint_tails():
    # Codegen workers only need the IR signatures and the proven edge.
    return {key: value[0] for key, value in _tails.items()}


def rebuild(key, index, producer_args, producer_kwargs, consumer_args, consumer_kwargs):
    entry = _tails.get(key)
    if entry is None or entry[0] != index:
        return False
    result = entry[1](producer_args, producer_kwargs, consumer_args, consumer_kwargs)
    if type(result) is not bool:
        raise TypeError('A checkpoint tail rebuild must return a boolean')
    return result


class _Save(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor, tensors):
        ctx.save_for_backward(*tensors)
        return anchor

    @staticmethod
    def backward(ctx, *gradients):
        raise RuntimeError('A checkpoint saved-state marker cannot be differentiated')


def checkpoint_save_for_backward(anchor, *tensors):
    """Pack real backward operands during replay; torch may early-stop here."""
    if not isinstance(anchor, torch.Tensor) or not anchor.requires_grad or not torch.is_grad_enabled():
        raise ValueError('Checkpoint state requires grad mode and a differentiable anchor')
    _Save.apply(anchor, tensors)


class _ReplayContext:
    def __init__(self, context, state):
        self.context, self.state = context, state

    def __enter__(self):
        self.context.__enter__()
        self.state.depth = getattr(self.state, 'depth', 0) + 1

    def __exit__(self, *exc):
        self.state.depth -= 1
        return self.context.__exit__(*exc)


def run(function, *args, context_fn=None):
    """Run one non-reentrant checkpoint with an explicit invocation-local phase."""
    state = local()

    def contexts():
        forward, replay = context_fn() if context_fn else (nullcontext(), nullcontext())
        return forward, _ReplayContext(replay, state)

    def invoke(*inputs):
        return function(*inputs, _rebuilding=bool(getattr(state, 'depth', 0)))

    return checkpoint(invoke, *args, use_reentrant=False, context_fn=contexts)
