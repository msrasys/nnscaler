# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Checkpoint boundaries compatible with pipeline output pseudo-free."""

import torch
from inspect import Parameter, signature
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_CONTROL_KEYWORDS = {
    name for name, parameter in signature(torch_checkpoint).parameters.items()
    if parameter.kind is Parameter.KEYWORD_ONLY
} | {'preserve_rng_state'}


def checkpoint(function, *args, use_reentrant=False, **kwargs):
    """Preserve saved input metadata when pipeline outputs are pseudo-freed.

    Saved-only views have independent TensorImpls and shared version counters.
    Unlike detached grad leaves, they do not pin offloaded storage through an
    AccumulateGrad node. The first forward uses original arguments to preserve
    its gradient graph and hooks; replay uses the saved aliases.
    """
    if use_reentrant is not False:
        raise ValueError('NNScaler checkpoint requires use_reentrant=False')
    controls = {name: kwargs.pop(name) for name in _CONTROL_KEYWORDS if name in kwargs}
    originals = [(args, kwargs)]
    flat, structure = tree_flatten((args, kwargs))
    aliases, saved_args, positions = {}, [], []
    for value in flat:
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in aliases:
                aliases[key] = len(saved_args)
                snapshot = (value.view_as(value)
                            if value.layout == torch.strided and not value.is_leaf
                            and value._base is None and value.numel() > 1 else value)
                saved_args.append(snapshot)
            positions.append(aliases[key])
        else:
            positions.append(len(saved_args))
            saved_args.append(value)

    def invoke(*saved):
        active_args, active_kwargs = (originals.pop() if originals else
                                      tree_unflatten([saved[index] for index in positions], structure))
        return function(*active_args, **active_kwargs)

    return torch_checkpoint(invoke, *saved_args, use_reentrant=False, **controls)
