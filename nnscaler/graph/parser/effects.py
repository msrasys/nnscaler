# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Known checkpoint barriers. DCE retention of opaque ops is not an effect."""

import torch
from torch.fx import Node

from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRFwOperation

try:
    from torch._higher_order_ops.effects import _get_effect as _get_registered_effect
except ImportError:
    # Older PyTorch versions may not expose the effect registry.
    _get_registered_effect = None


def _contains_generator(value, seen):
    """Inspect visible values without evaluating functions or arbitrary properties."""
    if isinstance(value, torch.Generator):
        return True
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, Node):
        if value.meta.get('type') is torch.Generator:
            return True
        return any(_contains_generator(value.meta[key], seen)
                   for key in ('tensor_meta', 'val', 'example_value') if key in value.meta)
    if type(value) is IRObject:
        return not value.is_value_missing() and _contains_generator(value.value, seen)
    if isinstance(value, dict):
        # Use the builtin container methods to avoid invoking user overrides.
        return any(_contains_generator(item, seen)
                   for pair in dict.items(value) for item in pair)
    if isinstance(value, tuple):
        return any(_contains_generator(item, seen) for item in tuple.__iter__(value))
    if isinstance(value, list):
        return any(_contains_generator(item, seen) for item in list.__iter__(value))
    return False


def recompute_barriers_from_fx(node: Node, ir_node: IRFwOperation) -> tuple[str, ...]:
    """Collect known effects; unknown ops retain the caller's replay contract.

    Opaque nodes may supply ``nnscaler_recompute_barriers`` metadata. Standard
    checkpoint-managed RNG is allowed; explicit Generators need separate state.
    """
    declared = node.meta.get('nnscaler_recompute_barriers', ())
    if (not isinstance(declared, (tuple, list))
            or any(not isinstance(reason, str) or not reason for reason in declared)):
        raise TypeError('nnscaler_recompute_barriers must be a tuple/list of nonempty strings')
    barriers = list(ir_node.recompute_barriers) + list(declared)
    schema = getattr(node.target, '_schema', None)
    if schema is not None and schema.is_mutable:
        barriers.append('mutation')
    if (_get_registered_effect is not None
            and isinstance(node.target, (torch._ops.OpOverload, torch._ops.HigherOrderOperator))
            and _get_registered_effect(node.target) is not None):
        barriers.append('side_effect')
    if _contains_generator((node.args, node.kwargs, ir_node.inputs(), ir_node.kwargs), set()):
        barriers.append('external_rng')
    return tuple(dict.fromkeys(barriers))
