# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Member-only NCCL groups with deterministic rendezvous names.

PyTorch's public local-synchronization naming includes the number of locally
registered groups in some versions. That count differs across PP stages.
Keep the compatibility bridge here rather than monkey-patching PyTorch globals.
"""
import hashlib
import inspect
from weakref import WeakKeyDictionary

import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

_generations = WeakKeyDictionary()


def supports_local_groups():
    """Use the public collective path for unsupported or device-bound worlds."""
    helper = getattr(c10d, '_new_process_group_helper', None)
    world = getattr(c10d, '_world', None)
    required = {'group_size', 'group_rank', 'global_ranks_in_group', 'backend',
                'store', 'group_name', 'timeout'}
    try:
        parameters = inspect.signature(helper).parameters if helper is not None else {}
        if (not required.issubset(parameters)
                or any(p.kind == inspect.Parameter.POSITIONAL_ONLY
                       or (p.default is inspect.Parameter.empty
                           and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          inspect.Parameter.KEYWORD_ONLY)
                           and name not in required)
                       for name, p in parameters.items())
                or not all(hasattr(world, name) for name in ('pg_map', 'pg_group_ranks'))
                or not all(callable(getattr(c10d, name, None)) for name in
                           ('_is_barrier_after_init', '_store_based_barrier'))):
            return False
    except (TypeError, ValueError):
        return False
    default = c10d._get_default_group()
    return (getattr(default, 'bound_device_id', None) is None
            and dist.get_backend(default) == 'nccl')


def new_local_group(ranks, *, namespace, timeout):
    """Create one cached NNScaler group; callers must preserve global ordering.

    Collective and P2P groups use different namespaces even for equal members.
    A per-membership generation supports repeated DeviceGroup lifetimes without
    depending on unrelated groups. Members must call creations in the same order.
    No PyTorch global group counter changes.
    """
    ranks = sorted(ranks)
    if len(set(ranks)) != len(ranks) or not ranks:
        raise ValueError('Group ranks must be nonempty and unique')
    if any(rank < 0 or rank >= dist.get_world_size() for rank in ranks):
        raise ValueError('Group rank is outside the default world')
    default = c10d._get_default_group()
    if getattr(default, 'bound_device_id', None) is not None:
        raise RuntimeError('Member-only groups cannot split a device-bound default group')
    rank = dist.get_rank()
    if rank not in ranks:
        return dist.GroupMember.NON_GROUP_MEMBER

    backend, store = c10d._world.pg_map[default]
    generations = _generations.setdefault(default, {})
    key = (namespace, tuple(ranks))
    generation = generations.get(key, 0)
    generations[key] = generation + 1
    membership = ','.join(str(value) for value in ranks)
    digest = hashlib.sha256(f'{namespace}:{membership}:{generation}'.encode()).hexdigest()
    name = 'nnscaler-local-' + digest
    group, group_store = c10d._new_process_group_helper(
        group_size=len(ranks), group_rank=ranks.index(rank),
        global_ranks_in_group=ranks, backend=backend, store=store,
        group_name=name, timeout=timeout,
    )
    c10d._world.pg_group_ranks[group] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }
    if c10d._is_barrier_after_init() == 1:
        c10d._store_based_barrier(rank, group_store, name, len(ranks), timeout)
    return group
