# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create disjoint NCCL subgroups in shared split rounds."""
import hashlib
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d


# Multicolor split costs more than a short sequence of ordinary splits on a
# single node. This conservative threshold reserves batching for large fan-out;
# it is a policy, not a measured crossover for every NCCL version or topology.
_MIN_GROUPS_PER_ROUND = 32


def partition_groups(groups):
    """Deterministic first-fit packing; members cannot overlap within a round."""
    batches, occupied = [], []
    for ranks in sorted(set(groups), key=lambda ranks: (-len(ranks), ranks)):
        members = set(ranks)
        for batch, used in zip(batches, occupied):
            if members.isdisjoint(used):
                batch.append(ranks)
                used.update(members)
                break
        else:
            batches.append([ranks])
            occupied.append(members)
    return batches


def _split_device():
    """Unbound, non-NCCL and older runtimes retain ordinary new_group calls."""
    if not callable(getattr(c10d, '_process_group_color', None)) or not dist.is_initialized():
        return None
    parent = dist.group.WORLD
    device = getattr(parent, 'bound_device_id', None)
    if device is None or dist.get_backend(parent) != 'nccl':
        return None
    backend = parent._get_backend(device)
    if not getattr(backend, 'supports_splitting', False):
        return None
    if not hasattr(getattr(c10d, '_world', None), 'pg_names'):
        return None
    return device


def create_groups(groups, *, rank, world_size, timeout, auxiliary_groups):
    """All world ranks must provide the same list and call in the same order.

    PyTorch's hashed names include the local registered-group count. Check that
    histories agree before splitting. Pad uncovered ranks with singleton groups
    so every rank registers exactly one group per round, even for P2P matchings.
    Keep padding groups alive with this DeviceGroup; they never carry traffic.
    The helper only reads PyTorch's registry and uses its public creation API.
    """
    if not groups:
        return {}
    batches = partition_groups(groups)
    device = _split_device() if len(groups) >= _MIN_GROUPS_PER_ROUND * len(batches) else None
    if device is not None:
        # NCCL colors are finite hashes; a collision would merge unrelated
        # memberships in the same split. Check padding singleton colors too.
        for batch in batches:
            covered = {member for ranks in batch for member in ranks}
            partition = batch + [(member,) for member in range(world_size) if member not in covered]
            colors = [c10d._process_group_color(list(ranks)) for ranks in partition]
            if len(set(colors)) != len(colors):
                device = None
                break
    if device is not None:
        signature = int.from_bytes(hashlib.sha256(repr(groups).encode()).digest()[:8], 'big') >> 1
        count = len(c10d._world.pg_names)
        agreement = torch.tensor([count, -count, signature, -signature], dtype=torch.int64, device=device)
        dist.all_reduce(agreement, op=dist.ReduceOp.MAX)
        maximum, negative_minimum, signature_max, signature_negative_min = agreement.tolist()
        if signature_max != -signature_negative_min:
            raise ValueError('Ranks supplied different pending communication groups')
        if maximum != -negative_minimum:
            # External groups may have left different per-rank histories. The
            # normal globally ordered API uses a shared counter instead.
            device = None
    if device is None:
        return {ranks: dist.new_group(list(ranks), timeout=timeout) for ranks in groups}

    result = {}
    for batch in batches:
        covered = {member for ranks in batch for member in ranks}
        members = next((ranks for ranks in batch if rank in ranks), (rank,))
        # All world ranks enter once per round, each with its own disjoint
        # membership (or singleton padding). The bound NCCL default group
        # makes new_group perform one shared, multicolor ncclCommSplit.
        group = dist.new_group(list(members), timeout=timeout, use_local_synchronization=True)
        if rank not in covered:
            auxiliary_groups.append(group)
        for ranks in batch:
            result[ranks] = group if rank in ranks else dist.GroupMember.NON_GROUP_MEMBER
    return result
