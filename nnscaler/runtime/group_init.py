# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create communication groups in deterministic, non-overlapping rounds."""
import hashlib

import torch
import torch.distributed as dist


def partition_groups(groups):
    """Pack memberships into rounds without putting a rank in two groups."""
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


def _split_context():
    if not dist.is_initialized():
        return None
    parent = dist.group.WORLD
    device = getattr(parent, 'bound_device_id', None)
    if device is None or dist.get_backend(parent) != 'nccl':
        return None
    backend = parent._get_backend(device)
    supported = callable(getattr(dist, 'split_group', None)) and getattr(backend, 'supports_splitting', False)
    # Some PyTorch builds index an empty WORLD rank map after reinitialization.
    # Do not attempt the split: this failure is a native crash, not an exception.
    rank_map = getattr(getattr(backend, 'options', None), 'global_ranks_in_group', ())
    supported = supported and (bool(rank_map) or getattr(backend, 'uid', None) == 0)
    return device, supported


def _agree_on_split(groups):
    """Validate the shared group plan and require split support on every rank."""
    context = _split_context()
    if context is None:
        return False
    device, supported = context
    signature = int.from_bytes(hashlib.sha256(repr(groups).encode()).digest()[:8], 'big') >> 1
    # One MIN reduction computes all(supported), min(signature), -max(signature).
    # Equal minimum and maximum mean every rank supplied the same plan hash.
    agreement = torch.tensor([int(supported), signature, -signature], dtype=torch.int64, device=device)
    dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
    all_supported, min_signature, neg_max_signature = agreement.tolist()
    if min_signature != -neg_max_signature:
        raise ValueError('Ranks supplied different pending communication groups')
    return bool(all_supported)


def create_groups(groups, *, timeout):
    """All WORLD ranks declare the same memberships in the same call order.

    Single groups use ordinary new_group. Multiple disjoint groups share a
    split; excluded ranks participate without creating padding communicators.
    """
    if not groups:
        return {}
    if not _agree_on_split(groups):
        return {ranks: dist.new_group(list(ranks), timeout=timeout) for ranks in groups}

    rank = dist.get_rank()
    result = {}
    for batch in partition_groups(groups):
        if len(batch) == 1:
            ranks = batch[0]
            result[ranks] = dist.new_group(list(ranks), timeout=timeout)
        else:
            group = dist.split_group(split_ranks=[list(ranks) for ranks in batch], timeout=timeout)
            for ranks in batch:
                # None from split_group means nonmember, but None in a collective
                # means WORLD. Preserve the ordinary new_group sentinel instead.
                result[ranks] = group if rank in ranks else dist.GroupMember.NON_GROUP_MEMBER
    return result
