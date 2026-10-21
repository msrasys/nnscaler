# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create communication groups in deterministic, non-overlapping rounds."""
import hashlib
import struct
import sys

import torch
import torch.distributed as dist


def _split_color(ranks):
    """PyTorch 2.9--2.11's native genNcclSplitColor on 64-bit Linux.

    The public split_group uses this C++ hash, not distributed_c10d's Python
    _process_group_color. Equal colors in one split merge unrelated groups.
    Keep the supported-version check in _split_context in sync with this hash.
    """
    value = 0
    for rank in ranks:
        value ^= rank + 0x9e3779b9 + (value << 6) + (value >> 2)
        value &= (1 << 64) - 1
    if value >= 1 << 63:
        value -= 1 << 64
    return abs(value) % ((1 << 31) - 1)


def partition_groups(groups):
    """Pack groups into rounds with disjoint ranks and distinct split colors."""
    batches, occupied, colors = [], [], []
    for ranks in sorted(set(groups), key=lambda ranks: (-len(ranks), ranks)):
        members = set(ranks)
        color = _split_color(ranks)
        for batch, used, used_colors in zip(batches, occupied, colors):
            if members.isdisjoint(used) and color not in used_colors:
                batch.append(ranks)
                used.update(members)
                used_colors.add(color)
                break
        else:
            batches.append([ranks])
            occupied.append(members)
            colors.append({color})
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
    # Unknown native color implementations must use individually created groups.
    supported = (supported and sys.platform == 'linux' and struct.calcsize('P') == 8
                 and (2, 9) <= torch.__version__ < (2, 12))
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
