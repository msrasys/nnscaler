#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Communication group settings among devices
"""
from typing import List, Dict, Optional
import numpy as np
import torch
import os
import logging
import datetime

from nnscaler.flags import CompileFlag
from nnscaler.utils import is_running_distributed

_logger = logging.getLogger(__name__)
_LARGE_TIMEOUT = datetime.timedelta(seconds=21600)


def _group_batches(groups: List[List[int]], world_size: int) -> List[List[List[int]]]:
    """Deduplicate non-WORLD groups and pack them into deterministic disjoint batches."""
    batches = []
    memberships = []
    unique_groups = {
        # dedup and ignore WORLD groups
        tuple(sorted(ranks)) for ranks in groups if len(ranks) < world_size
    }
    # Order by size, then rank membership, so all singleton groups share a batch.
    for ranks in sorted(unique_groups, key=lambda ranks: (len(ranks), ranks)):
        for batch, assigned in zip(batches, memberships):
            if assigned.isdisjoint(ranks):
                batch.append(list(ranks))
                assigned.update(ranks)
                break
        else:
            batches.append([list(ranks)])
            memberships.append(set(ranks))
    return batches


class _DeviceGroup:
    def __init__(self):
        self._is_pg_initer = False
        if CompileFlag.dev_mode or not is_running_distributed():
            self.rank = 0
            self.world_size = 1
            self.local_world_size = 1
            self.local_rank = 0
            self.node_rank = 0
        else:
            if not torch.distributed.is_initialized():
                if torch.__version__ >= (2, 3):
                    torch.distributed.init_process_group(
                        backend='nccl', timeout=_LARGE_TIMEOUT,
                        device_id=int(os.environ.get('LOCAL_RANK')),
                    )
                else:
                    torch.distributed.init_process_group(
                        backend='nccl', timeout=_LARGE_TIMEOUT,
                    )
                self._is_pg_initer = True

            # disable it for now due to connection refused error when nnodes > 1
            # TODO: investigate the root cause
            # create a barrier group for synchronization
            # it is OK even the user has already created this gloo group
            # this new timeout will override the old one.
            # self.barrier_gloo_group = torch.distributed.new_group(
            #     backend='gloo', timeout=_LARGE_TIMEOUT
            # )

            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            # assume each node has the same device number
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))
            self.local_rank = int(os.environ.get('LOCAL_RANK'))
            self.node_rank = int(os.environ.get('GROUP_RANK'))

        torch.cuda.set_device(self.local_rank)
        self.groups: Dict = { '1'*self.world_size: None }
        self.streams: Dict[str, torch.cuda.Stream] = {
            'default': torch.cuda.default_stream()}
        self.events: Dict[str, torch.cuda.Event] = {}

    def close(self):
        if self._is_pg_initer:
            torch.distributed.destroy_process_group()
            self._is_pg_initer = False

    def group_exists(self, ranks):
        """
        Check if group exists
        """
        rank_bits = self.bitmap(ranks)
        return rank_bits in self.groups

    def get_group(self, ranks):
        """
        Create and return rank groups on-demand

        None will be returned if length of ranks are equal to world size
        """
        if len(ranks) == self.world_size:
            return None
        rank_bits = self.bitmap(ranks)
        if rank_bits not in self.groups:
            self.groups[rank_bits] = torch.distributed.new_group(
                list(ranks), timeout=_LARGE_TIMEOUT)
        return self.groups[rank_bits]

    def get_groups(self, ranks: List[List[int]]):
        """Create groups in disjoint batches and return them in input order.

        All WORLD ranks must call with the same memberships. Cached groups are
        reused; WORLD returns None and nonmembers follow get_group's behavior.
        """
        self._init_groups(_group_batches(ranks, self.world_size))
        return [self.get_group(members) for members in ranks]

    def _new_groups(self, ranks: List[List[int]]):
        # split_group was introduced in PyTorch 2.5 and requires a bound NCCL WORLD.
        default_pg = torch.distributed.group.WORLD
        use_split = (
            torch.__version__ >= (2, 5)
            and torch.distributed.get_backend() == 'nccl'
            # init_process_group(device_id=...) eagerly initializes the NCCL communicator
            # instead of waiting for the first communication; split_group requires this binding.
            and default_pg.bound_device_id is not None
            and default_pg._get_backend(default_pg.bound_device_id).supports_splitting
        )
        if use_split:
            group = torch.distributed.split_group(
                split_ranks=ranks, timeout=_LARGE_TIMEOUT)
            # None from split_group means nonmembership, not WORLD.
            for members in ranks:
                yield group if self.rank in members else torch.distributed.GroupMember.NON_GROUP_MEMBER
        else:
            _logger.warning("Falling back to torch.distributed.new_group due to unsupported split_group.")
            for members in ranks:
                yield torch.distributed.new_group(members, timeout=_LARGE_TIMEOUT)

    def _init_groups(self, group_batches: List[List[List[int]]]):
        """Create disjoint group batches, with all WORLD ranks participating."""
        for batch in group_batches:
            pending = [ranks for ranks in batch if not self.group_exists(ranks)]
            if pending:
                for ranks, group in zip(pending, self._new_groups(pending)):
                    self.groups[self.bitmap(ranks)] = group

    def get_p2p_group(self, src: int, dst: int):
        """Look up a shared group without creating one; unregistered pairs use WORLD."""
        # Only the endpoints call this, whereas group creation requires all WORLD ranks.
        return self.groups.get(self.bitmap((src, dst)))

    def long_barrier(self):
        """
        Barrier synchronization with very long timeout
        """
        # torch.distributed.barrier(group=self.barrier_gloo_group)
        torch.distributed.barrier()

    def get_stream(self, name: str) -> torch.cuda.Stream:
        """
        Get stream by name. If name doesn't exist,
        will create a new one.
        if name is 'default', the default stream will be returned.
        """
        if name not in self.streams:
            stream = torch.cuda.Stream()
            self.streams[name] = stream
        return self.streams[name]

    def get_event(self, name: str, *, enable_timing: bool = False) -> torch.cuda.Event:
        """
        Get event by name. If name doesn't exist,
        will create a new one.

        Args:
            name: The name of the event.
            enable_timing: Whether to enable timing for the event. Default is False.
                This is only used when the event is created for the first time.
                If the event already exists, this argument will be ignored.
        """
        if name not in self.events:
            event = torch.cuda.Event(enable_timing=enable_timing)
            self.events[name] = event
        return self.events[name]

    def create_hybrid(self, group_num: List[int]) -> List[List[int]]:
        """
        Create hybrid (nested) groups given the each group number.

        The product of group_num should be same with total devices.
        """
        group_num = np.array(group_num)
        cnt = np.prod(group_num)
        if cnt != self.world_size:
            raise RuntimeError("product of group_num should be same with total device number")
        grid = np.arange(cnt).reshape(tuple(group_num))
        dims = list(range(len(group_num)))
        outputs = []
        groups = []
        for dim, num in enumerate(group_num):
            remain = np.prod(np.delete(group_num, dim))
            order = tuple(dims[:dim] + dims[dim+1:] + [dim])
            grid_dim = np.transpose(grid, order).reshape((remain,num))
            grid_dim = grid_dim.tolist()
            groups.extend(grid_dim)
            for ranks in grid_dim:
                if self.rank in ranks:
                    outputs.append(ranks)
        self.get_groups(groups)
        assert len(outputs) == len(group_num)
        return outputs

    def bitmap(self, ranks):
        """
        map the rank list to the bit map string
        """
        bits = '0' * self.world_size
        for rank in ranks:
            if rank >= len(bits):
                raise ValueError("rank {} out of range ({})".format(rank, len(bits)))
            bits = bits[0:rank] + '1' + bits[rank+1:]
        return bits

    def __repr__(self):
        msg = 'node rank: [{}] rank: [{}] local rank: [{}]\n'.format(self.node_rank, self.rank, self.local_rank)
        msg += 'communication groups (ranks):\n'
        for bitmap, group in self.groups.items():
            ranks = [rank for rank, bit in enumerate(bitmap) if bit == '1']
            if self.rank in ranks:
                msg += '\t group {}: my group rank: [{}]\n'.format(ranks, torch.distributed.get_rank(group))
        return msg


_instance: Optional[_DeviceGroup] = None

def DeviceGroup() -> _DeviceGroup:
    global _instance
    if _instance is None:
        _instance = _DeviceGroup()
    return _instance


def init_device():
    _ = DeviceGroup()


def uninit_device():
    global _instance
    if _instance is not None:
        _instance.close()
        _instance = None
