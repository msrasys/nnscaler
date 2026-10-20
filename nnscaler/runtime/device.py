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
                        device_id=torch.device('cuda', int(os.environ.get('LOCAL_RANK'))),
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
        self._owned_groups: List = []
        self.p2p_groups: Dict[tuple[int, int], Optional[torch.distributed.ProcessGroup]] = {}
        self.use_p2p_groups = (
            not CompileFlag.dev_mode
            and is_running_distributed()
            and self.world_size > 1
        )
        self.streams: Dict[str, torch.cuda.Stream] = {
            'default': torch.cuda.default_stream()}
        self.events: Dict[str, torch.cuda.Event] = {}

    def close(self):
        if torch.distributed.is_initialized():
            # PyTorch sorts all groups by name on WORLD destruction. Split
            # names can differ across ranks, so release owned groups in the
            # common declaration order before destroying the parent.
            for group in reversed(self._owned_groups):
                torch.distributed.destroy_process_group(group)
            if self._is_pg_initer:
                torch.distributed.destroy_process_group()
        self._owned_groups.clear()
        self._is_pg_initer = False

    def _register_group(self, cache, key, group):
        """Cache every lookup result; own only this rank's live subgroup handles."""
        assert key not in cache
        cache[key] = group
        if group is not None and group != torch.distributed.GroupMember.NON_GROUP_MEMBER:
            self._owned_groups.append(group)

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
            group = torch.distributed.new_group(
                list(ranks), timeout=_LARGE_TIMEOUT)
            self._register_group(self.groups, rank_bits, group)
        return self.groups[rank_bits]

    def init_groups(self, groups):
        """Compatibility entry point for generated group declarations."""
        self.get_groups(groups)

    def get_groups(self, groups):
        """Collectively create a group list, batching disjoint memberships.

        Every rank must call this with the same groups, even nonmembers. Cached
        groups are reused; results preserve input order, including duplicates.
        WORLD returns None; nonmembers return NON_GROUP_MEMBER.
        """
        from nnscaler.runtime.group_init import create_groups

        pending = set()
        keys = []
        for ranks in groups:
            ranks = tuple(sorted(ranks))
            if not ranks or len(set(ranks)) != len(ranks):
                raise ValueError('Communication group ranks must be nonempty and unique')
            if any(not isinstance(rank, int) or rank < 0 or rank >= self.world_size for rank in ranks):
                raise ValueError(f'Communication group rank outside world size {self.world_size}')
            key = self.bitmap(ranks)
            keys.append(key)
            if key not in self.groups:
                pending.add(ranks)
        created = create_groups(sorted(pending), timeout=_LARGE_TIMEOUT)
        for ranks, group in created.items():
            self._register_group(self.groups, self.bitmap(ranks), group)
        return [self.groups[key] for key in keys]

    def init_p2p_groups(self, pairs):
        """
        Create deterministic two-rank groups for P2P communication.

        All ranks must call this method in the same order.
        The generated module passes a global sorted list of actual P2P pairs.
        """
        if not self.use_p2p_groups:
            return

        normalized_pairs = []
        for src, dst in pairs:
            if src == dst:
                continue
            if src < 0 or src >= self.world_size or dst < 0 or dst >= self.world_size:
                raise ValueError(f'P2P pair ({src}, {dst}) is out of world size {self.world_size}')
            normalized_pairs.append((src, dst) if src < dst else (dst, src))
        normalized_pairs = sorted(set(normalized_pairs))

        from nnscaler.runtime.group_init import create_groups
        created = create_groups(
            [pair for pair in normalized_pairs if pair not in self.p2p_groups],
            timeout=_LARGE_TIMEOUT,
        )
        for pair, group in created.items():
            self._register_group(self.p2p_groups, pair, group)
        torch.distributed.barrier()

    def get_sync_group(self, stride: int, world_size: int):
        """Return this rank's strided replica group, reusing existing handles.

        All WORLD ranks must participate when creating a new layout.
        """
        if not 0 < stride <= world_size <= self.world_size or world_size % stride:
            raise ValueError('Sync group size must be a positive multiple of its stride within the world')
        rank_lists = [list(range(i, world_size, stride)) for i in range(stride)]
        groups = self.get_groups(rank_lists)
        index = self.rank % stride
        return rank_lists[index], groups[index]

    def get_p2p_group(self, src: int, dst: int):
        """
        Return the dedicated two-rank process group and local group ranks for src/dst.
        """
        if not self.use_p2p_groups or src == dst:
            return None, None, None
        pair = (src, dst) if src < dst else (dst, src)
        group = self.p2p_groups.get(pair)
        if group is None:
            return None, None, None
        group_src = 0 if src == pair[0] else 1
        group_dst = 0 if dst == pair[0] else 1
        return group, group_src, group_dst

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
