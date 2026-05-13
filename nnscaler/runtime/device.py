#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Communication group settings among devices
"""
from typing import Any, List, Dict, Optional, Tuple
import numpy as np
import torch
import os
import logging
import datetime
import threading

from nnscaler.flags import CompileFlag
from nnscaler.utils import is_running_distributed

_logger = logging.getLogger(__name__)
_LARGE_TIMEOUT = datetime.timedelta(seconds=21600)
_deferred_release_refs: List[Tuple[torch.cuda.Event, Tuple[torch.Tensor, ...]]] = []
_deferred_release_lock = threading.Lock()


def _collect_cuda_tensors(value: Any, refs: List[torch.Tensor], seen: set):
    if isinstance(value, torch.Tensor):
        if value.is_cuda and id(value) not in seen:
            refs.append(value)
            seen.add(id(value))
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_cuda_tensors(item, refs, seen)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_cuda_tensors(item, refs, seen)


def prune_deferred_releases():
    """Drop tensor refs whose last-use CUDA event has completed."""
    with _deferred_release_lock:
        if not _deferred_release_refs:
            return

        pending = []
        for event, refs in _deferred_release_refs:
            if not event.query():
                pending.append((event, refs))
        _deferred_release_refs[:] = pending


def flush_deferred_releases():
    """Synchronize and release all tensors held by defer_release."""
    with _deferred_release_lock:
        if not _deferred_release_refs:
            return

        for event, _ in _deferred_release_refs:
            event.synchronize()
        _deferred_release_refs.clear()


def defer_release(*values: Any, stream: Optional[torch.cuda.Stream] = None,
                  stream_name: Optional[str] = None):
    """Keep CUDA tensor refs alive until work already queued on a stream completes.

    This is used by multi-stream schedules before deleting local tensor variables.
    It avoids Tensor.record_stream while still preventing the caching allocator from
    reusing storage that a non-default stream may still be reading or writing.
    """
    refs: List[torch.Tensor] = []
    seen = set()
    for value in values:
        _collect_cuda_tensors(value, refs, seen)
    if not refs:
        return

    prune_deferred_releases()

    if stream is None:
        if stream_name is not None:
            stream = DeviceGroup().get_stream(stream_name)
        else:
            stream = torch.cuda.current_stream(refs[0].device)

    event = torch.cuda.Event()
    event.record(stream)
    with _deferred_release_lock:
        _deferred_release_refs.append((event, tuple(refs)))


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
                torch.distributed.init_process_group(
                    backend='nccl', timeout=_LARGE_TIMEOUT
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
        for dim, num in enumerate(group_num):
            remain = np.prod(np.delete(group_num, dim))
            order = tuple(dims[:dim] + dims[dim+1:] + [dim])
            grid_dim = np.transpose(grid, order).reshape((remain,num))
            grid_dim = grid_dim.tolist()
            for ranks in grid_dim:
                # initialize group
                _ = self.get_group(ranks)
                if self.rank in ranks:
                    outputs.append(ranks)
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
    flush_deferred_releases()
    if _instance is not None:
        _instance.close()
        _instance = None
