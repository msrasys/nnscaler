"""
Borrowed from Megatron Implementation
"""

from typing import List
import torch

from cube.runtime.device import DeviceGroup


class Reducer:

    def __init__(self, ranks: List[int]):

        self._params: List[torch.nn.Parameter] = list()
        # note this need to be called for every device
        self.ranks = ranks
        self._group = DeviceGroup().get_group(ranks)

    def add_param(self, param: torch.nn.Parameter):
        self._params.append(param)

    def allreduce(self):
        """
        Reduce gradients across given group
        """
        buckets = {}
        for param in self._params:
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # TODO: figure out why Megatron needs this?
                # param.main_grad = param.grad
        # for each bucket, do all-reduce
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = self._flatten_dense_tensors(grads)
            # coalesced /= len(self.ranks)
            torch.distributed.all_reduce(coalesced, group=self._group)
            all_synced = self._unflatten_dense_tensors(coalesced, grads)
            for grad, synced in zip(grads, all_synced):
                grad.copy_(synced)

    def sync(self):
        """
        Sync parameters before training
        """
        for param in self._params:
            torch.distributed.broadcast(param, self.ranks[0], group=self._group)
        torch.cuda.synchronize()

    def _flatten_dense_tensors(self, tensors):
        """
        Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
        same dense type.

        Since inputs are dense, the resulting tensor will be a concatenated 1D
        buffer. Element-wise operation on this buffer will be equivalent to
        operating individually.

        Args:
            tensors (Iterable[Tensor]): dense tensors to flatten.
        Returns:
            A contiguous 1D buffer containing input tensors.
        """
        return torch._utils._flatten_dense_tensors(tensors)

    def _unflatten_dense_tensors(self, flat, tensors):
        """
        View a flat buffer using the sizes of tensors. Assume that tensors are of
        same dense type, and that flat is given by _flatten_dense_tensors.

        Args:
            flat (Tensor): flattened dense tensors to unflatten.
            tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
              unflatten flat.

        Returns:
            Unflattened dense tensors with sizes same as tensors and values from
            flat.
        """
        return torch._utils._unflatten_dense_tensors(flat, tensors)
