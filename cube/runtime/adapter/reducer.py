"""
Borrowed from Megatron Implementation
"""

from typing import List
import torch
import warnings

from cube.runtime.device import DeviceGroup
from cube.profiler.timer import CudaTimer, print_each_rank


def get_nbytes(dtype: torch.dtype) -> int:
    try:
        if dtype.is_floating_point():
            return torch.finfo(dtype).bits // 8
        else:
            return torch.iinfo(dtype).bits // 8
    except Exception as e:
        warnings.warn(f'Cannot figure out bytes of dtype: {dtype}, set default as 4.')
        return 4


class Reducer:

    def __init__(self, ranks: List[int], max_bucket_size_bytes: int):
        """
        Create a reducer to synchronize weights and its gradients

        @param ranks List[int]: global ranks the reducer works on
        @param max_bucket_size_bytes int: max bytes for one allreduce call
        """

        self._params: List[torch.nn.Parameter] = list()
        # note this need to be called for every device
        self.ranks = ranks
        self._group = DeviceGroup().get_group(ranks)
        self._max_bucket_size_bytes = max_bucket_size_bytes

    def add_param(self, param: torch.nn.Parameter):
        self._params.append(param)

    def allreduce(self):
        """
        Reduce gradients across given group
        """
        buckets = {}
        for param in self._params:
            if param.requires_grad and param.grad is not None:
                tp = param.data.dtype
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)

        def sync(grads, non_blocking: bool = False) -> None:
            """
            inplacement synchronize gradients
            
            @param non_blocking bool: whether gradient copy is non-blocking
            """
            coalesced = self._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=self._group)
            all_synced = self._unflatten_dense_tensors(coalesced, grad_groups)
            for grad, synced in zip(grad_groups, all_synced):
                grad.copy_(synced, non_blocking=non_blocking)

        # for each bucket, do all-reduce
        CudaTimer().start(field_name='comm', predefined=True)
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            nbytes, grad_groups = 0, []
            for grad in grads:
                grad_groups.append(grad)
                nbytes += grad.nelement() * grad.element_size()
                if nbytes >= self._max_bucket_size_bytes:
                    # print(f'sync barrier: num gradients {len(grad_groups)}')
                    sync(grad_groups, non_blocking=True)
                    nbytes, grad_groups = 0, []
            if nbytes > 0:
                sync(grad_groups, non_blocking=True)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm', predefined=True)

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
