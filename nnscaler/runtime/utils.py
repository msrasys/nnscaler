#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""Runtime Utilities"""

from typing import Any, List, TYPE_CHECKING, Optional, Union
import logging
import heapq
import os

import torch

if TYPE_CHECKING:
    from nnscaler.runtime.adapter.reducer import FlattenParamInfo
    from nnscaler.runtime.module import AttrMeta


_logger = logging.getLogger(__name__)


_EMPTY_CACHE_RELEASE_COUNTER = 0


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'off', 'no', '')


def maybe_empty_cache_after_release() -> None:
    """Release cached CUDA blocks after schedule release points when fragmented.

    Multi-stream MoE overlap can create many differently sized temporary blocks.
    After Python references are dropped, PyTorch may still reserve enough free
    blocks to fragment later large allocations. This helper is intentionally
    thresholded so normal release points only pay a few allocator-stat queries.

    Environment knobs:
      NNSCALER_EMPTY_CACHE_ON_RELEASE=0 disables this helper.
      NNSCALER_EMPTY_CACHE_ON_RELEASE=force calls empty_cache every interval.
      NNSCALER_EMPTY_CACHE_THRESHOLD_MB sets the minimum cached/inactive bytes.
      NNSCALER_EMPTY_CACHE_FRAGMENTATION_RATIO sets the minimum pressure ratio.
      NNSCALER_EMPTY_CACHE_RELEASE_INTERVAL checks every N release points.
      NNSCALER_EMPTY_CACHE_SYNC=1 synchronizes before empty_cache.
    """
    if not torch.cuda.is_available():
        return

    mode = os.environ.get('NNSCALER_EMPTY_CACHE_ON_RELEASE', '1').strip().lower()
    if mode in ('0', 'false', 'off', 'no', ''):
        return

    global _EMPTY_CACHE_RELEASE_COUNTER
    _EMPTY_CACHE_RELEASE_COUNTER += 1
    interval = max(
        1, int(os.environ.get('NNSCALER_EMPTY_CACHE_RELEASE_INTERVAL', '1')))
    if _EMPTY_CACHE_RELEASE_COUNTER % interval != 0:
        return

    if mode != 'force':
        threshold = (
            int(os.environ.get('NNSCALER_EMPTY_CACHE_THRESHOLD_MB', '512'))
            * 1024 * 1024
        )
        ratio = float(os.environ.get(
            'NNSCALER_EMPTY_CACHE_FRAGMENTATION_RATIO', '0.10'))
        reserved = torch.cuda.memory_reserved()
        if reserved <= 0:
            return
        cached = max(0, reserved - torch.cuda.memory_allocated())
        try:
            inactive_split = torch.cuda.memory_stats().get(
                'inactive_split_bytes.all.current', 0)
        except Exception:
            inactive_split = 0
        pressure = max(cached, inactive_split)
        if pressure < threshold or pressure / reserved < ratio:
            return

    if _env_flag('NNSCALER_EMPTY_CACHE_SYNC', False):
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


class MicroBatchDataLoader:
    """
    MicroBatchDataLoader is used for scenarios of gradient accumulation,
    where a training iteration will have multiple data samples and perform
    multiple forward and backward on each sample (i.e., each refers to
    as a micro-batch).

    To support more flexible training patterns, e.g., pipeline parallelism,
    MicroBatchDataLoader supports wrapping all data samples of a training iteration
    into a light dataloader and passed as input for compilation.

    e.g.,

    ```python
    # compilation phase
    dataloader = MicroBatchDataLoader([(input1,),]) # only need one micro-batch

    @nnscaler.compile(model, dataloader, ...)
    def train_iter(model, dataloader):
        input1 = next(dataloader)
        loss = model(input1)
        loss.backward()
        return loss

    ...

    # runtime phase

    for mini_batch_samples in iter(dataloader):
        # mini_batch_samples are sample list for
        # all micro-batches in one iteration.
        dl = MicroBatchDataLoader(mini_batch_samples)
        loss =train_iter(model, dl)
        ...
    ```
    """

    def __init__(self, samples: List[Any], cycle: bool = False):
        """Create a micro-batch data loader for a mini-batch.

        Args:
            samples (List[Any]): a list of micro-batch samples. Each element
                in the list is a micro-batch sample.
            cycle (bool): whether to cycle the micro-batch samples. If True,
                the micro-batch samples will be cycled infinitely. Note this
                is only needed when the number of micro-batch samples is less
                than expected micro-batch number during runtime.
        """

        if not isinstance(samples, (tuple, list)):
            raise TypeError("Samples must be a tuple or list of samples.")
        self.samples = samples
        self.nmicros = len(samples)
        self.cycle = cycle
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx == self.nmicros:
            raise StopIteration
        batch = self.samples[self._idx]
        self._idx += 1
        if self.cycle:
            self._idx = self._idx % self.nmicros
        return batch

    def __len__(self):
        return self.nmicros

    def get_micro_batch(self, idx: int):
        idx = idx % self.nmicros if self.cycle else idx
        return self.samples[idx]


def microbatches(samples: List[Any], cycle: bool = False) -> MicroBatchDataLoader:
    """Create a micro-batch data loader for a mini-batch.

    This is for gradient accumulation scenarios. More details refer to
    documents of MicroBatchDataLoader.

    Args:
        samples (List[Any]): a list of micro-batch samples. Each element
            in the list is a micro-batch sample.
        cycle (bool): whether to cycle the micro-batch samples. If True,
            the micro-batch samples will be cycled infinitely. Note this
            is only needed when the number of micro-batch samples is less
            than expected micro-batch number during runtime.

    Returns:
        MicroBatchDataLoader: a micro-batch data loader.
    """
    return MicroBatchDataLoader(samples, cycle=cycle)


def split_array_min_max(nums: list[int], g: int, *, keep_order: bool = True) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split the array nums into g continuous subarrays such that the maximum sum
    of the subarrays is minimized.

    Args:
        nums (list[int]): The input array of integers.
        g (int): The number of groups to split the array into.
        keep_order (bool): Whether to keep the order of elements in the subarrays.
            If True, the order of elements in the original array is preserved
            in the subarrays. If False, the order can be changed.
    Returns:
        tuple[list[list[int]], list[list[int]]]:
            A tuple containing a list of g subarrays and their corresponding indices.
    """
    if g <= 0 or g > len(nums):
        raise ValueError("g must be in the range [1, len(nums)]")

    if not keep_order:
        return _split_array_min_max_out_of_order(nums, g)

    def _check(limit):
        count = 1
        count_sum = nums[0]
        for x in nums[1:]:
            if count_sum + x > limit:
                count += 1
                count_sum = x
            else:
                count_sum += x
        return count <= g

    # 1. Binary search to find the "minimum maximum sum" (Target Limit)
    left = max(nums)
    right = sum(nums)
    target_limit = right

    while left <= right:
        mid = (left + right) // 2
        if _check(mid):
            target_limit = mid
            right = mid - 1
        else:
            left = mid + 1

    # 2. Reconstruct the result based on the calculated target_limit
    # Note: A special greedy strategy is needed here to ensure exactly g groups
    # A simple greedy approach may result in fewer than g groups (although the maximum sum meets the condition, the number of groups is insufficient)

    result = [[nums[0]]]
    result_idx = [[0]]
    current_sum = nums[0]

    # We process in forward order, or forcefully reserve enough elements for the remaining groups during forward processing
    # Here we use forward iteration with a "remaining quota" check
    for i, x in enumerate(nums[1:], start=1):
        # Remaining groups needed
        groups_needed = g - len(result)
        # Remaining elements not yet processed
        elements_left = len(nums) - i
        if elements_left == groups_needed:
            # Each element must form a separate group
            result.append([x])
            result_idx.append([i])
            current_sum = x
            continue

        if current_sum + x > target_limit:
            result.append([x])
            result_idx.append([i])
            current_sum = x
        else:
            result[-1].append(x)
            result_idx[-1].append(i)
            current_sum += x

    return result, result_idx


def _split_array_min_max_out_of_order(nums: list[int], g: int) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split the array nums into g subarrays (order of elements can be changed)
    This problem (multi-way number partitioning) is NP-hard. We use a greedy approximation algorithm here.

    For more information, see https://en.wikipedia.org/wiki/Greedy_number_partitioning
    QUOTE:
        An improved greedy algorithm is called [LPT scheduling].
        It processes the inputs by descending order of value, from large to small.

        Since it needs to pre-order the inputs, it can be used only as an [offline algorithm].
        It guarantees that the largest sum is at most (4k-1)/3k  times the optimal (minimum) largest sum,
        and the smallest sum is at least  (3k-1)/(4k-2) times the optimal (maximum) smallest sum.

        See [LPT scheduling](https://en.wikipedia.org/wiki/LPT_scheduling) for more details.
    """
    # 1. Sort numbers in descending order
    nums_with_indices = list((nun, i) for i, nun in enumerate(nums))
    sorted_nums = sorted(nums_with_indices, reverse=True)

    # 2. Initialize heap
    heap = [(0, i) for i in range(g)]

    # groups to save results
    groups = [[] for _ in range(g)]
    group_idx = [[] for _ in range(g)]

    # 3. greedy assignment
    for num, idx in sorted_nums:
        # Pop the bucket with the smallest current sum
        current_sum, gidx = heapq.heappop(heap)

        # Add the number to this bucket
        groups[gidx].append(num)
        group_idx[gidx].append(idx)

        # Update the sum of this bucket and push it back to the heap
        new_sum = current_sum + num
        heapq.heappush(heap, (new_sum, gidx))

    return groups, group_idx


FLATTEN_META_KEY = '__nnscaler_flattened_meta__'
DISTRIBUTED_PARAM_META_KEY = '__nnscaler_distributed_meta__'


def is_fparam(param: torch.nn.Parameter) -> bool:
    """
    Check if a parameter is a flattened parameter.
    """
    return hasattr(param, FLATTEN_META_KEY)


def get_fparam_meta(param: torch.nn.Parameter) -> Optional['FlattenParamInfo']:
    """
    Get the meta information of a flattened parameter.
    None if the parameter is not a flattened parameter.
    """
    return getattr(param, FLATTEN_META_KEY, None)


def set_fparam_meta(param: torch.nn.Parameter, meta: 'FlattenParamInfo'):
    """
    Set the meta information of a flattened parameter.
    """
    setattr(param, FLATTEN_META_KEY, meta)


def is_dparam(param: torch.nn.Parameter) -> bool:
    """
    Check if a parameter is a distributed parameter.
    """
    return hasattr(param, DISTRIBUTED_PARAM_META_KEY)


def get_dparam_meta(param: torch.nn.Parameter) -> Optional['AttrMeta']:
    """
    Get the distributed meta information of a parameter.
    None if the parameter does not have distributed meta.
    """
    return getattr(param, DISTRIBUTED_PARAM_META_KEY, None)


def set_dparam_meta(param: torch.nn.Parameter, meta: 'AttrMeta'):
    """
    Set the distributed meta information of a parameter.
    """
    setattr(param, DISTRIBUTED_PARAM_META_KEY, meta)


def is_torch_version_at_least(major: int, minor: int) -> bool:
    """Compare PyTorch by release tuple, accepting nightly/pre-release builds."""
    release = str(torch.__version__).split('+', 1)[0]
    parts = []
    for part in release.split('.')[:2]:
        digits = ''.join(ch for ch in part if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (major, minor)


def set_grad_dtype(
        module_or_param: Union[torch.nn.Module, torch.nn.Parameter],
        dtype: Optional[torch.dtype],
        *,
        raise_on_unsupported: bool = True
):
    if dtype is None:
        return
    if not is_torch_version_at_least(2, 10):
        if raise_on_unsupported:
            raise RuntimeError(f'Setting grad dtype is only supported in PyTorch 2.10 or above, but got {torch.__version__}')
        return
    if isinstance(module_or_param, torch.nn.Module):
        for param in module_or_param.parameters():
            if param.requires_grad:
                param.grad_dtype = dtype
    else:
        assert isinstance(module_or_param, torch.nn.Parameter)
        module_or_param.grad_dtype = dtype


def get_grad_dtype(param: torch.nn.Parameter) -> torch.dtype:
    """
    Get the gradient dtype of a parameter.
    """
    return getattr(param, 'grad_dtype', param.dtype)
