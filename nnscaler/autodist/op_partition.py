#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.autodist.cube_operator import CubeOperator
from nnscaler.autodist.descs import NodePartitionDesc
from nnscaler.autodist.util import instantiate_partition_desc
from nnscaler.graph.function.dimops import DimAnno, IRDimops

import itertools
from typing import List, Optional, Tuple


def calc_factors(val: int, num: int) -> List[Tuple[int, ...]]:
    """
    Calculate all possible factors of val that can be divided into num parts.
    NOTE: 6=2*3 and 6=3*2 are considered the same.
    """
    plans = []

    def backtrace(target: int, remaining: int, path: List[int]):
        if remaining == 1:
            if target != 1:
                plans.append(path + [target])
            else:
                if target != 1 or path:
                    raise RuntimeError(f'invalid target {target}, path {path}')
                plans.append([1])
            return

        for i in range(2, target):
            if target % i == 0:
                backtrace(target // i, remaining - 1, path + [i])

    backtrace(val, num, [])

    visited = set()
    for plan in plans:
        plan.sort()
        visited.add(tuple(plan))
    return list(visited)


_factor_cache = {}


def calc_factors_cached(val: int, num: int) -> List[List[int]]:
    if (val, num) not in _factor_cache:
        _factor_cache[(val, num)] = calc_factors(val, num)
    return _factor_cache[(val, num)]


def generate_partitions(
        dim_ids: List[str],
        device_num: int) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """
    Generate all possible partitions of dim_ids into device_num parts.

    Args:
        dim_ids: a list of dimension names.
        device_num: the number of devices.

    Returns:
        A list of possible partitions.

    Example:
        dim_ids = ['a', 'b'], device_num = 4
        possible partitions:
            (('a', 'b'), (2, 2))
            (('b', 'a'), (2, 2))
            (('a',), (4,))
            (('b',), (4,))
    """
    candidates = []
    for i in range(1, device_num + 1):
        if i > len(dim_ids):
            break
        factors = calc_factors_cached(device_num, i)
        if not factors:
            break
        for factor in factors:
            visited = set()
            for factor_permutation in itertools.permutations(factor):
                if factor_permutation not in visited:
                    visited.add(factor_permutation)
                    for dim_permutation in itertools.permutations(dim_ids, i):
                        if -1 in dim_permutation and dim_permutation[0] != -1:
                            continue
                        candidates.append((dim_permutation, factor_permutation))
    return candidates


class OpPartition:
    """
    OpPartition represents a partition plan for a CubeOperator.
    It is defined by a list of partition_dims and a list of partition_nums.

    If there is a matrix multiplication operator with annotation 'm k+, k+ n -> m n'
    where m=512, k=1024, n=2048, a partition plan can be:
    partition_dims = [-1, 'm', 'k'], partition_nums = [2, 2, 2].
    It means that the operator will be split into 8 sub-operators with shape
    m=256, k=512, n=2048.
    NOTE:
    - if -1 in partition_dims, it should be placed at the first position.
    - the example partition above is different from [-1, 'k', 'm'], [2, 2, 2]
    """

    def __init__(self, partition_dims: Tuple[str, ...],
                 partition_nums: Tuple[int, ...], operator: CubeOperator,
                 partition_positions: Optional[Tuple[Tuple[int, int], ...]] = None):
        self.operator = operator
        self.partition_dims = tuple(partition_dims)
        self.partition_nums = tuple(partition_nums)
        self.is_partial_val = False

        if len(partition_dims) != len(partition_nums):
            raise ValueError(
                'partition_dims and partition_nums should have the same length')
        if not partition_dims:
            raise ValueError('partition plan must contain at least one step')
        if any(num < 1 for num in partition_nums):
            raise ValueError(
                f'partition numbers must be positive, got {partition_nums}'
            )

        if partition_positions is None:
            partition_positions = tuple(
                (-1, -1) if dim == -1 else operator.dim_id2pos(dim)
                for dim in partition_dims
            )
        else:
            partition_positions = tuple(tuple(pos) for pos in partition_positions)
        if len(partition_positions) != len(partition_dims):
            raise ValueError(
                'partition_positions and partition_dims should have the same length'
            )
        for dim_id, pos in zip(partition_dims, partition_positions):
            if dim_id == -1:
                if pos != (-1, -1):
                    raise ValueError(
                        f'replication must use position (-1, -1), got {pos}'
                    )
            elif pos[0] < 0 or pos[1] < 0:
                raise ValueError(f'invalid partition position {pos}')
        self.partition_positions = partition_positions

        if isinstance(self.operator.ir_cell, IRDimops):
            # Store one final local operator. All children at each ordered step
            # have identical shapes, while keyword modifiers are applied at the
            # same time as in graph partitioning.
            self.ir_cell = instantiate_partition_desc(
                operator.ir_cell,
                self.to_node_partition_desc(),
            )

            for dim, num in zip(partition_dims, partition_nums):
                if dim == -1:
                    continue
                if operator.get_reduce_type(dim) == DimAnno.ReduceType.Sum and \
                 num > 1:
                    self.is_partial_val = True
                    break
        else:
            if any(dim != -1 for dim in partition_dims):
                raise ValueError('only support replicated for non-dimops')
            self.ir_cell = operator.ir_cell

    def is_replicated(self):
        return all(dim == -1 for dim in self.partition_dims)

    def to_node_partition_desc(self) -> NodePartitionDesc:
        return NodePartitionDesc(list(zip(
            self.partition_positions,
            self.partition_nums,
        )))

    def __repr__(self):
        return (
            f'OpPartition({self.partition_dims}, {self.partition_nums}, '
            f'positions={self.partition_positions})'
        )
