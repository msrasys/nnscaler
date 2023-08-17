"""
pytest unit_tests/graph/function/test_dimops.py
"""

from typing import Callable, Tuple, List
from functools import partial

import cube.graph.function as F
from cube.graph.function.dimops import IRDimops
from cube.ir.tensor import IRFullTensor


def create_op(creator: Callable,
              input_shapes: List[Tuple[int]], *args, **kwargs):
    inputs = tuple(IRFullTensor(shape=shape).tosub() for shape in input_shapes)
    return creator(*(inputs+args), **kwargs)


def partitionable(node: IRDimops, **config):
    print(f'\n\n# {node.anno}')
    print(f'testing node: {node}')
    sub_nodes = node.algorithms('dim').instantiate(**config)
    print(f'partitioned sub nodes:')
    for sub_node in sub_nodes:
        print(f'# {sub_node.anno}')
        print(sub_node)


test_view1 = partial(partitionable,
    create_op(F.Reshape, [(2048, 16, 64),], shape=[2048, 2, 512]),
    idx=0, dim=1, num=2,
)

test_view2 = partial(partitionable,
    create_op(F.Reshape, [(2048, 8, 64),], shape=[2048, 1, 512]),
    idx=0, dim=1, num=2,          
)

def create_udf_op1(input, weight, signature='test_udf_op1'):
    anno = 'L 8^ (L 2), L E -> 8^ (L 2) E '
    return IRDimops(create_udf_op1, 'udf_op1', signature, [anno], [input, weight])

test_multi_dim_partition = partial(partitionable,
    create_op(create_udf_op1, [(2048, 8, 4096), (2048, 4096)]),
    idx=0, dim=0, num=2,
)