#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.tensor import IRSubTensor, IRFullTensor, ValueMap

import pytest


def test_tensor_grad():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    subtensor = ftensor.tosub()

    assert isinstance(ftensor.grad, IRFullTensor)
    subtensor.grad = ftensor.grad.tosub()

    assert isinstance(subtensor.grad, IRSubTensor)

    ftensor.requires_grad = False
    assert ftensor.grad is None
    assert subtensor.grad is None
    assert subtensor.requires_grad is False


def test_continous():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    with pytest.raises(ValueError):
        IRSubTensor.is_dim_continous([], dim=0)

    indmap = []
    for dimlen in ftensor.shape:
        indmap.append((0, dimlen))
    indmap[0] = (0, 2)
    sub1 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (2, 4)
    sub2 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (4, 6)
    sub3 = ftensor.select(tuple(indmap), (0, 1))

    assert IRSubTensor.is_dim_continous([sub1, sub2, sub3], dim=0)
    assert not IRSubTensor.is_dim_continous([sub1, sub2, sub3], dim=1)
    assert not IRSubTensor.is_dim_continous([sub1, sub3], dim=0)


def test_valuemap_is_complete():
    # full value (0, 1) is complete
    assert ValueMap.is_complete([ValueMap((0, 1))])

    # two halves
    assert ValueMap.is_complete([ValueMap((0, 2)), ValueMap((1, 2))])

    # three thirds
    assert ValueMap.is_complete([ValueMap((0, 3)), ValueMap((1, 3)), ValueMap((2, 3))])

    # mixed granularity: (0,3) (1,3) (4,6) (5,6)
    # = [0,1/3) + [1/3,2/3) + [4/6,5/6) + [5/6,1) = [0,1)
    assert ValueMap.is_complete([
        ValueMap((0, 3)), ValueMap((1, 3)), ValueMap((4, 6)), ValueMap((5, 6))
    ])

    # mixed granularity: half + two quarters
    assert ValueMap.is_complete([
        ValueMap((0, 2)), ValueMap((2, 4)), ValueMap((3, 4))
    ])

    # incomplete: only first half
    assert not ValueMap.is_complete([ValueMap((0, 2))])

    # incomplete: gap in middle
    assert not ValueMap.is_complete([ValueMap((0, 3)), ValueMap((2, 3))])

    # empty list
    assert not ValueMap.is_complete([])

    # duplicate raises ValueError
    with pytest.raises(ValueError, match="Overlapping"):
        ValueMap.is_complete([ValueMap((0, 2)), ValueMap((0, 2))])

    # overlap with different granularity raises ValueError
    with pytest.raises(ValueError, match="Overlapping"):
        ValueMap.is_complete([ValueMap((0, 1)), ValueMap((0, 2))])

    # partial overlap raises ValueError
    with pytest.raises(ValueError, match="Overlapping"):
        ValueMap.is_complete([ValueMap((0, 2)), ValueMap((0, 3)), ValueMap((1, 3))])
