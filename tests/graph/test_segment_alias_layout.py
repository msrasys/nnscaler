#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import nnscaler.graph.function.function as F
from nnscaler.graph.segment import IRSegment, IRSegmentExpander
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor


def _alias_segment(slices, alias_kind='identity'):
    source = IRFullTensor((8,), requires_grad=False)
    target = source.like()
    if alias_kind == 'identity':
        alias = F.Identity(source.tosub())
    else:
        alias = F.MultiRef(source.tosub(), 1)
        alias.set_kwarg('clone_level', int(alias_kind == 'clone'))
    alias.set_output(0, target.tosub())
    alias.device = (0, 1)

    consumers = []
    for device, bounds in enumerate(slices):
        consumer = IRFwOperation(
            'neg', 'torch.neg', [target.select((bounds,), (0, 1))], 1,
        )
        consumer.set_output(0, IRFullTensor(
            (bounds[1] - bounds[0],), requires_grad=False,
        ).tosub())
        consumer.device = device
        consumers.append(consumer)
    segment = IRSegment(
        [alias, *consumers], [source.tosub()],
        [consumer.output(0) for consumer in consumers],
    )
    return source, IRSegmentExpander(segment, set(), set())


@pytest.mark.parametrize('alias_kind', ['identity', 'multiref'])
def test_narrow_disjoint_partitions_through_alias(alias_kind):
    source, expander = _alias_segment(((0, 4), (4, 8)), alias_kind)

    partitions = expander._try_narrow_segment_ctensors(source)

    assert set(partitions) == {0, 1}
    assert partitions[0].parent == partitions[1].parent == source
    assert partitions[0].indmap == ((0, 4),)
    assert partitions[1].indmap == ((4, 8),)


@pytest.mark.parametrize('slices', [
    ((0, 4), (0, 4)),  # Equal subtensors on different devices are replicas.
    ((0, 5), (4, 8)),  # Unequal, overlapping subtensors also need an adapter.
    ((0, 8), (0, 8)),  # Real full-tensor consumers must remain full-sized.
])
def test_keep_overlapping_alias_consumers_full_sized(slices):
    source, expander = _alias_segment(slices)

    assert expander._try_narrow_segment_ctensors(source) is None


def test_do_not_propagate_layout_through_cloning_multiref():
    source, expander = _alias_segment(((0, 4), (4, 8)), 'clone')

    assert expander._try_narrow_segment_ctensors(source) is None


def test_keep_boundary_when_a_device_has_no_compute_partition():
    source, expander = _alias_segment(((0, 4),))

    assert expander._try_narrow_segment_ctensors(source) is None
