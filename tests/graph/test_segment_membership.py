"""Membership must follow mutable segments and restored compiler graphs."""
import copy
import gc
import pickle
import weakref

import pytest

from nnscaler.graph.segment import CellPosition, IRSegment
from nnscaler.ir.operator import IRFwOperation


def op(name):
    return IRFwOperation(name, name, inputs=[], num_outputs=0)


def assert_matches_scan(segment, candidates):
    def scan(current, node, flatten):
        return node in current.nodes() or (flatten and any(
            scan(child, node, True) for child in current.nodes() if isinstance(child, IRSegment)))
    for node in candidates:
        for flatten in (False, True):
            assert segment.exist(node, flatten) == scan(segment, node, flatten)


def test_membership_tracks_nested_mutations_and_duplicates():
    a, b, c, missing = [op(name) for name in ('a', 'b', 'c', 'missing')]
    child = IRSegment([a], [], [])
    outer = IRSegment([child, b, b], [], [])
    candidates = [a, b, c, missing, child]
    assert_matches_scan(outer, candidates)
    child.insert(c, 0)
    assert_matches_scan(outer, candidates)
    child.replace(a, [b])
    assert_matches_scan(outer, candidates)
    outer.remove(b)
    assert outer.exist(b, flatten=False)
    outer.remove(b)
    assert not outer.exist(b, flatten=False)
    assert outer.exist(b)
    outer.remove(child)
    assert_matches_scan(outer, candidates)


def test_bulk_expansion_rebuilds_membership():
    a, b = op('a'), op('b')
    segment = IRSegment([a], [], [])
    assert segment.exist(a)
    segment._nodes[:] = [b]
    segment._reorder_producer_consumer()
    assert_matches_scan(segment, [a, b])


def test_restored_graph_builds_indexes_lazily():
    source = IRSegment([IRSegment([op('inner')], [], []), op('outer')], [], [])
    assert '_node_lookup' not in source.__getstate__()
    for restored in (copy.deepcopy(source), pickle.loads(pickle.dumps(source))):
        assert not hasattr(restored, '_node_lookup')
        inner = restored.nodes()[0].nodes()[0]
        assert restored.exist(inner)
        restored.nodes()[0].remove(inner)
        assert not restored.exist(inner)
        new = op('new')
        restored.insert(new, 0)
        assert restored.exist(new, flatten=False)


def test_legacy_state_supports_mutation_before_first_query():
    a, b = op('a'), op('b')
    segment = IRSegment([a], [], [])
    del segment._node_lookup
    segment.insert(b, 0)
    segment.remove(a, _pos=CellPosition((1,)))
    assert_matches_scan(segment, [a, b])


def test_shallow_copies_share_indexes_until_the_node_list_is_replaced():
    a, b, c = op('a'), op('b'), op('c')
    original = IRSegment([a], [], [])
    shared = copy.copy(original)
    shared.insert(b, 1)
    assert_matches_scan(original, [a, b, c])
    shared._nodes[:] = [c]
    shared._reorder_producer_consumer()
    assert_matches_scan(original, [a, b, c])
    original.insert(a, 0)
    assert_matches_scan(shared, [a, b, c])
    # Reordering uses a replacement list; subsequent edits are independent.
    shared._nodes = list(reversed(shared._nodes))
    shared._rebuild_node_lookup()
    shared.insert(b, 0)
    assert not original.exist(b)
    assert shared.exist(b)


@pytest.mark.parametrize('restore', [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
                         ids=['deepcopy', 'pickle'])
@pytest.mark.parametrize('reverse_queries', [False, True])
def test_restored_shallow_copies_keep_shared_membership(restore, reverse_queries):
    repeated = op('repeated')
    source = IRSegment([repeated, repeated], [], [])
    left, right = restore((source, copy.copy(source)))
    assert left._nodes is right._nodes
    repeated = left.nodes()[0]
    for segment in (right, left) if reverse_queries else (left, right):
        assert segment.exist(repeated)

    inserted = op('inserted')
    left.insert(inserted, 0)
    assert inserted in right.nodes()
    assert right.exist(inserted, flatten=False)
    right.remove(inserted)
    assert not left.exist(inserted)
    left.remove(repeated)
    assert right.exist(repeated)
    right.remove(repeated)
    assert not left.exist(repeated)

    child = IRSegment([op('inner')], [], [])
    right.insert(child, 0)
    inner = child.nodes()[0]
    assert left.exist(inner) and not left.exist(inner, flatten=False)
    left.remove(child)
    assert not right.exist(inner)

    # Replacing one list must detach it from the shared lookup owner.
    left._nodes = []
    left.insert(inserted, 0)
    assert left.exist(inserted) and not right.exist(inserted)


def test_equal_node_lists_keep_independent_lookups():
    shared = op('shared')
    left, right = IRSegment([shared], [], []), IRSegment([shared], [], [])
    assert left._nodes == right._nodes and left._nodes is not right._nodes
    inserted = op('inserted')
    left.insert(inserted, 0)
    assert not right.exist(inserted)


def test_lookup_registry_does_not_keep_graphs_alive():
    node = op('node')
    left = IRSegment([node], [], [])
    right = copy.copy(left)
    owner_ref, node_ref = weakref.ref(left._node_lookup), weakref.ref(node)
    # IRCell's unrelated method caches retain self; isolate lookup ownership.
    node.iobjs.cache_clear()
    node.oobjs.cache_clear()
    del left, right, node
    gc.collect()
    assert owner_ref() is None
    assert node_ref() is None
