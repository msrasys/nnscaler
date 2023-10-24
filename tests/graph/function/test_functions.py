### Only test the anno creation in these tests

import cube.graph.function.function as F
from cube.ir.cten import IRTensor


def test_handle_broadcast_multi():
    ins_anno, out_anno = F._handle_broadcast_multi([IRTensor([4]), IRTensor([3, 4]), IRTensor([2, 3, 4])])
    assert ins_anno[0] == ['c']
    assert ins_anno[1] == ['b', 'c']
    assert ins_anno[2] == ['a', 'b', 'c']
    assert out_anno == ['a', 'b', 'c']

    ins_anno, out_anno = F._handle_broadcast_multi([IRTensor([1]), IRTensor([2, 1, 4]), IRTensor([2, 3, 4])])
    assert ins_anno[0] == ['1']
    assert ins_anno[1] == ['a', '1', 'c']
    assert ins_anno[2] == ['a', 'b', 'c']
    assert out_anno == ['a', 'b', 'c']

def test_Full():
    op = F.Full([1, 2, 3], 1.)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 1 2 3'

    op = F.Full([], 1.)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 1'

def test_Expand():
    inp = IRTensor([10, 1])
    out = IRTensor([10, 2])
    op = F.Expand(inp, 10, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ -> a 2'

    op.new([inp], [out], size=[10, 2])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ -> a 2'

def test_Where():
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([4]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, b, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([1]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, 1, a b -> a b'
    op = F.Where(IRTensor([3, 4]), 1, IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, ?, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, 1 -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, ? -> a b'
    op = F.Where(IRTensor([3, 4]), 1, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, ?, ? -> a b'

def test_FullSlice():
    op = F.FullSlice(IRTensor([2, 3, 4]), (1, 2, 3))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 1'
    op = F.FullSlice(IRTensor([2, 3, 4]), (..., 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b'
    op = F.FullSlice(IRTensor([2, 3, 4]), (1, 2, slice(0, 3, 2)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 2'
    op = F.FullSlice(IRTensor([2, 3, 4]), (1, 2, slice(1, 10, 1)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 3'

def test_Max():
    op = F.Max(IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 1'
    op = F.Max(IRTensor([2, 3, 4]), IRTensor([4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c, c -> a b c'
    op = F.Max(IRTensor([2, 3, 4]), 1, True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a 1 c, a 1 c'
    op = F.Max(IRTensor([2, 3, 4]), 1, False)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a c, a c'

def test_Squeeze():
    op = F.Squeeze(IRTensor([2, 1, 4, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c d'
