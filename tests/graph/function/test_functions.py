### Only test the anno creation in these tests

import cube.graph.function.function as F
from cube.ir.cten import IRObject, IRTensor

import pytest
import torch


def o(value):
    return IRObject(value=value)


def assert_anno(op, expected):
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected


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
    assert op.kwargs['size'] == [-1, 2]

    op.new([inp], [out], size=[10, 2])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ -> a 2'

    with pytest.raises(ValueError):
        F.Expand(inp, (10, 2), 1)

    with pytest.raises(ValueError):
        F.Expand(inp, 1, (10, 2))

    with pytest.raises(ValueError):
        F.Expand(inp, 1)

    with pytest.raises(ValueError):
        F.Expand(inp, (5, 2))

    op = F.Expand(inp, -1, o(1))
    assert_anno(op,  'a b^ -> a 1')
    assert op.kwargs['size'][0] == -1
    assert op.kwargs['size'][1].value == 1

    op = F.Expand(inp, -1, 1)
    assert_anno(op,  'a b^ -> a 1')
    assert op.kwargs['size'] == [-1, 1]

    assert_anno(F.Expand(inp, -1, o(2)),  'a b^ -> a 2')
    assert_anno(F.Expand(inp, o((10, 2))),  'a^ b^ -> 10 2')

    op = F.Expand(inp, o(10), o(2))
    assert_anno(op,  'a^ b^ -> 10 2')
    assert op.kwargs['size'][0].value == 10
    assert op.kwargs['size'][1].value == 2

    op = F.Expand(inp, o(10), o(-1))
    assert_anno(op,  'a^ b^ -> 10 b^')
    assert op.kwargs['size'][0].value == 10
    assert op.kwargs['size'][1].value == -1

    op = F.Expand(inp, 10, 10, 2)
    assert_anno(op, 'b c^ -> 10 b 2')
    assert op.kwargs['size'] == [10, -1, 2]


def test_variadic_extraction():
    def o(value):
        return IRObject(value=value)
    assert F.extract_variadic([]) == ([], [])
    assert F.extract_variadic(1) == ([1], [False])
    assert F.extract_variadic([1]) == ([1], [False])
    assert F.extract_variadic((1,)) == ([1], [False])
    assert F.extract_variadic([1, 2]) == ([1, 2], [False, False])

    assert F.extract_variadic(o([])) == ([], [])
    assert F.extract_variadic(o(1)) == ([1], [True])
    assert F.extract_variadic(o([1])) == ([1], [True])
    assert F.extract_variadic(o((1,))) == ([1], [True])
    assert F.extract_variadic(o([1, 2])) == ([1, 2], [True, True])

    assert F.extract_variadic([1, o(2)]) == ([1, 2], [False, True])
    assert F.extract_variadic([1, o(2), 3, o(4)]) == ([1, 2, 3, 4], [False, True, False, True])

    with pytest.raises(ValueError, match='.*nested.*'):
        F.extract_variadic([1, o([2, 3])])
    with pytest.raises(ValueError, match='.*nested.*'):
        F.extract_variadic([1, [2, 3]])
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic(True)
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic([1, True])
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic(o([1, True]))


def test_Repeat():
    inp = IRTensor([3])
    op = F.Repeat(inp, (4, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 (2 b^)'

    inp = IRTensor([3])
    op = F.Repeat(inp, (4, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b -> 4 b'

    inp = IRTensor([3])
    op = F.Repeat(inp, 4, 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b -> 4 b'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, 4, 2, 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ c -> 4 (2 b^) c'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 2), 1)  # the args(1) is ignored
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b -> (4 a^) b'

    inp = IRTensor([3, 2])
    with pytest.raises(ValueError):
        op = F.Repeat(inp, (2))

    with pytest.raises(ValueError, match='.*nested.*'):
        op = F.Repeat(inp, 4, (4, 2))

    inp = IRTensor([3])
    op = F.Repeat(inp, o((4, 2)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 (2 b^)'

    inp = IRTensor([3])
    op = F.Repeat(inp, (4, o(1)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 b^'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, o(2)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (o(4), 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b -> (4 a^) b'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, 4, 2, o(1), 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'c^ d^ -> 4 2 c^ (2 d^)'

    inp = IRTensor([3, 2])
    with pytest.raises(ValueError):
        op = F.Repeat(inp, o(2))


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
    with pytest.raises(RuntimeError):
        op = F.FullSlice(IRTensor([2, 3, 4]), (IRTensor([1, 2, 3]),))
    with pytest.raises(RuntimeError):
        op = F.FullSlice(IRTensor([2, 3, 4]),(slice(1, IRTensor([2]), 3),))


def test_GetItem():
    op = F.GetItem(IRTensor([4, 2]), IRTensor([3, 5], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c d -> c d b'
    op = F.GetItem(IRTensor([4, 2]), IRTensor([3], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c -> c b'
    op = F.GetItem(IRTensor([3, 4, 2]), IRTensor([3], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, d -> d b c'
    op = F.GetItem(IRTensor([3, 4, 2]), IRTensor([3, 5], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, d e -> d e b c'


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
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c^ d -> a^ c^'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c d -> a^ b c d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), -1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), -2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a b c^ d'

    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (-1, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a b c^'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (1, -1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (1, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a c^ d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (0, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c^ d -> a^ b c^ d'


def test_Unsqueeze():
    op = F.Unsqueeze(IRTensor([2, 4]), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> 1 a b'
    op = F.Unsqueeze(IRTensor([2, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a 1 b'
    op = F.Unsqueeze(IRTensor([2, 4]), 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a b 1'
    op = F.Unsqueeze(IRTensor([2, 4]), -3)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> 1 a b'
    op = F.Unsqueeze(IRTensor([2, 4]), -2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a 1 b'
    op = F.Unsqueeze(IRTensor([2, 4]), -1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a b 1'


def test_ScaledDotProductAttention():
    op = F.ScaledDotProductAttention(IRTensor([8, 128, 64]), IRTensor([8, 256, 64]), IRTensor([8, 256, 32]), None, 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a e d^, a b^ d^, a b^ c -> a e c'
