#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/graph/function/test_dimops.py
"""

from typing import Callable, Tuple, List
from functools import partial

import pytest

import nnscaler.graph.function as F
from nnscaler.graph.function.dimops import IRDimops, OpAnno, ShapeAnno, DimAnno
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.cten import IRObject, IRTensor


def create_op(creator: Callable,
              input_shapes: List[Tuple[int]], *args, **kwargs):
    inputs = tuple(IRFullTensor(shape=shape).tosub() for shape in input_shapes)
    return creator(*(inputs+args), **kwargs)


def set_outputs(op: IRDimops):
    inputs = op.inputs()
    require_grads = any(
            t.requires_grad for t in inputs if isinstance(t, IRTensor))
    inferred_shape = op.infer_shape()
    for idx, shape in inferred_shape.items():
        op.set_output(idx, IRFullTensor(shape=shape, requires_grad=require_grads).tosub())
    return op


def partitionable(node: IRDimops, **config):
    set_outputs(node)
    print(f'\n\n# {node.anno}')
    print(f'testing node: {node}')
    sub_nodes = node.algorithm('dim').instantiate(**config)
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

def UDFOp1(input, weight, signature='test_udf_op1'):
    anno = 'L 8^ (L 2), L E -> 8^ (L 2) E '
    return IRDimops(UDFOp1, 'udf_op1', signature, [anno], [input, weight])

test_multi_dim_partition = partial(partitionable,
    create_op(UDFOp1, [(2048, 8, 4096), (2048, 4096)]),
    idx=0, dim=0, num=2,
)


def test_no_return_op():

    def NoReturnOp(input, weight, signature='no_return_op'):
        anno = 'a b, b c -> ?'
        return IRDimops(NoReturnOp, 'no_return_op', signature, [anno], [input, weight])

    op = create_op(NoReturnOp, [(1024, 512), (512, 1024)])
    assert len(op.outputs()) == 1 and isinstance(op.output(0), IRObject) and (not isinstance(op.output(0), IRFullTensor))


def test_inner_dimensions_infer():

    def TestFunc(input, weight, signature='test_func'):
        anno = 'a b, (b c) -> a (b c)'
        return IRDimops(TestFunc, 'test_func', signature, [anno], [input, weight])

    op = create_op(TestFunc, [(1024, 512), (2048,)])
    partitionable(op, idx=0, dim=1, num=2)

def test_anno_kwargs_infer():

    def TestFunc(input, weight, number=128, signature='test_func'):
        anno = '(a number), (b number) -> (a b)'
        return IRDimops(TestFunc, 'test_func', signature, [anno], [input, weight], number=number)

    op = create_op(TestFunc, [(1024,), (2048,)])
    partitionable(op, idx=0, dim=0, num=2)


def test_constant_folding_infer():
    # TODO: please note that this test should be rewritten after we can fully support constant folding
    def TestFunc(input, weight, bias, number=128, signature='test_func'):
        anno = '(a number), (b number), (a b) -> 1'
        return IRDimops(TestFunc, 'test_func', signature, [anno], [input, weight, bias], number=number)

    op = create_op(TestFunc, [(1024,), (2048,), (128,)], number=IRObject(value=128))
    partitionable(op, idx=0, dim=0, num=2)


def test_transform_space():
    assert OpAnno('a b, b c -> a c').transform_space() == [(0, 0), (0, 1), (1, 1)]
    assert OpAnno('a^ b, b c -> a^ c').transform_space() == [(0, 1), (1, 1)]
    assert OpAnno('a b, (b n) c -> a (n c)').transform_space() == [(0, 0), (0, 1)]
    assert OpAnno('a b, (b n) c -> a (n b c)').transform_space() == [(0, 0)]
    assert OpAnno('a b, (b n) c -> a (1 b c) n').transform_space() == [(0, 0), (0, 1)]
    assert OpAnno('a b, (b n) c -> a (1 1 1 b c) n').transform_space() == [(0, 0), (0, 1)]
    assert OpAnno('a b, (b n) c^ -> a (1 1 1 b) n c^').transform_space() == [(0, 0), (0, 1)]
    assert OpAnno('a b, (d^ n) c -> a (c n) d^').transform_space() == [(0, 0), (0, 1), (1,1)]


def test_parse_op():
    assert str(OpAnno('a b, b c -> a c')) == 'a b, b c -> a c'
    assert str(OpAnno('a^ b, b c -> a^ c')) == 'a^ b, b c -> a^ c'
    assert str(OpAnno('a b, (b n) c -> a (n c)')) == 'a b, (b n^) c^ -> a (n^ c^)'
    assert str(OpAnno('a b, (b n) c -> a (n b c)')) == 'a b^, (b^ n^) c^ -> a (n^ b^ c^)'
    assert str(OpAnno('a b, (b n) c -> a (1 b c) n')) == 'a b, (b n^) c^ -> a (1^ b c^) n^'
    assert str(OpAnno('a b, (b n) c -> a (1 1 1 b c) n')) == 'a b, (b n^) c^ -> a (1^ 1^ 1^ b c^) n^'
    assert str(OpAnno('a b, (b n) c^ -> a (1 1 1 b) n c^')) == 'a b, (b n^) c^ -> a (1^ 1^ 1^ b) n^ c^'
    assert str(OpAnno('a b, (d^ n) c -> a (c n) d^')) == 'a b, (d^ n^) c -> a (c n^) d^'

    with pytest.raises(ValueError):
        str(OpAnno('a b^, b^ c -> a c^')) == 'a b^, b^ c -> a c^'


def test_plain_shape_no_meta():
    """Plain shape without ':' has no no_grad_reduce_ids."""
    s = ShapeAnno('a b c')
    assert s.no_grad_reduce_ids == ()
    assert s.no_grad_reduce is False
    assert s.no_grad_reduce_for('a') is False


def test_single_modifier():
    """'a b : /m' marks only 'm'."""
    s = ShapeAnno('a b : /m')
    assert s.no_grad_reduce_ids == ('m',)
    assert s.no_grad_reduce is False
    assert s.no_grad_reduce_for('m') is True
    assert s.no_grad_reduce_for('a') is False


def test_multiple_modifiers():
    """'a b : /m /n' marks 'm' and 'n'."""
    s = ShapeAnno('l h^ : /m /n')
    assert s.no_grad_reduce_ids == ('m', 'n')
    assert s.no_grad_reduce_for('m') is True
    assert s.no_grad_reduce_for('n') is True
    assert s.no_grad_reduce_for('l') is False
    assert s.no_grad_reduce is False


def test_wildcard_modifier():
    """'a b : /' is a wildcard — all identifiers match."""
    s = ShapeAnno('l h^ : /')
    assert s.no_grad_reduce_ids == ('',)
    assert s.no_grad_reduce is True
    assert s.no_grad_reduce_for('l') is True
    assert s.no_grad_reduce_for('h') is True
    assert s.no_grad_reduce_for('anything') is True


def test_slash_shortcut():
    """'/' alone is shortcut for '?:/' — ignore=True with wildcard no_grad_reduce."""
    s = ShapeAnno('/')
    assert s.ignore is True
    assert s.no_grad_reduce is True
    assert s.no_grad_reduce_for('x') is True
    # dims should be '?'
    assert s.ndims == 1
    assert s._dims[0].name == '?'


def test_question_mark():
    """'?' is ignore but has no no_grad_reduce."""
    s = ShapeAnno('?')
    assert s.ignore is True
    assert s.no_grad_reduce is False
    assert s.no_grad_reduce_for('x') is False


def test_repr_plain():
    s = ShapeAnno('a b^ c')
    assert repr(s) == 'a b^ c'


def test_repr_with_modifier():
    s = ShapeAnno('a b : /m')
    assert repr(s) == 'a b : /m'


def test_repr_with_multiple_modifiers():
    s = ShapeAnno('l h^ : /m /n')
    assert repr(s) == 'l h^ : /m /n'


def test_repr_wildcard():
    s = ShapeAnno('l h^ : /')
    assert repr(s) == 'l h^ : /'


def test_repr_slash_shortcut():
    """'/' becomes '?:/' internally, repr shows '?^ : /'."""
    s = ShapeAnno('/')
    assert repr(s) == '?^ : /'


def test_construct_from_tuple_with_no_grad_reduce_ids():
    dims = ShapeAnno._parse_dims('a b c')
    s = ShapeAnno(dims, no_grad_reduce_ids=('a', 'b'))
    assert s.no_grad_reduce_ids == ('a', 'b')
    assert s.no_grad_reduce_for('a') is True
    assert s.no_grad_reduce_for('c') is False


def test_construct_from_tuple_wildcard():
    dims = ShapeAnno._parse_dims('a b')
    s = ShapeAnno(dims, no_grad_reduce_ids=('',))
    assert s.no_grad_reduce is True
    assert s.no_grad_reduce_for('a') is True


def test_str_with_kwarg_raises():
    """Cannot pass no_grad_reduce_ids when dim_annos is a string."""
    with pytest.raises(ValueError):
        ShapeAnno('a b', no_grad_reduce_ids=('m',))


def test_invalid_meta_token():
    with pytest.raises(ValueError, match="expected /identifier"):
        ShapeAnno('a b : m')


def test_invalid_meta_token2():
    with pytest.raises(ValueError, match="expected /identifier"):
        ShapeAnno('a b : foo')


def test_star_expansion_preserves_no_grad_reduce_ids():
    """When '*' expansion happens in IRDimops.align(), no_grad_reduce_ids should be preserved."""
    def StarOp(input, weight, signature='star_op'):
        anno = '* a : /c, a b -> * b'
        return IRDimops(StarOp, 'star_op', signature, [anno], [input, weight])

    op = create_op(StarOp, [(4, 8, 16), (16, 32)])
    # After align, the first input's '*' expands but no_grad_reduce_ids should remain
    ianno = op.anno.input(0)
    assert ianno.no_grad_reduce_for('c') is True


def test_opanno_with_no_grad_reduce():
    """OpAnno should preserve no_grad_reduce modifiers in input ShapeAnnos."""
    anno = OpAnno('a b : /a, b c -> a c')
    ianno = anno.input(0)
    assert ianno.no_grad_reduce_for('a') is True
    assert ianno.no_grad_reduce_for('b') is False


def test_opanno_slash_input():
    """OpAnno with '/' input (the ignore+no_grad_reduce shortcut)."""
    anno = OpAnno('a b, / -> a b')
    ianno = anno.input(1)
    assert ianno.ignore is True
    assert ianno.no_grad_reduce is True
