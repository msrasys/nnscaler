# run tests: 
# pytest ./tests/test_parser.py

import pytest
import torch
from cube.graph.function.creators import IROnes, IRZeros

from cube.graph.parser.frame import Frame
from cube.graph.parser.parser import ScriptModuleParser
from cube.graph.torch_dtype_mapping import DType2IRDType
from cube.ir.dtype import IRDType
from cube.ir.tensor import IRFullTensor
from cube import ir

@pytest.mark.parametrize(
    "aten_op, ir_op_cls",
    [("zeros", IRZeros), ("ones", IROnes)]
)
def test_optional_dtype_none(aten_op, ir_op_cls):
    g = torch._C.parse_ir(f'''
        graph():
            %d : int = prim::Constant[value=2]()
            %shape : int[] = prim::ListConstruct(%d, %d, %d)
            %none : NoneType = prim::Constant()
            %z : Tensor = aten::{aten_op}(%shape, %none, %none, %none, %none)
            return (%z)
    ''')
    frame = Frame()
    frame.push_var()
    frame.push_attr()

    for node in g.nodes():
        ir_nodes = ScriptModuleParser.parse_node(node, {}, frame)
        for node in ir_nodes:
            if isinstance(node, ir_op_cls):
                assert node.output(0).dtype == DType2IRDType.map(torch.get_default_dtype())

@pytest.mark.parametrize(
    "aten_op, ir_op_cls",
    [("zeros", IRZeros), ("ones", IROnes)]
)
def test_optional_dtype_underlying_int(aten_op, ir_op_cls):
    # ScalarType(3) == torch.int32
    g = torch._C.parse_ir(f'''
        graph():
            %d : int = prim::Constant[value=2]()
            %shape : int[] = prim::ListConstruct(%d, %d, %d)
            %none : NoneType = prim::Constant()
            %scalarType : int = prim::Constant[value=3]()
            %z : Tensor = aten::{aten_op}(%shape, %scalarType, %none, %none, %none)
            return (%z)
    ''')
    frame = Frame()
    frame.push_var()
    frame.push_attr()

    for node in g.nodes():
        ir_nodes = ScriptModuleParser.parse_node(node, {}, frame)
        for node in ir_nodes:
            if isinstance(node, ir_op_cls):
                assert node.output(0).dtype == IRDType.int32
