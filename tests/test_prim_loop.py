# run tests: 
# pytest ./tests/test_prim_loop.py

import pytest
import torch
import cube

from cube.graph.parser.frame import Frame
from cube.graph.parser.parser import ScriptModuleParser
from cube.ir.tensor import IRFullTensor
from cube import ir

# Stub objects:
# - A stub object for 'ScriptModule' should have members:
#   --  entry_method_normally_forward: Stub[ScriptMethod]
#   --  code: str   # only to avoid AttributeError, could be empty
#   and optionally:
#   --  other_script_method: Stub[ScriptMethod]
#   
# - A stub object for 'ScriptMethod' should have fields:
#   --  graph: torch._C.Graph

class StubScriptMethod(object):
    def __init__(self, graph: torch._C.Graph) -> None:
        self.graph = graph

# REMARK:
# 'torch._C.parse_ir' will change local variable names into unique-number ID, e.g.
#   graph(%p: int):
#       %local = ...
# becomes:
#   graph(%p: int):
#       %1  = ...

def out_var_name0(g):
    return next(g.outputs()).debugName()

def test_simple_unroll_evaluation():
    g = torch._C.parse_ir('''
        graph(%a : int):
            %ub : int = prim::Constant[value=100]()
            %truth : bool = prim::Constant[value=1]()
            %z : int = prim::Loop(%ub, %truth, %a)
                block0(%step : int, %p : int):
                    %r : int = aten::add(%step, %p)
                    -> (%truth, %r)
            return (%z)
    ''')
    frame = Frame()
    frame.push_var()
    frame.add_var("a", 0)

    for node in g.nodes():
        ir_nodes = ScriptModuleParser.parse_node(node, {}, frame)
        assert len(ir_nodes) == 0
    
    # %z becomes %3
    assert frame.get_var(out_var_name0(g)) == (0+99)*100//2

def test_unroll_with_structural_info():
    g = torch._C.parse_ir('''
        graph(%a : Tensor):
            %ub : int = prim::Constant[value=3]()
            %truth : bool = prim::Constant[value=1]()
            %i0 : int = prim::Constant[value=0]()
            %z : Tensor = prim::Loop(%ub, %truth, %a)
                block0(%step : int, %p : Tensor):
                    %ts : Tensor[] = prim::ListConstruct(%p, %p) # at each step, double the 0-th dim
                    %r : Tensor = aten::cat(%ts, %i0)
                    -> (%truth, %r)
            return (%z)
    ''')
    frame = Frame()
    frame.push_var()

    t_a = IRFullTensor(shape=[2,3])
    frame.add_var("a", t_a)

    all_ir_nodes = []
    for node in g.nodes():
        ir_nodes = ScriptModuleParser.parse_node(node, {}, frame)
        all_ir_nodes += ir_nodes
    
    assert len(all_ir_nodes) == 3

    p = t_a
    for i in range(3):
        ir_node = all_ir_nodes[i]
        in_val_1, in_val_2 = ir_node.inputs()
        assert in_val_1 == p
        assert in_val_2 == p

        out_val = ir_node.outputs(0)
        assert out_val.shape == [ 2**(i+2) , 3]

        p = out_val


def test_nested_unroll():
    '''
    The outer loop has 3 steps, and the inner loop has 3 steps too.
    '''

    subp = torch._C.parse_ir('''
        graph(%self: int, %a : Tensor):
            %ub : int = prim::Constant[value=3]()
            %truth : bool = prim::Constant[value=1]()
            %i0 : int = prim::Constant[value=0]()
            %z : Tensor = prim::Loop(%ub, %truth, %a)
                block0(%step : int, %p : Tensor):
                    %ts : Tensor[] = prim::ListConstruct(%p, %p) # at each step, double the 0-th dim
                    %r : Tensor = aten::cat(%ts, %i0)
                    -> (%truth, %r)
            return (%z)
    ''')
    main = torch._C.parse_ir('''
        graph(%self: int, %a : Tensor):
            %ub : int = prim::Constant[value=3]()
            %truth : bool = prim::Constant[value=1]()
            %z : Tensor = prim::Loop(%ub, %truth, %a)
                block0(%step : int, %p : Tensor):
                    %r : Tensor = prim::CallMethod[name="subp"](%self, %p)
                    -> (%truth, %r)
            return (%z)
    ''')

    class StubScriptModule(object):
        def __init__(self) -> None:
            self.main = StubScriptMethod(main)
            self.subp = StubScriptMethod(subp)
    module = StubScriptModule()

    frame = Frame()
    frame.push_var()

    t_a = IRFullTensor(shape=[2,3])
    frame.add_var("a", t_a)

    all_ir_nodes = []
    for node in main.nodes():
        ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
        all_ir_nodes += ir_nodes
    
    assert len(all_ir_nodes) == 9

    p = t_a
    for i in range(9):
        ir_node = all_ir_nodes[i]
        in_val_1, in_val_2 = ir_node.inputs()
        assert in_val_1 == p
        assert in_val_2 == p

        out_val = ir_node.outputs(0)
        assert out_val.shape == [ 2**(i+2) , 3]

        p = out_val