from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import pytest
from cube.codegen.codegen import ModelCodeGen
from cube.execplan.execplan import ExecutionPlan

from cube.graph.graph import IRGraph, IRSegment
from cube.ir.cten import IRCell, IRTensor
from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor, IRSubTensor

# Override tensor naming to omit TensorID since the ID assignment is too hard to predict.
class FakeModelCodeGen(ModelCodeGen):
    def tensor_naming(self, tensor: Any, prefix_attr: Optional[str] = None) -> str:
        if isinstance(tensor, IRTensor):
            name = tensor.name.replace(".", "_")
            if prefix_attr is not None and tensor.is_param():
                name = prefix_attr + name
            return name
        else:
            return super().tensor_naming(tensor, prefix_attr)

def make_nodes(args_list: list, input_vars:List[str], output_vars:List[str]) \
        -> Tuple[List[IRFwOperation], List[IRTensor], List[IRTensor]]:
    """
    Each element of `args_list` is in a form of:
    
    (RCGID:Optional[int], OutputNames:str|List[str], Signature:str, OpArg...:OpArgType...)

    If any `OpArg` is string, it's automatically mapped to a `IRTensor` with the same name.

    E.g.
    ```
    [
        ("sum_res", "sum_fn", "a", IRFullTensor(name="b"))
        (1, ["prod_res"], "prod_fn", "sum", "a")
    ]
    ```
    ... results in
    ```
    sum_res = sum_fn(a, b)
    def recompute(sum_res, a):
        prod_res = prod_fn(sum_res, a)
        return prod_res
    prod_res = checkpoint(recompute, sum_res, a)
    ```

    REMARK:
    `signature:str` will affect how the call is dumped, see also `cube/codegen/frontend_mapping.py`.
    Generally if it's not a 'torch.some_fn' operator, it's dumped as-it-is.
    """
    var_tensor_map = dict()

    def _convert(output_names:Union[str, List[str]], signature:str, op_args, rc_gid:Optional[int]):
        if type(output_names) is str:
            output_names = [output_names]

        op_kwargs = {}
        if type(op_args[-1]) is dict:
            op_args, op_kwargs = op_args[:-1], op_args[-1]

        mapped_inputs = [var_tensor_map.setdefault(arg, IRFullTensor(name=arg).tosub()) if type(arg) is str else arg for arg in op_args]
        mapped_outputs = [var_tensor_map.setdefault(oname, IRFullTensor(name=oname).tosub()) for oname in output_names]

        op = IRFwOperation("not_matter_name", signature, len(mapped_inputs), len(output_names))
        for i, input in enumerate(mapped_inputs):
            op.set_input(i, input)
        for i, output in enumerate(mapped_outputs):
            op.set_output(i, output)
        op.kwargs.update(op_kwargs)
        
        # All devices are the same
        op.device = 0

        op.recompute = rc_gid

        return op

    def convert(args):
        rc_gid = None
        if type(args[0]) is int:
            rc_gid, args = args[0], args[1:]
        return _convert(args[0], args[1], args[2:], rc_gid)

    nodes = [convert(args) for args in args_list]
    inputs = [var_tensor_map[n] for n in input_vars]
    outputs = [var_tensor_map[n] for n in output_vars]
    return nodes, inputs, outputs

def gen(node_defs, invars, outvars):
    nodes, inputs, outputs = make_nodes(node_defs, invars, outvars)
    # REMARK
    # Do not directly create a 'IRSegment' from 'nodes', instead, re-retrieve the IRSegment
    # using 'graph.segment(nodes)'
    # Because we rely on proper dataflow analysis when segmentation, which requires all nodes
    # are properly registered/'attach'-ed into the graph.
    graph = IRGraph(nodes, inputs, outputs, "module_name_not_matter")
    segment = graph.segment(nodes)
    assert list(segment.inputs()) == inputs
    assert list(segment.outputs()) == outputs

    codegen = FakeModelCodeGen(ExecutionPlan(graph))
    code : list = codegen.emit_segment_code(segment)
    return str.join("\n", code)


def test_codegen_segment_recompute__simple():
    code = gen([
        ("c", "add", "a", "b"),
        ("d", "add", "a", "c"),
    ], invars=["a","b"], outvars=["d"])
    assert code == """\
c = add(a, b)
del b
d = add(a, c)"""


def test_codegen_segment_recompute_rc__simple():
    code = gen([
        ("c", "add", "a", "b"),
        (1, "d", "add", "a", "c"),
    ], invars=["a","b"], outvars=["d"])

    assert code == """\
c = add(a, b)
del b

def recompute(a, c):
    d = add(a, c)
    return d

d = ckpt.checkpoint(recompute, a, c)"""


def test_codegen_segment_recompute_rc__del_args():
    code = gen([
        ("c", "add", "a", "b"),
        (1, "d", "add", "a", "c"),
        ("e", "sub", "d", "a"),
    ], invars=["a","b"], outvars=["e"])

    assert code == """\
c = add(a, b)
del b

def recompute(a, c):
    d = add(a, c)
    return d

d = ckpt.checkpoint(recompute, a, c)
del c
e = sub(d, a)"""


def test_codegen_segment_recompute_rc__multi_rc():
    code = gen([
        ("c", "add", "a", "b"),
        (1, "d", "add", "a", "c"),
        (1, "e", "sub", "d", "a"),
        ("f", "mul", "a", "d"),
        (2, "g", "div", "f", "e"),
        (2, "h", "pow", "f", "g"),
    ], invars=["a","b"], outvars=["g","h"])

    assert code == """\
c = add(a, b)
del b

def recompute(a, c):
    d = add(a, c)
    del c
    e = sub(d, a)
    return d, e

d, e = ckpt.checkpoint(recompute, a, c)
del c
f = mul(a, d)
del a, d

def recompute(f, e):
    g = div(f, e)
    del e
    h = pow(f, g)
    return g, h

g, h = ckpt.checkpoint(recompute, f, e)""" # e,f will be auto rel-ed when it returns
