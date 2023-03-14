# Some operators should be specially handled during codegen to the frontend code,
# here we define the customized rule for code emisson.

from typing import Any, Callable, Dict, List, Optional

from cube import ir
from cube.ir.cten import IRTensor
from cube.ir.dtype import IRDType
from cube.ir.operator import IRFwOperation

import torch

# By default, we flatten all args and join them by ","
# this includes ops with a fixed number of parameters like 'add(x,y)',
# or ops allowing multiple parameters at the frontend like 'block_diag(t1,t2'
def _common_rule_join_all(node:IRFwOperation, arg_vars:List[str], kw_pairs:dict) -> str:
    signature = node.signature

    kw_assigns = list()
    for key, val in kw_pairs.items():
        code = f'{key}={val}'
        kw_assigns.append(code)

    args = ", ".join(arg_vars + kw_assigns)
    return f"{signature}({args})"

def _common_rule_input_as_list(node:IRFwOperation, arg_vars:List[str], kw_pairs:dict) -> str:
    signature = node.signature

    kw_assigns = list()
    for key, val in kw_pairs.items():
        code = f'{key}={val}'
        kw_assigns.append(code)
    
    args = ", ".join(arg_vars)
    kwargs = ", ".join(kw_assigns)
    return f"{signature}([{args}], {kwargs})"

def emit_slice(node, arg_vars:list, kw_pairs:dict) -> str:
    """
    The op is:
        aten::slice(input:Tensor, dim:int=0, start:Optional[int]=None, end:Optional[int]=None, step:int=1) -> Tensor
    
    but at the frontend such an invocation must be rewritten as 'x[:, l:h:s, :, :]'
    depending on the 'input's rank and the 'dim' value.
    """
    out_tensors : tuple = node.outputs()
    assert len(out_tensors) == 1
    out_tensor : IRTensor = out_tensors[0]

    assert len(arg_vars) == 1
    in_tensor_var : str = arg_vars[0]

    dim : int = kw_pairs["dim"]
    start : Optional[int] = kw_pairs["start"]
    end : Optional[int] = kw_pairs["end"]
    step : int = kw_pairs["step"]
    
    rank = len(out_tensor.shape)
    subscript_components = [":"] * rank

    slice_str = f"{start or ''}:{end or ''}:{step}"
    subscript_components[dim] = slice_str

    return f"{in_tensor_var}[{', '.join(subscript_components)}]"


# TODO consider making the IR-Torch conversion like IRDType2TorchDType intrinsic to codegen,
# so that we don't need to ad hoc do the conversion as in these emission functions.
# Also, we'd better limit the complexity of the values in 'kw_pairs' so we know for sure we have
# done all necessary conversion.
#
# Basically to convert internal 'IRDType' to frontend 'torch.dtype'
def emit_zeros(node, arg_vars:list, kw_pairs:dict) -> str:
    """
    zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """
    kw_pairs = kw_pairs.copy()
    if 'dtype' in kw_pairs:
        ir_dtype : IRDType = kw_pairs['dtype']
        if ir_dtype is not None:
            kw_pairs['dtype'] = IRDType2DType.map(ir_dtype)
    
    # TODO make all intermediately created tensors CUDA, to fit with other parts of the system, like SynDataLoader.
    if 'device' in kw_pairs:
        print(f'WARNING: overload device info. of {node}')
    kw_pairs['device'] = 'torch.cuda.current_device()' # str will get directly dumped as it's.

    if len(arg_vars) != 0:
        print(f'WARNING: emit_zero with len(arg_vars) {len(arg_vars)} != 0')
    
    return _common_rule_join_all(node, arg_vars, kw_pairs)

def emit_ones(node, arg_vars:list, kw_pairs:dict) -> str:
    """
    ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """
    kw_pairs = kw_pairs.copy()
    if 'dtype' in kw_pairs:
        ir_dtype : IRDType = kw_pairs['dtype']
        if ir_dtype is not None:
            kw_pairs['dtype'] = IRDType2DType.map(ir_dtype)
    
    # TODO make all intermediately created tensors CUDA, to fit with other parts of the system, like SynDataLoader.
    assert 'device' not in kw_pairs
    kw_pairs['device'] = 'torch.cuda.current_device()' # str will get directly dumped as it's.

    assert len(arg_vars) == 0
    return _common_rule_join_all(node, arg_vars, kw_pairs)

def emit_rand(node, arg_vars:list, kw_pairs:dict) -> str:
    """
    rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """
    kw_pairs = kw_pairs.copy()
    if 'dtype' in kw_pairs:
        ir_dtype : IRDType = kw_pairs['dtype']
        if ir_dtype is not None:
            kw_pairs['dtype'] = IRDType2DType.map(ir_dtype)
    
    # TODO make all intermediately created tensors CUDA, to fit with other parts of the system, like SynDataLoader.
    assert 'device' not in kw_pairs
    kw_pairs['device'] = 'torch.cuda.current_device()' # str will get directly dumped as it's.

    assert len(arg_vars) == 0
    return _common_rule_join_all(node, arg_vars, kw_pairs)


def emit_new_tensor(node, arg_vars:list, kw_pairs:dict) -> str:
    """
    rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """
    kw_pairs = kw_pairs.copy()
    if 'dtype' in kw_pairs:
        ir_dtype : IRDType = kw_pairs['dtype']
        if ir_dtype is not None:
            kw_pairs['dtype'] = IRDType2DType.map(ir_dtype)
    
    # TODO make all intermediately created tensors CUDA, to fit with other parts of the system, like SynDataLoader.
    assert 'device' not in kw_pairs
    kw_pairs['device'] = 'torch.cuda.current_device()' # str will get directly dumped as it's.
    
    assert len(arg_vars) == 0
    assert 'data' in kw_pairs
    assert 'shape' in kw_pairs
    data_str = str(kw_pairs['data'])
    _ = kw_pairs.pop('data')
    _ = kw_pairs.pop('shape')
    
    kw_assigns = list()
    for key, val in kw_pairs.items():
        assert key != 'data'
        code = f'{key}={val}'
        kw_assigns.append(code)
    args = data_str + ', ' + ', '.join(kw_assigns)
    return f'{node.signature}({args})'

# Basically to convert internal 'IRDType' to frontend 'torch.dtype'
def emit_to(node, arg_vars:list, kw_pairs:dict) -> str:
    kw_pairs = kw_pairs.copy()

    # Unlike 'zeros' who has 'ScalarType? dtype', 'to' has a non-nullable 'dtype'.
    ir_dtype : IRDType = kw_pairs['dtype']
    assert ir_dtype is not None
    kw_pairs['dtype'] = IRDType2DType.map(ir_dtype)

    return _common_rule_join_all(node, arg_vars, kw_pairs)

def emit_setattr(node, arg_vars: List[str], kw_pairs: Dict[str, str]) -> str:

    assert arg_vars[1].startswith('self.')
    member = f'"{arg_vars[1][5:]}"'
    return f"{node.signature}({arg_vars[0]}, {member}, {arg_vars[2]})"

def emit_index_select(node, arg_vars:list, kw_pairs:dict) -> str:
    assert 'dim' in kw_pairs
    dim = kw_pairs['dim']
    return f'{node.signature}({arg_vars[0]}, {dim}, {arg_vars[1]})'

def emit_einsum(node, arg_vars:list, kw_pairs:dict) -> str:
    assert 'equation' in kw_pairs
    equation = kw_pairs['equation']
    args_str = ', '.join(arg_vars)
    return f'{node.signature}({equation}, {args_str})'


class Sign2EmitRule:

    @staticmethod
    def map(signature:str) -> Callable[[IRFwOperation, List[str], Dict[str, Any]], str]:
        """
        The definition of the emit rule is like:
        
        ```
        def emit_for_lstm_cell(node, arg_vars, kw_pairs) -> str:
            x_var, h_var, c_var = arg_vars
            return f"lstm({x_var}, [{h_var}, {c_var}], OTHER_ARG_VARS)"
        ```

        'arg_vars' are inputs (all are Tensor-typed) variable names as string, e.g., ["x", "y"]
        'kw_pairs' are dict whose values has been preprocessed and can be directly stringified, 
            e.g., {"dim":1, "layout"="nchw"}
        """
        return Sign2EmitRule._signMap.get(signature) or _common_rule_join_all


    _signMap = {
        'torch.slice': emit_slice,
        'torch.zeros': emit_zeros,
        'torch.ones': emit_ones,
        'torch.Tensor.to': emit_to,
        'torch.rand': emit_rand,
        'torch.tensor': emit_new_tensor,

        'setattr': emit_setattr,
    }


# The reverse mapping of DType2IRDType in /graph/parser/mapping.py
class IRDType2DType:
    
    @staticmethod
    def map(ir_dtype:IRDType) -> torch.dtype:
        return IRDType2DType._map[ir_dtype] # subscript/[]-access will throw if not found

    _map = {
        ir.float64: torch.float64,
        ir.float32: torch.float32,
        ir.float16: torch.float16,
        ir.uint8:   torch.uint8,
        ir.int8:    torch.int8,
        ir.int16:   torch.int16,
        ir.int32:   torch.int32,
        ir.int64:   torch.int64,
        ir.boolean: torch.bool
    }
