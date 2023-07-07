# Some operators should be specially handled during codegen to the frontend code,
# here we define the customized rule for code emisson.

from typing import Callable, Dict, List, Optional

from cube import ir
from cube.ir.cten import IRTensor
from cube.ir.dtype import IRDType
from cube.ir.operator import IRFwOperation

import torch


class Sign2EmitRule:
    """Emit rule for frontend PyTorch codegen"""

    _sign2rule = {}

    @staticmethod
    def map(signature: str) -> Callable:
        """Get the emit rule for the given signature
        
        Args:
            signature (str): signature of the operator

        Returns:
            Callable: emit rule that takes the node, args (List[str]) and kwargs (Dict[str, str]) as input
        """
        return Sign2EmitRule._sign2rule.get(signature, Sign2EmitRule.emit_common)

    @staticmethod
    def emit_common(node: IRFwOperation, args: List[str], kwargs: Dict[str, str]) -> str:
        """Default rule to join all args and kwargs"""

        signature = node.signature

        kw_pairs = list()
        for key, val in kwargs.items():
            code = f'{key}={val}'
            kw_pairs.append(code)

        args = ", ".join(list(args) + kw_pairs)
        return f"{signature}({args})"

    @staticmethod
    def emit_slice(node: IRFwOperation, arg_vars: List[str], kw_pairs: Dict[str, str]) -> str:
        """Special rule for generating slice node

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

    @staticmethod
    def emit_setattr(node, arg_vars: List[str], kw_pairs: Dict[str, str]) -> str:
        """Special rule for generating setattr node
        """

        assert arg_vars[1].startswith('self.')
        member = f'"{arg_vars[1][5:]}"'
        return f"{node.signature}({arg_vars[0]}, {member}, {arg_vars[2]})"

    @staticmethod
    def emit_getattr(node, arg_vars: List[str], kw_pairs: Dict[str, str]) -> str:
        """Special rule for generating getattr node
        """
        return f"{node.signature}({arg_vars[0]}, '{arg_vars[1]}')"

    @staticmethod
    def emit_getitem(node, arg_vars: List[str], kw_pairs: Dict[str, str]) -> str:
        """Special rule for generating getitem node
        """
        if len(arg_vars) == 2 and len(kw_pairs) == 0 and not arg_vars[1].replace('_', '').isdigit():
            return f"{node.signature}({arg_vars[0]}, '{arg_vars[1]}')"
        else:
            return Sign2EmitRule.emit_common(node, arg_vars, kw_pairs)


# the registered emit rules
Sign2EmitRule._sign2rule = {
    'torch.slice': Sign2EmitRule.emit_slice,
    'setattr': Sign2EmitRule.emit_setattr,
    'builtins.getattr': Sign2EmitRule.emit_getattr,
    '_operator.getitem': Sign2EmitRule.emit_getitem,
}


class IRDType2DType:
    """
    The reverse mapping of DType2IRDType in /graph/parser/mapping.py
    """
    
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
