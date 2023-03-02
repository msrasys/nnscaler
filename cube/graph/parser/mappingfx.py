import torch

from typing import Callable, Dict, Union
from functools import partial

import cube.graph.function as function
import cube.ir as ir
from cube.ir.operator import IRFwOperation

class SignFx2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if 'torch.' not in signature and 'cube.runtime.' not in signature:
            signature = signature.split('.')[-1]
        if signature in SignFx2Op.kOpMap:
            function = SignFx2Op.kOpMap[signature]
            # signature = 'torch.sum' if signature == 'sum' else signature #TODO fixme
            return partial(function, signature=signature)
        else:
            raise KeyError(f"{signature} is not supported yet")
            # return partial(function.UnkownOperator, signature=signature)

    @staticmethod
    def register(signature: str, op: Callable[..., Union[IRFwOperation, int, float]], code):
        """
        Register an operator
        """
        if not isinstance(signature, str):
            raise TypeError(f"Expected signature to be str but got {type(signature)}")
        if signature in SignFx2Op.kOpMap:
            raise KeyError(f"function {signature} is already registered")
        SignFx2Op.kOpMap[signature] = op
        SignFx2Op.kOpCodeDef[signature] = code

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'
    __fcntemplate = lambda name: f'torch._C._nn.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # torch.Tensor template
    __tttemplate = lambda name: f'torch.Tensor.{name}'

    # runtime template
    __rtemplate = lambda name: f'cube.runtime.function.function.{name}'

    # einops
    __einopsize = lambda name: f'einops._torch_specific.{name}'

    # custom ops
    __customops = lambda name: f'examples.custom_ops.{name}'

    kOpMap = {
        __fcntemplate('linear'): function.Linear,
        __ftemplate('dropout') : function.Dropout,
        __ttemplate('sum'): function.Sum,
        __ttemplate('squeeze'): function.Squeeze,
        __ttemplate('unsqueeze'): function.Unsqueeze,
        __tttemplate('type_as'): function.TypeAs,
        __ttemplate('triu'): function.Triu,
        __ftemplate('relu') : function.ReLU,
        __ttemplate('ne') : function.NE,
        __ttemplate('nan_to_num') : function.NanToNum,
        __tttemplate('long'): function.Long,
        __ttemplate('masked_fill'): function.MaskedFill,

        # # torch nn functional
        #
        # __ftemplate('linear') : function.Linear,
        #
        # __ttemplate('matmul'): function.Matmul,
        #
        # __ftemplate('softmax') : function.Softmax,
        #
        # __ftemplate('dropout') : function.Dropout,
        #
        # __ftemplate('gelu') : function.GeLU,
        # __ttemplate('gelu') : function.GeLU,
        #
        # __ftemplate('silu') : function.SiLU,
        # __ttemplate('silu') : function.SiLU,
        #
        # __ftemplate('_pad'): function.Pad,
        #
        # __ftemplate('layer_norm'): function.LayerNorm,
        #
        # __ftemplate('embedding'): function.Embedding,
        #
        # __ftemplate('cross_entropy'): function.CrossEntropy,
        #
        # # torch aten
        #
        # # creators
        # __ttemplate('zeros'): function.Zeros,
        # __ttemplate('ones'): function.Ones,
        # __ttemplate('tensor'): function.NewTensor,
        # __ttemplate('to'): function.ToTensor,
        # __ttemplate('rand'): function.Rand,
        # __ttemplate('clone'): function.Clone,
        #
        # __ttemplate('add') : function.Add,
        #
        # __ttemplate('sub') : function.Sub,
        #
        # __ttemplate('mul') : function.Mul,
        #
        # __ttemplate('div') : function.Div,
        #
        # __ttemplate('floordiv') : function.FloorDiv,
        #
        # __ttemplate('neg'): function.Neg,
        #
        # __ttemplate('gt'): function.CompareGT,
        # __ttemplate('lt'): function.CompareLT,
        # __ttemplate('ge'): function.CompareGE,
        # __ttemplate('le'): function.CompareLE,
        #
        # __ttemplate('pow'): function.Pow,
        #
        # __ttemplate('sin'): function.Sin,
        #
        # __ttemplate('cos'): function.Cos,
        #
        # __ttemplate('tanh'): function.Tanh,
        #
        # __ttemplate('bmm') : function.BatchLinear,
        #
        # __ttemplate('sum') : function.Sum,
        # __ttemplate('mean') : function.Mean,
        #
        # __ttemplate('transpose') : function.Transpose,
        #
        # __ttemplate('view'): function.View,
        #
        # __ttemplate('reshape'): function.Reshape,
        #
        # __ttemplate('conv2d'): function.Conv2D,
        #
        # __ttemplate('conv3d'): function.Conv3D,
        #
        # __ttemplate('pad'): function.Pad,
        #
        # __ttemplate('select'): function.Select,
        #
        # __ttemplate('slice'): function.Slice,
        #
        # #pytorch1.11
        # __ttemplate('select_scatter'): function.SelectScatter,
        #
        # __ttemplate('repeat'): function.Repeat,
        #
        # #pytorch1.11
        # __ttemplate('linear'): function.Linear,
        #
        # __ttemplate('cat'): function.Cat,
        #
        # __ttemplate('stack'): function.Stack,
        #
        # __ttemplate('chunk'): function.Chunk,
        #
        # __ttemplate('flatten'): function.Flatten,
        #
        # __ttemplate('roll'): function.Roll,
        #
        # __ttemplate('adaptive_avg_pool1d'): function.AdaptiveAvgPool1d,
        #
        # # runtime functions
        # __rtemplate('anchor'): function.GraphAnchor,
        #
        # __rtemplate('identity'): function.Identity,
        #
        # __rtemplate('multiref'): function.MultiRef,
        #
        # __rtemplate('accum'): function.Accum,
        #
        # #einops
        # __einopsize('apply_for_scriptable_torch'): function.ScriptEinOps,

    }

    # customized operator code: signature -> code
    kOpCodeDef: Dict[str, str] = {}