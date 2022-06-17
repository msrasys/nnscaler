"""
Mapping of
    Signature -> IROperator
"""
from typing import Any, Callable, Dict, Union
import torch

import operator
from functools import partial

import cube.graph.function as function
from cube.ir.operator import IRFwOperation

# TODO this is a backwards-compatible alias
from cube.graph.torch_dtype_mapping import DType2IRDType

class Sign2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if signature in Sign2Op.kOpMap:
            return partial(Sign2Op.kOpMap[signature], signature=signature)
        else:
            raise KeyError(f"{signature} is not supported yet")
            # print(f'warning: {signature} is not recognized')
            # return partial(function.UnkownOperator, signature=signature)

    @staticmethod
    def register(signature: str, op: Callable[..., Union[IRFwOperation, int, float]], code):
        """
        Register an operator
        """
        if not isinstance(signature, str):
            raise TypeError(f"Expected signature to be str but got {type(signature)}")
        if signature in Sign2Op.kOpMap:
            raise KeyError(f"function {signature} is already registered")
        Sign2Op.kOpMap[signature] = op
        Sign2Op.kOpCodeDef[signature] = code

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # einops
    __einopsize = lambda name: f'einops._torch_specific.{name}'

    # custom ops
    __customops = lambda name: f'examples.custom_ops.{name}'

    kOpMap = {

        # torch nn functional

        __ftemplate('linear') : function.Linear,

        __ftemplate('softmax') : function.Softmax,

        __ftemplate('dropout') : function.Dropout,

        __ftemplate('gelu') : function.GeLU,
        __ttemplate('gelu') : function.GeLU,

        __ftemplate('_pad'): function.Pad,

        __ftemplate('layer_norm'): function.LayerNorm,

        __ftemplate('embedding'): function.Embedding,

        # __ftemplate('layer_norm'): function.LayerNorm,

        # torch aten

        # creators
        __ttemplate('zeros'): function.Zeros,
        __ttemplate('tensor'): function.NewTensor,
        __ttemplate('to'): function.ToTensor,

        __ttemplate('add') : function.Add,

        __ttemplate('sub') : function.Sub,

        __ttemplate('mul') : function.Mul,

        __ttemplate('div') : function.Div,

        __ttemplate('floordiv') : function.FloorDiv,

        __ttemplate('neg'): function.Neg,

        __ttemplate('gt'): partial(function.comparison_einops, operator.gt, 'gt'),
        __ttemplate('lt'): partial(function.comparison_einops, operator.lt, 'lt'),
        __ttemplate('ge'): partial(function.comparison_einops, operator.ge, 'ge'),
        __ttemplate('le'): partial(function.comparison_einops, operator.le, 'le'),

        __ttemplate('pow'): function.Pow,

        __ttemplate('sin'): function.Sin,

        __ttemplate('cos'): function.Cos,

        __ttemplate('bmm') : function.BatchLinear,

        __ttemplate('sum') : function.Sum,

        __ttemplate('transpose') : function.Transpose,

        __ttemplate('view'): function.View,

        __ttemplate('reshape'): function.Reshape,

        __ttemplate('conv2d'): function.Conv2D,

        __ttemplate('conv3d'): function.Conv3D,

        __ttemplate('select'): function.Select,

        __ttemplate('slice'): function.Slice,

        #pytorch1.11
        __ttemplate('select_scatter'): function.SelectScatter,

        __ttemplate('repeat'): function.Repeat,

        #pytorch1.11
        __ttemplate('linear'): function.Linear,

        __ttemplate('cat'): function.Cat,

        __ttemplate('stack'): function.Stack,

        #einops
        __einopsize('apply_for_scriptable_torch'): function.ScriptEinOps,

        #custom ops
        __customops('strip_2_borders'): function.CustomOps,
        __customops('update_diag_'): function.CustomOps,
        __customops('update_geopotential_'): function.CustomOps,
    }

    # customized operator code: signature -> code
    kOpCodeDef: Dict[str, str] = {}
