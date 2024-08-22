# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
import importlib
import operator
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Set, Tuple

import torch
from torch.fx.node import Node, map_aggregate, _side_effectful_functions

DICT_KEYS_TYPE = type({}.keys())
DICT_VALUES_TYPE= type({}.values())
DICT_ITEMS_TYPE= type({}.items())

_orig_node_is_impure: Callable = Node.is_impure

side_effectful_inplace_ops = {
    operator.iadd, operator.isub, operator.imul, operator.itruediv, operator.ifloordiv,
    operator.iand, operator.ior, operator.ixor, operator.ilshift, operator.irshift,
    operator.imod, operator.ipow,
    # operator.imatmul is not implemented in torch
    # so let's ignore it now
    operator.setitem,
}


class ExtraSEFPatcher:
    def __init__(self, extra_side_effectful_functions: Set[Callable]):
        self.extra_side_effectful_functions = extra_side_effectful_functions
        self.incontext_funcs = set()

    def __enter__(self):
        self.incontext_funcs = self.extra_side_effectful_functions - _side_effectful_functions
        _side_effectful_functions.update(self.incontext_funcs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _side_effectful_functions.difference_update(self.incontext_funcs)


class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    requires_grad : bool
    stride : Tuple[int]
    memory_format : Optional[torch.memory_format]

    # Quantization metadata
    is_quantized : bool
    qparams: Dict[str, Any]


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)


def extract_tensor_metadata(obj: Any):
    if isinstance(obj, torch.Tensor):
        return _extract_tensor_metadata(obj)
    else:
        return obj


def extract_results_metadata(results: Any, node: Node):
    if results is not EmptyResult:
        res = tuple(results) if isinstance(results, (DICT_KEYS_TYPE, DICT_VALUES_TYPE, DICT_ITEMS_TYPE)) else results
        meta = map_aggregate(res, extract_tensor_metadata)
        # we should get the meta info of the inner element of these type obj
        if isinstance(results, DICT_KEYS_TYPE):
            meta = {i: m for i, m in enumerate(meta)}.keys()
        if isinstance(results, DICT_VALUES_TYPE):
            meta = {i: m for i, m in enumerate(meta)}.values()
        if isinstance(results, DICT_ITEMS_TYPE):
            meta = {i: m for i, m in meta}.items()
        node.meta['tensor_meta'] = meta
        node.meta['type'] = type(results)


class EmptyResult:
    """Used for identification no results.
    """
    pass


@dataclass
class FrameRecord:
    filename: str
    lineno: str
    line: str
    # the name of the frame is the function name
    name: str

    def __repr__(self) -> str:
        if self.filename:
            return f'File "{self.filename}", line {self.lineno}, in {self.name},  {self.line}'
        else:
            return ''


def get_frame_record() -> Optional[FrameRecord]:
    # record code frame, include filename, line number, and function name
    frame_record = None
    cube_path = str(Path(importlib.util.find_spec('nnscaler').origin).parent) + '/'  # the cube path
    torch_path = str(Path(importlib.util.find_spec('torch').origin).parent) + '/'  # the torch path
    ignore_dirs = [cube_path, torch_path]
    # the last frame is the current frame [get_frame_record], so we need to skip it
    for frame in traceback.extract_stack()[-2::-1]:
        if any(p in frame.filename for p in ignore_dirs):
            continue
        frame_record = FrameRecord(frame.filename, frame.lineno, frame.line, frame.name)
        break
    return frame_record
