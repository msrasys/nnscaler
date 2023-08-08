from typing import Any, Dict, Optional, List, Union
import logging
from pathlib import Path
import os

from cube.ir.tensor import IRFullTensor
from cube.graph.parser.register import CustomizedOps
from cube.graph import IRGraph
from cube.flags import CompileFlag

from cube.graph.parser.fx.parser import FxModuleParser
from cube.graph.parser.fx.concrete_trace_utils import concrete_trace

import torch
import torch.fx

_logger = logging.getLogger(__name__)

try:
    import apex
    HAS_APEX = True
except:
    HAS_APEX = False


def to_fx_graph(model: torch.nn.Module, dummy_input) -> torch.fx.GraphModule:
    """
    Convert torch.nn.Module based model into torch.fx.GraphModule
    Args:
        model (torch.nn.Module): single-device model description
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
    Returns:
        torch.fx.GraphModule representation of model
    """
    # get registered leaf function
    autowrap_funcs = [CustomizedOps.kOpRuntime.get(sign, None) for sign in CustomizedOps.kOpAutowrap]
    leaf_functions = {func: ([], True, None) for func in autowrap_funcs if func is not None}

    if HAS_APEX:
        leaf_module = (
            # torch.nn.Dropout, #torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d,
            apex.normalization.FusedLayerNorm,
            # NOTE: the following modules also have different behavior depending on self.training. but currently in used.
            # torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout,
            # torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            # torch.nn.LazyBatchNorm1d, torch.nn.LazyBatchNorm2d, torch.nn.LazyBatchNorm3d, torch.nn.SyncBatchNorm,
            # torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            # torch.nn.LazyInstanceNorm1d, torch.nn.LazyInstanceNorm2d, torch.nn.LazyInstanceNorm3d,
            )
    else:
        _logger.warning('apex package is not installed')
        leaf_module = None
    traced_model = concrete_trace(
        model,
        dummy_input,
        use_operator_patch=True,
        leaf_module=leaf_module,
        autowrap_leaf_function=leaf_functions,
        cpu_offload=True,
        record_frames=not CompileFlag.disable_code_line_info,
    )
    return traced_model


def to_ir_graph(
    traced_model: torch.fx.GraphModule,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    dynamic_shape: bool = False,
) -> IRGraph:
    """Convert torch.fx.GraphModule based model into IRGraph

    Args:
        traced_model (torch.fx.GraphModule): single-device model description in fx format
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        dynamic_shape (bool):
            whether to use dynamic shape. Default False.
        attr_savedir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    _logger.info(f"use {'dynamic' if dynamic_shape else 'static'} shape to parse graph")

    inputs, nodes, outputs = FxModuleParser.parse(
        traced_model, dummy_input,
        attr_savedir=attr_savedir,
        dynamic_shape=dynamic_shape,
        save_content=True,
    )
    module_name = traced_model.__class__.__name__

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph


def convert_model(
    model: torch.nn.Module,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    dynamic_shape: bool = False
) -> IRGraph:
    """Convert torch.nn.Module based model into IRGraph

    Args:
        model (torch.nn.Module): single-device model description
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        dynamic_shape (bool):
            whether to use dynamic shape. Default False.
        attr_save_dir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    traced_model = to_fx_graph(model, dummy_input)
    graph = to_ir_graph(traced_model, dummy_input, attr_savedir, dynamic_shape)
    return graph
