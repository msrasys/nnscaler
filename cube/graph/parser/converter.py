from typing import Optional, List
import logging

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph.parser.register import CustomizedOps
from cube.graph import IRGraph
from cube.flags import CompileFlag

import torch
import torch.fx

_logger = logging.getLogger(__name__)

try: 
    import apex
    HAS_APEX = True
except:
    HAS_APEX = False

def convert_model(model: torch.nn.Module,
                  input_shapes: Optional[ List[List[int],] ] = None,
                  dummy_input = None,
                  save_content: bool = True,
                  dynamic_shape: bool = False) -> IRGraph:
    """Convert torch.nn.Module based model into IRGraph

    Args:
        model (torch.nn.Module): single-device model description
        input_shapes (Optional[ List[List[int],] ]):
            input shapes of model, only required for torch.jit.script parser
        dummy_input (Optional[Any]):
            dummy input of model, only required for torch.fx parser
        save_content (bool):
            whether to save the content of model and load it into generated model. Default True.
        dynamic_shape (bool):
            whether to use dynamic shape. Default False.
    
    Returns:
        IRGraph: IRGraph of model
    """
    # get registered leaf function
    autowrap_funcs = [CustomizedOps.kOpRuntime.get(sign, None) for sign in CustomizedOps.kOpAutowrap]
    leaf_functions = {func: ([], True, None) for func in autowrap_funcs if func is not None}


    # step 1: trace model
    if CompileFlag.use_torchfx:
        _logger.info('use concrete torch.fx tracer')
        from cube.graph.parser.fx.concrete_trace_utils import concrete_trace
        from cube.graph.parser.fx.parser import FxModuleParser
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
    else:
        _logger.info('use torch.jit.script tracer')
        traced_model = torch.jit.script(model)

    # step 2: convert traced model into IRGraph
    if CompileFlag.use_torchfx:
        FxModuleParser.save_content = save_content
        FxModuleParser.dynamic_shape = dynamic_shape
        _logger.info(f"use {'dynamic' if dynamic_shape else 'static'} shape to parse graph")
        inputs, nodes, outputs = FxModuleParser.parse(traced_model, dummy_input)
        module_name = model.__class__.__name__
    else:
        if dynamic_shape:
            _logger.warning('dynamic shape is not supported in torch.jit.script')
        ScriptModuleParser.save_content = save_content
        inputs, nodes, outputs = ScriptModuleParser.parse_module(traced_model, input_shapes)
        module_name = traced_model.original_name

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph

