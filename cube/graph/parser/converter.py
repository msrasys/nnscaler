from typing import Optional, List
import warnings

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph.parser import FxModuleParser, FxFuncOpTracer
from cube.graph.parser.concrete_trace_utils import concrete_trace
from cube.graph.parser.register import CustomizedOps
from cube.graph import IRGraph
from cube.flags import CompileFlag

import torch
import torch.fx

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
    """
    Convert torch.nn.Module based model into IRGraph
    """
    # get registered leaf function
    customized_funcs = CustomizedOps.kOpRuntime.values()
    leaf_functions = {func: ([], False, None) for func in customized_funcs}

    try:
        if CompileFlag.use_torchfx:
            if CompileFlag.use_default_fx_tracer:
                if CompileFlag.log_parser:
                    print('> use default torch.fx tracer')
                # Symbolic tracing frontend - captures the semantics of the module
                tracer = FxFuncOpTracer()
                traced_graph: torch.fx.Graph = tracer.trace(model)
                traced_model: torch.fx.GraphModule = torch.fx.GraphModule(model, traced_graph)
                if CompileFlag.log_parser:
                    traced_model.graph.print_tabular()
            else:
                if CompileFlag.log_parser:
                    print('> use concrete torch.fx tracer')
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
                    print('WARNING: apex package is not installed')
                    leaf_module = None
                traced_model = concrete_trace(
                    model,
                    dummy_input,
                    use_operator_patch=True,
                    leaf_module=leaf_module,
                    autowrap_leaf_function=leaf_functions,
                    cpu_offload=True,
                )
        else:
            if CompileFlag.log_parser:
                print('> use default torch.jit.script tracer')
            traced_model = torch.jit.script(model)

    except Exception as ex:
        print(ex)
        raise RuntimeError("Cannot convert module into torchscript/torch.fx module.")

    if CompileFlag.use_torchfx:
        FxModuleParser.save_content = save_content
        FxModuleParser.dynamic_shape = dynamic_shape
        if CompileFlag.log_parser:
            print(f"> use {'dynamic' if dynamic_shape else 'static'} shape to parse graph")
        inputs, nodes, outputs = FxModuleParser.parse(traced_model, dummy_input)
        module_name = model.__class__.__name__
    else:
        if dynamic_shape:
            warnings.warn('dynamic shape is not supported in torch.jit.script',
                          category=RuntimeWarning)
        ScriptModuleParser.save_content = save_content
        inputs, nodes, outputs = ScriptModuleParser.parse_module(traced_model, input_shapes)
        module_name = traced_model.original_name

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph

