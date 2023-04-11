from typing import Optional, List

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph.parser import FxModuleParser, FxFuncOpTracer
from cube.graph.parser.concrete_trace_utils import concrete_trace
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
                  save_content: bool = True) -> IRGraph:
    """
    Convert torch.nn.Module based model into IRGraph
    """
    try:
        if CompileFlag.use_torchfx:
            if CompileFlag.use_default_fx_tracer:
                print('INFO: using torch.fx tracer')
                from torch.fx import symbolic_trace
                # Symbolic tracing frontend - captures the semantics of the module
                tracer = FxFuncOpTracer()
                traced_graph: torch.fx.Graph = tracer.trace(model)
                smodule: torch.fx.GraphModule = torch.fx.GraphModule(model, traced_graph)
                smodule.graph.print_tabular()
            else:
                print('INFO: using concrete tracer')
                if HAS_APEX:
                    leaf_module = (
                        torch.nn.Dropout, #torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d,
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
                    leaf_module = (torch.nn.Dropout, )
                traced_model = concrete_trace(
                    model,
                    dummy_input,
                    use_operator_patch=True,
                    autowrap_leaf_class={
                        torch.finfo: ((), False),
                        # type(output_origin): ((), False),
                    },
                    # FIXME: check if dropout is not included in it, can self.training be handled properly in the new version of 
                    # concrete_trace
                    leaf_module=leaf_module,
                    fake_device_type='cpu',
                )
        else:
            print('using torchscript tracer')
            smodule = torch.jit.script(model)

    except Exception as ex:
        print(ex)
        raise RuntimeError("Cannot convert module into torchscript/torch.fx module.")

    if CompileFlag.use_torchfx:
        if not dummy_input:
            FxModuleParser.save_content = save_content
            inputs, nodes, outputs = FxModuleParser.parse(smodule, input_shapes)
            module_name = model.__class__.__name__
        else:
            FxModuleParser.save_content = save_content
            inputs, nodes, outputs = FxModuleParser.parse(traced_model, input_shapes=None, dummy_inputs=dummy_input)
            module_name = model.__class__.__name__
    else:
        ScriptModuleParser.save_content = save_content
        inputs, nodes, outputs = ScriptModuleParser.parse_module(smodule, input_shapes)
        module_name = smodule.original_name

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph

