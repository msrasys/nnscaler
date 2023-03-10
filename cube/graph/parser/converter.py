from typing import Optional, List

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph.parser import FxModuleParser, FxFuncOpTracer
from cube.graph.parser.concrete_trace_utils import concrete_trace
from cube.graph import IRGraph
from cube.flags import CompileFlag

import torch
import torch.fx

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
                with torch.no_grad():
                    if isinstance(dummy_input, torch.Tensor):
                        output_origin = model(dummy_input)
                        dummy_input = (dummy_input, )
                    elif isinstance(dummy_input, tuple) or isinstance(dummy_input, list):
                        output_origin = model(*dummy_input)
                    elif isinstance(dummy_input, dict):
                        print(f'WARNING dict dummy_input')
                        output_origin = model(**dummy_input)
                    else:
                        raise RuntimeError(f'dummy_input should be a tuple (not a {type(dummy_input)}) = {dummy_input}')
                traced_model = concrete_trace(
                    model,
                    dummy_input,
                    use_operator_patch=True,
                    autowrap_leaf_class={
                        torch.finfo: ((), False),
                        type(output_origin): ((), False),
                    },
                )
                traced_model.graph.print_tabular()
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
            inputs, nodes, outputs = FxModuleParser.parse(traced_model, input_shapes=input_shapes, dummy_inputs=dummy_input)
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

