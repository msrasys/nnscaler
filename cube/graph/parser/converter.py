from typing import Optional, List

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph.parser import FxModuleParser, FxFuncOpTracer
from cube.graph import IRGraph
from cube.flags import CompileFlag

import torch
import torch.fx

def convert_model(model: torch.nn.Module,
                  input_shapes: Optional[ List[List[int],] ] = None,
                  save_content: bool = True) -> IRGraph:
    """
    Convert torch.nn.Module based model into IRGraph
    """
    try:
        if CompileFlag.use_torchfx:
            # from torch.fx import symbolic_trace
            # # Symbolic tracing frontend - captures the semantics of the module
            tracer = FxFuncOpTracer()
            traced_graph: torch.fx.Graph = tracer.trace(model)
            smodule: torch.fx.GraphModule = torch.fx.GraphModule(model, traced_graph)
            smodule.graph.print_tabular()
        else:
            smodule = torch.jit.script(model)

    except Exception as ex:
        print(ex)
        raise RuntimeError("Cannot convert module into torchscript/torch.fx module.")

    if CompileFlag.use_torchfx:
        FxModuleParser.save_content = save_content
        inputs, nodes, outputs = FxModuleParser.parse(smodule, input_shapes)
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

