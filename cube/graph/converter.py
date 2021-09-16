from cube.graph.parser import ScriptModuleParser
from cube.graph.graph import IRGraph

import torch

def convert(model: torch.nn.Module) -> IRGraph:
    """
    Convert toch.nn.Module based model into IRGraph
    """
    try:
        smodule = torch.jit.script(model)
    except Exception:
        raise RuntimeError("Cannot convert module into torchscript moudle.")
    module_name = smodule.original_name
    inputs, nodes, outputs = ScriptModuleParser.parse_module(smodule)
    graph = IRGraph(nodes, inputs, outputs, module_name)
    return graph
