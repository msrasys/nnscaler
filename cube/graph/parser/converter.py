from cube.graph.ir_opten import IRTensor
from typing import Optional, List

from cube.graph.parser import ScriptModuleParser
from cube.graph import IRGraph, IRTensor

import torch

def convert(model: torch.nn.Module,
            input_shapes: Optional[ List[List[int],] ] = None) -> IRGraph:
    """
    Convert toch.nn.Module based model into IRGraph
    """
    try:
        smodule = torch.jit.script(model)
    except Exception:
        raise RuntimeError("Cannot convert module into torchscript moudle.")
    module_name = smodule.original_name
    inputs, nodes, outputs = ScriptModuleParser.parse_module(smodule, input_shapes)
    for input in inputs:
        if isinstance(input, IRTensor):
            input.requires_grad = False
    graph = IRGraph(nodes, inputs, outputs, module_name)
    return graph
