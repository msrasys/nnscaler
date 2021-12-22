from typing import Optional, List

from cube.ir.cten import IRTensor
from cube.graph.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph import IRGraph

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
    # convert to SubTensor
    for idx, tensor in enumerate(inputs):
        if isinstance(tensor, IRFullTensor):
            tensor = tensor.tosub()
        inputs[idx] = tensor
    for idx, tensor in enumerate(outputs):
        if isinstance(tensor, IRFullTensor):
            tensor = tensor.tosub()
        outputs[idx] = tensor
    for node in nodes:
        for idx, tensor in enumerate(node.inputs()):
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
                node.set_input(idx, tensor)
        for idx, tensor in enumerate(node.outputs()):
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
                node.set_output(idx, tensor)
    graph = IRGraph(nodes, inputs, outputs, module_name)
    return graph
