from typing import Optional, List

from cube.ir.tensor import IRFullTensor
from cube.graph.parser import ScriptModuleParser
from cube.graph import IRGraph
from cube.logics.dataloader import IRDataLoader

import torch

def convert_model(model: torch.nn.Module,
            input_shapes: Optional[ List[List[int],] ] = None) -> IRGraph:
    """
    Convert toch.nn.Module based model into IRGraph
    """
    try:
        smodule = torch.jit.script(model)
    except Exception as ex:
        print(ex)
        raise RuntimeError("Cannot convert module into torchscript moudle.")
    module_name = smodule.original_name
    inputs, nodes, outputs = ScriptModuleParser.parse_module(smodule, input_shapes)
    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False
    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph


def convert_dataloader(dataloader) -> IRDataLoader:
    """
    convert pytorch dataloader into IRDataLoader
    """
    from cube.graph.parser.mapping import DType2IRDType
    dataloader = IRDataLoader(dataloader, dtype_map=DType2IRDType)
    return dataloader
