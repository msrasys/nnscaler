"""
Spatial primitives for policy
"""
from cube.graph.ir_cten import IRCell, IRTensor
from cube.graph.ir_graph import IRGraph

from typing import List, Union


def assign(inst: Union[IRTensor, IRCell], ranks: List[int], graph: IRGraph) -> None:
    """
    Assign a IRTensor / IRCell with spatial rank placement

    For IRCell:
        the device attribute will be set to ranks,
        the inputs and outputs of this IRCell will also be changed
        to ranks.

    For IRTensor:
        A move operation will be changed and inserted in order:
            output_node -> move -> input_node
    """
    if not all([isinstance(rank, int) for rank in ranks]):
        raise TypeError("Expected ranks to be List[int]")
    if isinstance(inst, IRCell):
        inst.device = ranks
    elif isinstance(inst, IRTensor):
        if set(inst.device) == set(ranks):
            return
        # find nodes that generated this tensor from the graph
        src_node = list()
        dst_node = list()
        for node in graph.nodes():
            if inst in node.outputs():
                src_node.append(node)
            if inst in node.inputs():
                dst_node.append(node)
        if len(src_node) == 0:  # a leaf tensor
            raise NotImplementedError(
                "Prim [assign]: moving parameter is not supported"
            )
        if len(dst_node) == 0:  # a loss tensor
            raise RuntimeError(
                "Prim [assign]: moving a tensor that is never used in graph"
            )
        raise NotImplementedError(
            "Prim [assign]: moving tensor is not supported yet"
        )
    else:
        raise TypeError("Expected inst to ba Union[IRTensor, IRCell]")


def select(tensor: IRTensor, indices, val_op, shape) -> IRTensor:
    raise NotImplementedError("Prim [select]: selecting sub IRTensor is not supported")

