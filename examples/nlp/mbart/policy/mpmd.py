from typing import List

from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.cten import IRCell
    

def _group_to_blocks(fnodes) -> List[List[IRCell]]:
    """
    Grouping to [
        [Encoder Embed],
        [Encoder Layer], [Encoder Layer], ...,
        [Decoder Embed],
        [Decoder Layer], [Decoder Layer], ...
    ]
    """
    blocks = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    # encoder embedding
    fnodes[indices[0] + 1].comment = f'==> start of encoder embedding'
    assert anchors[0].name == 'encoder embedding'
    blocks.append(fnodes[0:indices[1]])
    indices.pop(0)
    anchors.pop(0)
    # encoder layers
    lid = 0
    while anchors[0].name == 'encoder layer':
        start, end = indices[0], indices[1]
        fnodes[start + 1].comment = f'==> start of encoder layer {lid}'
        blocks.append(fnodes[start:end])
        indices.pop(0)
        anchors.pop(0)
        lid += 1
    # decoder embedding
    assert anchors[0].name == 'decoder embedding'
    blocks.append(fnodes[indices[0]:indices[1]])
    indices.pop(0)
    anchors.pop(0)
    # decoder layers
    lid = 0
    while len(indices) != 0:
        assert anchors[0].name == 'decoder layer'
        start, end = indices[0], indices[1] if len(indices) > 1 else len(fnodes)
        fnodes[start + 1].comment = f'==> start of decoder layer {lid}'
        blocks.append(fnodes[indices[0]:end])
        indices.pop(0)
        anchors.pop(0)
        lid += 1
    return blocks


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    _ = _group_to_blocks(graph.select(ntype=IRFwOperation))
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph

