from typing import Tuple, Dict, Any
from cube.graph.graph import IRGraph
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.cten import IRCell
from cube.ir.operator import IRFwOperation


class IRScheduleStrategy:

    def __init__(self, num_microbatch: int, devmesh: Tuple[Tuple[int]]) -> None:
        self.num_microbatch = num_microbatch
        self.devmesh = devmesh
        self.signature: str = ''

    def apply(self, graph: IRGraph) -> IRGraph:
        raise NotImplementedError

    def kwargs(self, device: int) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def segmentation(graph: IRGraph, devmesh: Tuple[Tuple[int]]) -> IRGraph:
        """
        Utilities for grouping operators into segments with device mesh
        """
        stages = [[] for _ in range(len(devmesh))]
        for node in graph.nodes():
            for meshid, devices in enumerate(devmesh):
                if set(node.device).issubset(set(devices)):
                    stages[meshid].append(node)
                    break
        # grouping
        for stage in stages:
            fconsecutive, bconsecutive = [], []
            for node in stage:
                if isinstance(node, IRFwOperation) or (isinstance(node, IRAdapter) and node.forward):
                    fconsecutive.append(node)
                    if node.mirror:
                        bconsecutive.append(node.mirror)
                else:
                    assert len(fconsecutive) == len(bconsecutive) or len(bconsecutive) == 0, 'mismatch number of forward and backward operators.'
                    if len(fconsecutive) != 0:
                        fsegment = graph.group(fconsecutive)
                    if len(bconsecutive) != 0:
                        bsegment = graph.group(bconsecutive[::-1])
                        IRCell.make_pair(fsegment, bsegment)
                    fconsecutive, bconsecutive = [], []
        return graph

    @staticmethod
    def merging(graph: IRGraph) -> IRGraph:
        """
        merge the adapters into one
        """
        pass
