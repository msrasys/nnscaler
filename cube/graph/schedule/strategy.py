from typing import Tuple, Dict, Any, List
from cube.graph.graph import IRGraph, IRSegment
from cube.ir.adapter.adapter import IRAdapter, IRWeightReducer
from cube.ir.cten import IRCell


class IRScheduleStrategy:

    def __init__(self, graph: IRGraph, nmicros: int) -> None:
        self.graph : IRGraph = graph
        self.nmicros : int = nmicros
        self.devmesh: List[Tuple[int]] = []
        # preprocess before segments
        self.pre_process: List[IRCell] = []
        self.segments: List[IRSegment] = []
        # the recver adapters for this segment
        self.recvers: Dict[IRSegment, List[IRAdapter]] = dict()
        # the sender adapters for this segment
        self.senders: Dict[IRSegment, List[IRAdapter]] = dict()
        # postprocess of weight reducers
        self.reducers: List[IRWeightReducer] = []
        self.signature: str = ''

    def apply(self, graph: IRGraph) -> IRGraph:
        raise NotImplementedError

    def kwargs(self, device: int) -> Dict[str, Any]:
        raise NotImplementedError

    def mesh(self) -> List[List[int]]:
        """!
        Group operators into segments corresponding to graph stage.
        Reorder adapter output to match with segment input order
        """
        for segment in self.graph.nodes():
            if isinstance(segment, IRSegment):
                self.segments.append(segment)
            self.recvers[segment] = []
            self.senders[segment] = []
        
        for adapter in self.graph.nodes():
            if isinstance(adapter, IRAdapter):
                for segment in self.segments:
                    if self.graph.depends(adapter, segment):
                        self.recvers[segment].append(adapter)
                    elif self.graph.depends(segment, adapter):
                        self.senders[segment].append(adapter)
            if isinstance(adapter, IRWeightReducer):
                self.reducers.append(adapter)
