
from typing import Dict, Tuple
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.cten import IRCell

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy


class IRSchedule1F1B(IRScheduleStrategy):
    """
    1F1B Scheduling
    
    This requires a micro-batch can be grouped into continguous segments
    which are placed on distinct device groups (refered as a stage):

    [Recv-Forward/Dataloader] Forward-Segment [Send-Forward]
    [Recv-Backward] Backward-Segment [Send-Backward]
    """

    def __init__(self, num_microbatch: int, devmesh: Tuple[Tuple[int]], recompute=False):
        super().__init__(num_microbatch, devmesh)
        self.signature = 'cube.runtime.schedule.Schedule1F1B.run'
        # forward body
        self.segment = dict()
        # forward send
        self.sfadapter = dict()
        # forward recv
        self.rfadapter = dict()
        # backard send
        self.sbadapter = dict()
        # backward recv
        self.rbadapter = dict()
        # num_stage
        self.num_stages = len(devmesh)
        # stage id
        self.stage_id = dict()
        # recompute
        self.recompute = recompute

    def apply(self, graph: IRGraph) -> IRGraph:
        graph = IRSchedule1F1B.segmentation(graph, self.devmesh)
        for stage_id, devices in enumerate(self.devmesh):
            for devid in devices:
                nodes = [n for n in graph.nodes() if devid in n.device]
                # forward body
                fsegments = [seg for seg in nodes if isinstance(seg, IRSegment) and seg.forward]
                assert len(fsegments) == 1, "find more than one segment."
                fsegment = fsegments[0]
                self.segment[devid] = fsegment
                fidx = nodes.index(fsegment)
                bidx = nodes.index(fsegment.mirror)
                # adapters
                adapters = [adapter for adapter in nodes if isinstance(adapter, IRAdapter)]
                # forward sends
                forward_sends = [n for n in adapters if n.forward and nodes.index(n) > fidx]
                if stage_id == self.num_stages - 1:
                    assert len(forward_sends) == 0, f"stage: {stage_id}: last stage should not send forward outputs"
                    self.sfadapter[devid] = None
                else:
                    assert len(forward_sends) == 1, f"stage: {stage_id}: last stage should not send forward outputs"
                    self.sfadapter[devid] = forward_sends[0]
                # forward recvs
                forward_recvs = [n for n in adapters if n.forward and nodes.index(n) < fidx]
                if stage_id == 0:
                    assert len(forward_recvs) == 0, f"stage: {stage_id}: first stage should not recv inputs"
                    self.rfadapter[devid] = None
                else:
                    assert len(forward_recvs) == 1, f"stage: {stage_id}: non-first stage should recv 1 inputs"
                    self.rfadapter[devid] = forward_recvs[0]
                # backward sends
                backward_sends = [n for n in adapters if not n.forward and nodes.index(n) > bidx]
                if stage_id == 0:
                    assert len(backward_sends) == 0, f"stage: {stage_id}: first stage should not send back gradient"
                    self.sbadapter[devid] = None
                else:
                    assert len(backward_sends) == 1, f"stage: {stage_id}: non-first stage should not send back gradient"
                    self.sbadapter[devid] = backward_sends[0]
                # backward recvs
                backward_recvs = [n for n in adapters if not n.forward and nodes.index(n) < bidx]
                if stage_id == self.num_stages - 1:
                    assert len(backward_recvs) == 0, f"stage: {stage_id}: last stage should not recv gradient"
                    self.rbadapter[devid] = None
                else:
                    assert len(backward_recvs) == 1, f"stage: {stage_id}: non-last stage should recv 1 gradient"
                    self.rbadapter[devid] = backward_recvs[0]
                # stage id
                self.stage_id[devid] = stage_id
        return graph

    def kwargs(self, devid: int) -> Dict[str, IRCell]:
        """
        return kwargs for runtime caller
        """
        return dict(
            segment = self.segment[devid],
            sfadapter = self.sfadapter[devid],
            rfadapter = self.rfadapter[devid],
            sbadapter = self.sbadapter[devid],
            rbadapter = self.rbadapter[devid],
            dataloader = 'dataloader',
            stage_id = self.stage_id[devid],
            num_stages = self.num_stages,
            num_microbatch = self.num_microbatch,
            recompute = self.recompute
        )

    def __repr__(self) -> str:
        dscp = ''
        for mesh in self.devmesh:
            dscp += (f"1F1B-Schedule-stage[{self.stage_id[mesh[0]]}](dev {mesh})(\n"
                     f"  segment = {self.segment[mesh[0]]}\n"
                     f"  send-fw = {self.sfadapter[mesh[0]]}\n"
                     f"  recv-fw = {self.rfadapter[mesh[0]]}\n"
                     f"  send-bw = {self.sbadapter[mesh[0]]}\n"
                     f"  recv-bw = {self.rbadapter[mesh[0]]}\n"
                     f")\n")
        return dscp
