
from typing import Dict, Tuple, Optional

from cube.ir.cten import IRCell
from cube.ir.adapter.adapter import IRAdapter

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy


class IRSchedule1F1B(IRScheduleStrategy):
    """
    1F1B Scheduling
    
    This treats model as a linear graph which can be
    grouped into continous stages.

    [Recv-Forward/Dataloader] Forward-Segment [Send-Forward]
    [Recv-Backward] Backward-Segment [Send-Backward]
    """

    def __init__(self, graph, nmicros: int, devmesh: Tuple[Tuple[int]]):
        super().__init__(graph, nmicros, devmesh)
        self.signature = 'cube.runtime.schedule.Schedule1F1B.run'
        # forward body
        self.segment: Dict[int, IRSegment] = dict()
        # forward send
        self.sfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # forward recv
        self.rfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # backard send
        self.sbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # backward recv
        self.rbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # num_stage
        self.num_stages: int = len(devmesh)
        # stage id
        self.stage_id: Dict[int, int] = dict()
        # recompute
        self.recompute = False

    def apply(self) -> IRGraph:
        self.segmentation()
        for gid, devices in enumerate(self.devmesh):
            for devid in devices:
                # forward recv
                self.rfadapter[devid] = None if gid == 0 else self.cross_groups[gid-1]
                # forward body
                self.segment[devid] = self.inner_groups[gid]
                # forward send
                if gid == len(self.devmesh)-1: assert self.cross_groups[gid] is None
                self.sfadapter[devid] = self.cross_groups[gid]
                # backward recv
                self.rbadapter[devid] = None if gid == len(self.devmesh)-1 else self.sfadapter[devid].mirror 
                # backward send
                self.sbadapter[devid] = None if gid == 0 else self.rfadapter[devid].mirror
                # stage id
                self.stage_id[devid] = gid
        return self.graph

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
            num_microbatch = self.nmicros,
            recompute = self.recompute
        )

    def __repr__(self) -> str:
        dscp = ''
        for mesh in self.devmesh:
            devid = mesh[0]
            segment = self.segment[devid].to_str(skip_attr=True) if self.segment[mesh[0]] else None
            dscp += (f"1F1B Schedule: Stage[{self.stage_id[mesh[0]]}](dev {mesh})(\n"
                     f"  segment = {segment}\n"
                     f"  send-fw = {self.sfadapter[mesh[0]]}\n"
                     f"  recv-fw = {self.rfadapter[mesh[0]]}\n"
                     f"  send-bw = {self.sbadapter[mesh[0]]}\n"
                     f"  recv-bw = {self.rbadapter[mesh[0]]}\n"
                     f")\n")
        return dscp
