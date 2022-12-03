
from typing import Dict, Optional, List
import warnings

from cube.ir.cten import IRCell
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.adapter.adapter import IRWeightReducer

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule.sched1f1b import IRSchedule1F1B


class IRScheduleNF1B(IRSchedule1F1B):
    """
    NF1B Scheduling
    
    This treats model as a linear graph which can be
    grouped into continous stages.

    [Recv-Forward/Dataloader] Forward-Segment [Send-Forward]
    [Recv-Backward] Backward-Segment [Send-Backward]
    """

    def __init__(self, graph, nmicros: int, recycle: int):
        super().__init__(graph, nmicros)
        self.signature = 'cube.runtime.schedule.ScheduleNF1B.run'
        # forward body
        self.fsegments: Dict[int, IRSegment] = dict()
        # forward send
        self.sfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # forward recv
        self.rfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # backard send
        self.sbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # backward recv
        self.rbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # num_stage
        self.num_stages: int = -1
        # stage id
        self.stage_id: Dict[int, int] = dict()
        # reducers
        self.dev_reducers: Dict[int, List[IRWeightReducer]] = dict()
        # recycle
        self.recycle = recycle

    def kwargs(self, devid: int) -> Dict[str, IRCell]:
        """
        return kwargs for runtime caller
        """
        return dict(
            segment = self.fsegments[devid],
            sfadapter = self.sfadapter[devid],
            rfadapter = self.rfadapter[devid],
            sbadapter = self.sbadapter[devid],
            rbadapter = self.rbadapter[devid],
            dataloader = 'dataloader',
            stage_id = self.stage_id[devid],
            num_stages = self.num_stages,
            num_microbatch = self.nmicros,
            recycle = self.recycle,
            reducers = self.dev_reducers[devid],
        )

    def __repr__(self) -> str:
        dscp = ''
        for mesh in self.devmesh:
            devid = mesh[0]
            # segment = self.segments[devid].to_str(skip_attr=True) if self.segment[mesh[0]] else None
            dscp += (f"NF1B Schedule: Stage[{self.stage_id[mesh[0]]}](dev {mesh})(\n"
                     f"  segment = {self.segments[devid]}\n"
                     f"  send-fw = {self.sfadapter[mesh[0]]}\n"
                     f"  recv-fw = {self.rfadapter[mesh[0]]}\n"
                     f"  send-bw = {self.sbadapter[mesh[0]]}\n"
                     f"  recv-bw = {self.rbadapter[mesh[0]]}\n"
                     f"  recycle = {self.recycle}\n"
                     f")\n")
        return dscp
