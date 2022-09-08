
from typing import Dict, Optional, List

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

    def __init__(self, graph, nmicros: int):
        super().__init__(graph, nmicros)
        self.signature = 'cube.runtime.schedule.Schedule1F1B.run'
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
        # recompute
        self.recompute = False


    def apply(self) -> IRGraph:
        self.mesh()
        # each forward has corresponding backward
        assert all(fseg.mirror in self.segments for fseg in self.segments if fseg.isfw()), \
            "Require backward of each forward stage"
        # stage doesn't share devices
        fsegments: List[IRSegment] = [fseg for fseg in self.segments if fseg.isfw()]
        self.num_stages = len(fsegments)
        for sid, fseg in enumerate(fsegments):
            for devid in fseg.device:
                # forward body
                assert devid not in self.fsegments, "One device cannot have multiple forward stages"
                self.fsegments[devid] = fseg
                # forward recv / backward send
                assert len(self.recvers[fseg]) <= 1, "Corss-stage adapter can only be one"
                if sid == 0:
                    assert len(self.recvers[fseg]) == 0, "Expect no forward send at first stage"
                    assert len(self.senders[fseg.mirror]) == 0, "Expect no backward send at first stage"
                else:
                    assert len(self.recvers[fseg]) == 1, "Expect one forward recv at non-first stage"
                    assert len(self.senders[fseg.mirror]) == 1, "Expect one backward send at non-first stage"
                self.rfadapter[devid] = None if sid == 0 else self.recvers[fseg][0]
                self.sbadapter[devid] = None if sid == 0 else self.senders[fseg.mirror][0]
                # forward send / backward recv
                if sid == self.num_stages - 1:
                    assert len(self.senders[fseg]) == 0, "Expect no forward send at last stage"
                    assert len(self.recvers[fseg.mirror]) == 0, "Expect no backward recv at last stage"
                else:
                    assert len(self.senders[fseg]) == 1, "Expect no forward send at last stage"
                    assert len(self.recvers[fseg.mirror]) == 1, "Expect no forward send at last stage"
                self.sfadapter[devid] = None if sid == self.num_stages - 1 else self.senders[fseg][0]
                self.rbadapter[devid] = None if sid == self.num_stages - 1 else self.recvers[fseg.mirror][0]
                # stage id
                self.stage_id[devid] = sid

        return self.graph

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
