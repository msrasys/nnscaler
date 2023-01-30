
from typing import Dict, Optional, List
import warnings

from cube.ir.cten import IRCell
from cube.ir.adapter.adapter import IRAdapter

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy


class IRScheduleInfer(IRScheduleStrategy):
    """
    1F1B Scheduling
    
    This treats model as a linear graph which can be
    grouped into continous stages.

    [Recv-Forward/Dataloader] Forward-Segment [Send-Forward]
    [Recv-Backward] Backward-Segment [Send-Backward]
    """

    def __init__(self, graph, nmicros: int):
        super().__init__(graph, nmicros)
        self.signature = 'cube.runtime.schedule.ScheduleInfer.run'
        # forward body
        self.fsegments: Dict[int, IRSegment] = dict()
        # forward send
        self.sfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # forward recv
        self.rfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # num_stage
        self.num_stages: int = -1

    def apply(self) -> IRGraph:
        self.mesh()
        for node in self.graph.nodes():
            if isinstance(node, IRAdapter) and node.forward:
                if len(set(node.outputs())) > 1 or len(set(node.inputs())) > 1:
                    warnings.warn(
                        "Detected one adapter has more than one input/output in stage transmission, "
                        "which is not safe for current scheduling implementation due to potential "
                        "mis-ordering of arguments. Better to use torch.cat and torch.chunk to "
                        "merge multiple tensors into one and unpack it at next stage."
                    )
        # no backward
        for seg in self.graph.select(ntype=IRSegment):
            assert seg.isfw(), "Detected backward, which should not exist in inference"
        # stage doesn't share devices
        fsegments: List[IRSegment] = [fseg for fseg in self.segments if fseg.isfw()]
        self.num_stages = len(fsegments)
        for sid, fseg in enumerate(fsegments):
            for devid in fseg.device:
                # forward body
                assert devid not in self.fsegments, "One device cannot have multiple forward stages"
                self.fsegments[devid] = fseg
                if sid == 0:
                    assert len(self.recvers[fseg]) == 0, "Expect no forward send at first stage"
                else:
                    assert len(self.recvers[fseg]) == 1, "Expect one forward recv at non-first stage"
                self.rfadapter[devid] = None if sid == 0 else self.recvers[fseg][0]
                # forward send
                if sid == self.num_stages - 1:
                    assert len(self.senders[fseg]) == 0, "Expect no forward send at last stage"
                else:
                    assert len(self.senders[fseg]) == 1, "Expect no forward send at last stage"
                self.sfadapter[devid] = None if sid == self.num_stages - 1 else self.senders[fseg][0]

        return self.graph
    
    def kwargs(self, devid: int) -> Dict[str, IRCell]:
        """
        return kwargs for runtime caller
        """
        return dict(
            segment = self.fsegments[devid],
            sfadapter = self.sfadapter[devid],
            rfadapter = self.rfadapter[devid],
            dataloader = 'dataloader',
            num_microbatch = self.nmicros,
        )
    
    def __repr__(self) -> str:
        dscp = ''
        for mesh in self.devmesh:
            devid = mesh[0]
            # segment = self.segments[devid].to_str(skip_attr=True) if self.segment[mesh[0]] else None
            dscp += (f"GPipe Infer Schedule: Stage[{self.stage_id[mesh[0]]}](dev {mesh})(\n"
                     f"  segment = {self.segments[devid]}\n"
                     f"  send-fw = {self.sfadapter[mesh[0]]}\n"
                     f"  recv-fw = {self.rfadapter[mesh[0]]}\n"
                     f")\n")
        return dscp
