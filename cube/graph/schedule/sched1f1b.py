
from typing import Dict, Optional, List
import numpy as np

from cube.ir.cten import IRCell
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.adapter.adapter import IRWeightReducer

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy


def reorder_inputs_outputs(node: IRCell, also_mirror: bool = True):
    """
    Inplacement reorder forward node inputs and outputs by tensor ID.

    The order of inputs/outputs in backward can also be reordered correspondingly.
    """
    assert isinstance(node, (IRCell, IRSegment))
    inputs_tid = np.array([t.tid for t in node.inputs()])
    inputs_idx = np.argsort(inputs_tid)
    inputs = [node.input(idx) for idx in inputs_idx]
    outputs_tid = np.array([t.tid for t in node.outputs()])
    outputs_idx = np.argsort(outputs_tid)
    outputs = [node.output(idx) for idx in outputs_idx]
    node._inputs = inputs
    node._outputs = outputs
    bnode: IRCell = node.mirror
    if also_mirror and isinstance(bnode, IRCell):
        if isinstance(bnode, IRSegment):
            assert len(bnode.inputs()) == len(node.outputs()), f"fnode:\n{node}\nbnode:\n{bnode}"
            bnode._inputs = [bnode.input(idx) for idx in outputs_idx]
            assert len(bnode.outputs()) == len(node.inputs()), f"fnode:\n{node}\nbnode:\n{bnode}"
            bnode._outputs = [bnode.output(idx) for idx in inputs_idx]
        else:
            # setup input
            ftids = [t.tid for t in node.outputs()]
            grads = [t.grad for t in node.outputs()]
            actvs = []
            for t in bnode.inputs():
                assert t in grads, f"backward gradient is not required by its forward node "
                actvs.append(ftids[grads.index(t)])
            inputs_idx = np.argsort(np.array(actvs))
            inputs = [bnode.input(idx) for idx in inputs_idx]
            bnode._inputs = inputs
            # setup outputs
            ftids = [t.tid for t in node.inputs()]
            grads = [t.grad for t in node.outputs()]
            actvs = []
            for t in bnode.outputs():
                assert t in grads, f"backward gradient is not required by its forward"
                actvs.append(ftids[grads.index(t)])
            outputs_idx = np.argsort(np.array(actvs))
            outputs = [bnode.output(idx) for idx in outputs_idx]
            bnode._outputs = outputs


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
        # reducers
        self.dev_reducers: Dict[int, List[IRWeightReducer]] = dict()
        # recompute
        self.recompute = False


    def apply(self) -> IRGraph:
        self.mesh()
        # reorder input and output by tensor id
        for node in self.graph.nodes():
            if isinstance(node, IRSegment) and node.isfw():
                reorder_inputs_outputs(node)
            elif isinstance(node, IRAdapter) and node.forward:
                reorder_inputs_outputs(node)
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
                # weight reducer
                self.dev_reducers[devid] = [reducer for reducer in self.reducers if devid in reducer.device]
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
            reducers = self.dev_reducers[devid],
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
