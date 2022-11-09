
from typing import Dict, Optional, List
import warnings

from cube.ir.cten import IRCell
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.adapter.adapter import IRWeightReducer

from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule import IRScheduleStrategy
from cube.ir.adapter.prim import IdentityPrim


class IRScheduleMix(IRScheduleStrategy):
    """
    1F1B Scheduling
    
    This treats model as a linear graph which can be
    grouped into continous stages.

    [Recv-Forward/Dataloader] Forward-Segment [Send-Forward]
    [Recv-Backward] Backward-Segment [Send-Backward]
    """

    def __init__(self, graph, nmicros: int):
        super().__init__(graph, nmicros)
        self.signature = 'cube.runtime.schedule.ScheduleMix.run'
        # forward body
        self.encoder_barriers: Dict[int, IRSegment] = dict()
        self.decoder_barriers: Dict[int, IRSegment] = dict()
        self.fsegments: Dict[int, IRSegment] = dict()
        # body forward recv adapter
        self.rfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # body forward send adapter 
        self.sfadapter: Dict[int, Optional[IRAdapter]] = dict()
        # body backward recv adapter
        self.rbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # body backward send adapter
        self.sbadapter: Dict[int, Optional[IRAdapter]] = dict()
        # encoder barrier backward prepare adapter
        self.enc_badapter: Dict[int, IRAdapter] = dict()
        # decoder barrier forward input prepare adapter
        self.dec_fadapter: Dict[int, IRAdapter] = dict()
        # decoder barrier backward input prepare adapter
        self.dec_badapter: Dict[int, IRAdapter] = dict()
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
        # each forward adapter has only one input and one output for each device
        for node in self.graph.nodes():
            if isinstance(node, IRAdapter) and node.forward:
                if len(set(node.outputs())) > 1 or len(set(node.inputs())) > 1:
                    warnings.warn(
                        "Detected one adapter has more than one input/output in stage transmission, "
                        "which is not safe for current scheduling implementation due to potential "
                        "mis-ordering of arguments. Better to use torch.cat and torch.chunk to "
                        "merge multiple tensors into one and unpack it at next stage."
                    )
        # each forward has corresponding backward
        assert all(fseg.mirror in self.segments for fseg in self.segments if fseg.isfw()), \
            "Require backward of each forward stage"

        fsegments: List[IRSegment] = [fseg for fseg in self.segments if fseg.isfw()]
        self.num_stages = len(fsegments) - 2

        shard_enc_sid, shard_dec_sid = (0, self.num_stages // 2)
        print(f'> shard encoder stage id: {shard_enc_sid} | shard decoder stage id: {shard_dec_sid} | num stages: {self.num_stages}')

        shard_enc, shard_dec = fsegments[0], fsegments[shard_dec_sid + 1]
        assert len(shard_enc.device) == len(shard_dec.device) and len(shard_enc.device) >= 4, (
            f"This scheduling can only be applied to number of devices >= 4"
        )
        pipe_stages = [seg for lid, seg in enumerate(fsegments) if lid not in (shard_enc_sid, shard_dec_sid + 1)]
    
        # setup shard encoder embedding
        assert len(self.recvers[shard_enc.mirror]) == 1
        for devid in shard_enc.device:
            self.encoder_barriers[devid] = shard_enc
            self.enc_badapter[devid] = self.recvers[shard_enc.mirror][0]
        # setup shard decoder embedding
        assert len(self.recvers[shard_dec]) == 1
        assert len(self.recvers[shard_dec.mirror]) == 1
        for devid in shard_dec.device:
            self.decoder_barriers[devid] = shard_dec
            self.dec_fadapter[devid] = self.recvers[shard_dec][0]
            self.dec_badapter[devid] = self.recvers[shard_dec.mirror][0]
        # pipeline stages
        for sid, stage in enumerate(pipe_stages):
            assert len(stage.device) == 1
            devid = stage.device[0]
            # forward body
            assert devid not in self.fsegments, f"Pipeline stage cannot be overlapped"
            self.fsegments[devid] = stage
            # forward recv
            if sid in (shard_enc_sid, shard_dec_sid):
                for adapter in self.recvers[stage]:
                    assert all(isinstance(prim, IdentityPrim) for prim in adapter.prims), (
                        f"stage {sid} got unexpected forward recv adapters: {self.recvers[stage]}"
                    )
                self.rfadapter[devid] = None
            else:
                assert len(self.recvers[stage]) == 1
                self.rfadapter[devid] = self.recvers[stage][0]
            # forward send
            if sid == shard_dec_sid - 1: # decoder recv broadcast
                assert len(self.senders[stage]) == 1
                self.sfadapter[devid] = None
            elif sid == self.num_stages - 1:
                assert len(self.senders[stage]) == 0
                self.sfadapter[devid] = None
            else:
                assert len(self.senders[stage]) == 1
                self.sfadapter[devid] = self.senders[stage][0]
            # backward recv
            if sid in (shard_dec_sid - 1, self.num_stages - 1):
                for adapter in self.recvers[stage.mirror]:
                    assert all(isinstance(prim, IdentityPrim) for prim in adapter.prims), (
                        f"stage {sid} got unexpected backward recv adapters: {self.recvers[stage]}"
                    )
                self.rbadapter[devid] = None
            else:
                assert len(self.recvers[stage.mirror]) == 1, \
                    f"stage {sid} got unexpected backward recv adapters: {self.recvers[stage.mirror]}"
                self.rbadapter[devid] = self.recvers[stage.mirror][0]
            # backward send:
            if sid == shard_dec_sid: # decoder broadcast
                assert len(self.senders[stage.mirror]) == 1
                self.sbadapter[devid] = None
            elif sid == shard_enc_sid: # encoder broadcast
                assert len(self.senders[stage.mirror]) == 1
                self.sbadapter[devid] = None
            else:
                self.sbadapter[devid] = self.senders[stage.mirror][0]
            
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
            encoder_barrier = self.encoder_barriers[devid],
            decoder_barrier = self.decoder_barriers[devid],
            segment = self.fsegments[devid],
            sfadapter = self.sfadapter[devid],
            rfadapter = self.rfadapter[devid],
            sbadapter = self.sbadapter[devid],
            rbadapter = self.rbadapter[devid],
            enc_badapter = self.enc_badapter[devid],
            dec_fadapter = self.dec_fadapter[devid],
            dec_badapter = self.dec_badapter[devid],
            dataloader = 'dataloader',
            stage_id = self.stage_id[devid],
            num_stages = self.num_stages,
            num_microbatch = self.nmicros,
            reducers = self.dev_reducers[devid],
            recompute = self.recompute
        )

    def __repr__(self) -> str:
        dscp = ''
        devices = self.devmesh[0]
        for devid in devices:
            dscp += (f"Interplaced Schedule: Stage[{self.stage_id[devid]}](dev {devid})(\n"
                     f"  encoder_barrier = {self.encoder_barriers[devid]}\n"
                     f"  decoder_barrier = {self.decoder_barriers[devid]}\n"
                     f"  segment = {self.fsegments[devid]}\n"
                     f"  send-fw = {self.sfadapter[devid]}\n"
                     f"  recv-fw = {self.rfadapter[devid]}\n"
                     f"  send-bw = {self.sbadapter[devid]}\n"
                     f"  recv-bw = {self.rbadapter[devid]}\n"
                     f"  enc_badapter = {self.enc_badapter[devid]}\n"
                     f"  dec_fadapter = {self.dec_fadapter[devid]}\n"
                     f"  dec_badapter = {self.dec_badapter[devid]}\n"
                     f")\n")
        return dscp
