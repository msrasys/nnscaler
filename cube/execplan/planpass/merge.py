from typing import List

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.graph.operator.operator import IRBpOperation
from cube.schedule.su import SUType, ScheduleUnit


class MergeComputeSU(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Merge consecutive forward SUs
        """
        for devid in execplan.devices():
            dev_seq = execplan.sequence(devid)
            pieces: List[ScheduleUnit] = list()
            for seqidx, su in enumerate(execplan.sequence(devid)):
                if su.stype in [SUType.Forward]:
                    allow_merge = len(pieces) == 0
                    for psu in pieces[::-1]:
                        if execplan.sugraph.happen_before(psu, su):
                            allow_merge = True
                            break
                    if allow_merge:
                        dev_seq[seqidx] = None
                        if su.mirror is not None:
                            if su.mirror not in dev_seq:
                                raise RuntimeError(
                                    "Expected backward and forward on same device")
                            idx = dev_seq.index(su.mirror)
                            dev_seq[idx] = None
                        pieces.append(su)
                        continue
                # merge pieces
                if len(pieces) > 0:
                    # merged forward su
                    mfsu = MergeComputeSU._merge(pieces, devid)
                    mbsu = mfsu.mirror
                    # insert merged forward su
                    dev_seq[seqidx-1] = mfsu
                    # insert merged backward su
                    bidx = len(dev_seq)
                    for fsu in pieces:
                        bsu = fsu.mirror
                        if bsu is not None:
                            idx = execplan.sequence(devid).index(bsu)
                            dev_seq[idx] = None
                            bidx = min(bidx, idx)
                    if bidx != len(dev_seq):
                        dev_seq[bidx] = mbsu
                pieces = list()
            dev_seq = [su for su in dev_seq if su is not None]
            execplan.set(devid, dev_seq)
        return execplan

    @staticmethod
    def _merge(pieces: List[ScheduleUnit], devid: int) -> ScheduleUnit:
        """
        Merge a list of SU into one.
        """
        if len(pieces) == 1:
            return pieces[0]
        fnodes = list()
        for fsu in pieces:
            fnodes += fsu.nodes()
        # TODO: fix multi-branch
        mfsu = ScheduleUnit(fnodes, SUType.Forward, name='fsu')
        mfsu.device = devid

        # merged backward su
        mbnode = IRBpOperation(
            data_num=len(mfsu.inputs()),
            grad_num=len(mfsu.outputs())
        )
        for idx, fin in enumerate(mfsu.inputs()):
            mbnode.set_data(idx, fin)
            mbnode.set_output(idx, fin.grad)
        for idx, fout in enumerate(mfsu.outputs()):
            mbnode.set_grad(idx, fout.grad)
        mbsu = ScheduleUnit([mbnode], SUType.Backward, name='bsu')
        mbsu.device = devid

        mfsu.mirror = mbsu
        mbsu.mirror = mfsu
        return mfsu
