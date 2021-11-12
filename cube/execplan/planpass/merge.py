from typing import List

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.graph.operator.operator import IRBpOperation
from cube.schedule.su import SUType, ScheduleUnit


class MergeComputeSU(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Merge consecutive backward SUs. The forward SUs will
        also be merged if possible
        """
        for devid in execplan.devices():
            dev_seq = execplan.sequence(devid) + [None]
            pieces: List[ScheduleUnit] = list()
            for seqidx, su in enumerate(dev_seq):
                if su and su.stype in [SUType.Backward]:
                    allow_merge = len(pieces) == 0
                    for psu in pieces[::-1]:
                        if execplan.sugraph.happen_before(psu, su):
                            allow_merge = True
                            break
                    if allow_merge:
                        dev_seq[seqidx] = None
                        pieces.append(su)
                        continue
                # merged forward su
                if len(pieces) > 0:
                    fsus = [bsu.mirror for bsu in pieces][::-1]
                    if not all([fsu and (fsu in dev_seq) for fsu in fsus]):
                        raise RuntimeError("Expected same device fw-bw")
                    mfsu = MergeComputeSU._merge(fsus, devid)
                    mbsu = mfsu.mirror
                    # insert merged backward su
                    dev_seq[seqidx-1] = mbsu
                    fsus_idx = [dev_seq.index(fsu) for fsu in fsus]
                    # insert merged forward su
                    if max(fsus_idx) - min(fsus_idx) == len(fsus) - 1:
                        for fidx in fsus_idx:
                            dev_seq[fidx] = None
                        dev_seq[min(fsus_idx)] = mfsu
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
