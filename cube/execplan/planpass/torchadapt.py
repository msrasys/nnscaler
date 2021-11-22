"""
PyTorch Adapter for multi-branch reference

If a tensor is the input for multiple operators:

    the gradient of this tensor will be value splitted for each op-backward.

However, in pytorch, the gradient is accumulated by default, this
will cause inconsistent behaviour for transoform SU when the referred
operators are on the same device or not.

For the situation when the referred operators are on different devices:
    Nothing happens

For the situation when the referred operators are on same device:
    The gradient will change to match `auto accumulation` semantics.
    For first referred op: grad will be set to ValueMap(idx, num_referred_devices)
    For other referred op: grad is set to None
"""

from typing import Dict, List

from cube.execplan import ExectuionPlan
from cube.graph.tensor import IRSubTensor, ValueMap
from cube.schedule.adapter.transform import IRTensorTransform
from cube.schedule.su import SUType, ScheduleUnit
from cube.execplan.planpass.planpass import PlanPass


class TorchRefAdapter(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan):
        # same device multiple reference
        multiref = TorchRefAdapter.gather_tensor(execplan)
        for tid in multiref:
            print(f'tensor id: {tid}')
            for devid in multiref[tid]:
                for fsu in multiref[tid][devid]:
                    print(f'dev {devid}: {fsu}')

        for tid in multiref:
            grad_num = len(multiref[tid])
            for idx, devid in enumerate(multiref[tid]):
                # the first forward, the last backward
                fsu = multiref[tid][devid][0]
                ftensor = None
                for input in fsu.inputs():
                    if isinstance(input, IRSubTensor):
                        if input._id == tid:
                            ftensor = input
                            break
                if ftensor is None:
                    raise RuntimeError("Internal Error: fsu not found input tensor")
                grad = ftensor.parent.grad.select(
                    indices = ftensor.indices,
                    val_map = ValueMap(idx, grad_num),
                    shape = ftensor.shape
                )
                rm_grad = TorchRefAdapter.set_grad(fsu, ftensor, grad)
                TorchRefAdapter.replace_all(execplan, rm_grad, grad)

                # all the other reference place: set grad to none
                for fsu in multiref[tid][devid][1:]:
                    rm_grad = TorchRefAdapter.set_grad(fsu, ftensor, grad=None)
                    TorchRefAdapter.replace_all(execplan, rm_grad, None)

        # reset select and merge adapters
        for devid in execplan.devices():
            for idx, su in enumerate(execplan.sequence(devid)):
                if su.stype == SUType.Transform:
                    ins = [input for input in su.inputs() if input is not None]
                    ous = [ou for ou in su.outputs() if ou is not None]
                    if len(ins) < len(su.inputs()) or len(ous) < len(su.outputs()):
                        for ou in ous:
                            if ou in ins:
                                break
                        trans = IRTensorTransform(
                            src_tensors=ins, dst_tensors=ous
                        )
                        trans_su = ScheduleUnit([trans], SUType.Transform, name='trans')
                        trans_su.device = devid
                        if len(trans_su.outputs()) == 0:
                            # meaning outputs in inputs
                            execplan.at(devid).remove(su)
                            execplan.sugraph.sequence.remove(su)
                        else:
                            execplan.at(devid)[idx] = trans_su
                            suidx = execplan.sugraph.sequence.index(su)
                            execplan.sugraph.sequence[suidx] = trans_su
        execplan.sugraph.reset_dependency(execplan.sugraph.sus())
        return execplan

    @staticmethod
    def gather_tensor(execplan: ExectuionPlan) -> Dict:
        """
        Return:
        {
            sub_tensor id:
                device id:
                    [forward su]
        }
        """
        fwsus = dict()
        for devid in execplan.devices():
            for fsu in execplan.sequence(devid):
                if fsu.stype == SUType.Forward:
                    for input in fsu.inputs():
                        if isinstance(input, IRSubTensor):
                            tid = input._id
                            if tid not in fwsus:
                                fwsus[tid] = dict()
                            if devid not in fwsus[tid]:
                                fwsus[tid][devid] = list()
                            fwsus[tid][devid].append(fsu)
        multiref = dict()
        for tid in fwsus:
            for devid in fwsus[tid]:
                if len(fwsus[tid][devid]) != 1:
                    multiref[tid] = fwsus[tid]
                    break
        return multiref

    @staticmethod
    def set_grad(fsu: ScheduleUnit, input: IRSubTensor, grad):
        """
        Return removed grad
        """
        if not isinstance(fsu, ScheduleUnit) or fsu.stype != SUType.Forward:
            raise TypeError("Require SU to be forward SU")
        # forward SU
        findex = fsu.inputs().index(input)
        fsu.inputs(findex).grad = grad
        # backward SU
        bsu = fsu.mirror
        bindex = bsu.inputs().index(input)
        bin = bsu.inputs(bindex)
        gindex = bsu.outputs().index(bin.grad)
        removed_grad = bin.grad
        bin.grad = grad
        bsu.set_output(gindex, grad)
        return removed_grad

    @staticmethod
    def replace_all(execplan: ExectuionPlan, src: IRSubTensor, dst):
        for devid in execplan.devices():
            for su in execplan.sequence(devid):
                if src in su.inputs():
                    if len(su.inputs()) == 1:
                        execplan.at(devid).remove(su)
                        execplan.sugraph.sequence.remove(su)
                    else:
                        index = su.inputs().index(src)
                        su.set_input(index, dst)
                if src in su.outputs():
                    if len(su.outputs()) == 1:
                        execplan.at(devid).remove(su)
                        execplan.sugraph.sequence.remove(su)
                    else:
                        index = su.outputs().index(src)
                        su.set_output(index, dst)
