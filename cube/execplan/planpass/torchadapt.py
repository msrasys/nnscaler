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

from typing import Dict

from cube.execplan import ExectuionPlan
from cube.graph.tensor import IRSubTensor, ValueMap
from cube.schedule.adapter.transform import IRTensorTransform
from cube.schedule.su import SUType, ScheduleUnit
from cube.execplan.planpass.planpass import PlanPass


class TorchRefAdapter(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan):
        # same device multiple reference
        multiref_fsus, multiref_fnodes = TorchRefAdapter.multi_ref_cells(execplan)
        for tid in multiref_fsus:
            print(f'multi-referred tensor id: {tid}')
            for devid in multiref_fsus[tid]:
                for fsu in multiref_fsus[tid][devid]:
                    print(f'dev {devid}: {fsu}')


        for tid in multiref_fsus:
            # check chunk num for each device
            total_ops = set()
            for devid in multiref_fnodes[tid]:
                for op in multiref_fnodes[tid][devid]:
                    total_ops.add(op._id)
            total_ops = list(total_ops)
            num_ops = len(total_ops)
            # how many ops are computed for each device
            dev_ops = dict()
            for devid in multiref_fnodes[tid]:
                op_index = list()
                for op in multiref_fnodes[tid][devid]:
                    op_index.append(total_ops.index(op._id))
                cnt = len(op_index)
                if cnt != 1 and cnt != num_ops:
                    raise NotImplementedError("Only support even chunk for multi-ref")
                dev_ops[devid] = op_index

            for idx, devid in enumerate(multiref_fsus[tid]):
                # the value map should be op_num / total_ops
                op_index = dev_ops[devid]
                if len(op_index) == num_ops:
                    grad_idx, grad_num = 0, 1
                elif len(op_index) == 1:
                    grad_idx, grad_num = op_index[0], num_ops

                # the first forward, the last backward
                fsu = multiref_fsus[tid][devid][0]
                ftensor = None
                for input in fsu.inputs():
                    if isinstance(input, IRSubTensor):
                        if input._id == tid:
                            ftensor = input
                            break
                if ftensor is None:
                    raise RuntimeError("Internal Error: fsu not found input tensor")
                grad = ftensor.parent.grad.select(
                    indmap = ftensor.indmap,
                    valmap = ValueMap(grad_idx, grad_num),
                    shape = ftensor.shape
                )
                rm_grad = TorchRefAdapter.set_grad(fsu, ftensor, grad)
                TorchRefAdapter.replace_all(execplan, rm_grad, grad, devid)

                # all the other reference place: set grad to none
                for fsu in multiref_fsus[tid][devid][1:]:
                    rm_grad = TorchRefAdapter.set_grad(fsu, ftensor, grad=None)
                    TorchRefAdapter.replace_all(execplan, rm_grad, None, devid)

        print(execplan)

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
    def multi_ref_cells(execplan: ExectuionPlan) -> Dict:
        """
        Return:
        {
            sub_tensor id:
                device id:
                    [forward su or forward node]
        }
        """
        fnodes = dict()
        fsus = dict()
        for devid in execplan.devices():
            for fsu in execplan.sequence(devid):
                if fsu.stype == SUType.Forward:
                    for input in fsu.inputs():
                        if isinstance(input, IRSubTensor):
                            tid = input._id
                            if tid not in fnodes:
                                fnodes[tid] = dict()
                                fsus[tid] = dict()
                            if devid not in fnodes[tid]:
                                fnodes[tid][devid] = list()
                                fsus[tid][devid] = list()
                            fsus[tid][devid].append(fsu)
                            for node in fsu.nodes():
                                if input in node.inputs():
                                    fnodes[tid][devid].append(node)
        multiref_fnodes = dict()
        multiref_sus = dict()
        for tid in fnodes:
            for devid in fnodes[tid]:
                if len(fnodes[tid][devid]) != 1:
                    multiref_sus[tid] = fnodes[tid]
                    multiref_fnodes[tid] = fsus[tid]
                    break
        return multiref_fnodes, multiref_sus
 

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
        if not len(fsu.nodes()) == 1:
            raise RuntimeError("TorchAdapt should call before merge")
        fnode = fsu.nodes(0)
        findex = fnode.inputs().index(input)
        fnode.inputs(findex).grad = grad
        # backward SU
        bsu = fsu.mirror
        bindex = bsu.inputs().index(input)
        bin = bsu.inputs(bindex)
        try:
            gindex = bsu.outputs().index(bin.grad)
        except ValueError:
            raise RuntimeError(
                (f"Internal Error: cannot find given grad in bsu: {bsu}:\n"
                 f"gradient given tensor: {bin}, grad: {bin.grad}")
            )
        removed_grad = bin.grad
        bin.grad = grad
        bsu.set_output(gindex, grad)
        return removed_grad

    @staticmethod
    def replace_all(execplan: ExectuionPlan, src: IRSubTensor, dst, devid: int):
        for su in execplan.sequence(devid):
            # pair removement for p2p will already remove su
            if su not in execplan.at(devid):
                continue
            rm_su = None
            if src in su.inputs():
                if len(su.inputs()) == 1 and dst is None:
                    execplan.at(devid).remove(su)
                    execplan.sugraph.sequence.remove(su)
                    rm_su = su
                else:
                    index = su.inputs().index(src)
                    su.set_input(index, dst)
            if src in su.outputs():
                if len(su.outputs()) == 1 and dst is None:
                    execplan.at(devid).remove(su)
                    execplan.sugraph.sequence.remove(su)
                    rm_su = su
                else:
                    index = su.outputs().index(src)
                    su.set_output(index, dst)
            # pair removement
            if rm_su is not None and rm_su.stype == SUType.P2P:
                mirror = rm_su.mirror
                dev = mirror.device[0]
                if mirror in execplan.at(dev):
                    execplan.at(dev).remove(mirror)
                    execplan.sugraph.sequence.remove(mirror)
