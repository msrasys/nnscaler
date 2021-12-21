"""
Gradient Allreduce Fusion
"""
from typing import Dict, Tuple, List
import sys
import copy


from cube.graph.operator.operator import IROptimOperation
from cube.graph.tensor import IRSubTensor, ValueMap

from cube.execplan import ExectuionPlan
from cube.schedule.su import SUType, ScheduleUnit
from cube.execplan.planpass.planpass import PlanPass


class WeightGradAllreduceFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Apply weight gradient allreduce fusion
        """
        reducers: Dict[Tuple[int], List[IRSubTensor]] = dict()
        weights, params = WeightGradAllreduceFusion._get_weight_grads(execplan)
        for param_id in params:
            grads = params[param_id]
            ranks = list(grads.keys())
            ranks.sort()
            ranks = tuple(ranks)  # ranks are used for group
            if len(ranks) == 1:
                continue
            if ranks not in reducers:
                reducers[ranks] = list()
            reducers[ranks].append(weights[param_id])
        # generate reducer for each rank
        for ranks in reducers:
            weights = reducers[ranks]
            # even though some ranks don't need allreduce,
            # pytorch still requires each rank simutaneously call the
            # communication group initialization
            for devid in execplan.devices():
                dev_weights = copy.copy(weights)
                for idx, weight in enumerate(dev_weights):
                    if devid not in params[weight._id]:
                        dev_weights[idx] = None
                dev_weights = [w for w in dev_weights if w is not None]
                opt_op = IROptimOperation(dev_weights, ranks)
                reduce_su = ScheduleUnit([opt_op], SUType.Optimizer)
                reduce_su.device = devid
                execplan.at(devid).append(reduce_su)
        return execplan

    @staticmethod
    def _get_weight_grads(execplan: ExectuionPlan) -> Dict:
        """
        Get weight and gradient
        
        weights: Dict[param_id: int, IRSubTensor]
        grads  : Dict[param_id: int, Dict[device: int, List[grad: IRSubTensor]]]

        """
        grads = dict()
        weights = dict()
        for devid in execplan.devices():
            bsus = [su for su in execplan.sequence(devid) if su.stype == SUType.Backward]
            for bsu in bsus:
                # bsu has only one node
                for input in bsu.inputs():
                    if isinstance(input, IRSubTensor) and input.is_param():
                        grad = input.grad
                        if grad is None:
                            print(input.name, input)
                            print(grad)
                            assert grad is not None
                        # nothing to sync
                        if grad.valmap == ValueMap(0, 1):
                            continue
                        if input._id not in grads:
                            grads[input._id] = dict()
                            weights[input._id] = input
                        if devid not in grads[input._id]:
                            grads[input._id][devid] = list()
                        if grad in grads[input._id][devid]:
                            raise RuntimeError("Already logged grad?")
                        grads[input._id][devid].append(grad)
        return weights, grads
