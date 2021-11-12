"""
Gradient Allreduce Fusion
"""
from typing import Dict, Tuple, List
from cube.graph.operator.operator import IROptimOperation

from cube.graph.tensor import IRSubTensor

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
        params = WeightGradAllreduceFusion._get_weight_grads(execplan)
        for param in params:
            grads = params[param]
            ranks = tuple(grads.keys())  # ranks are used for group
            grads = [grads[devid][-1] for devid in grads]
            if len(ranks) == 1:
                continue
            if ranks not in reducers:
                reducers[ranks] = list()
            for grad in grads:
                reducers[ranks].append(grad)
        # generate reducer for each rank
        for ranks in reducers:
            grads = reducers[ranks]
            # even though some ranks don't need allreduce,
            # pytorch still requires each rank simutaneously call the
            # communication group initialization
            for devid in execplan.devices():
                opt_op = IROptimOperation(grads, ranks)
                reduce_su = ScheduleUnit([opt_op], SUType.Optimizer)
                reduce_su.device = devid
                execplan.at(devid).append(reduce_su)
        return execplan

    @staticmethod
    def _get_weight_grads(execplan: ExectuionPlan) -> Dict:
        """
        Get weight gradient
        
        Return Dict[IRSubTensor, Dict[int, List[IRSubTensor]]]
               (grads = params[param][device])
        """
        # grad = params[param][device]
        params = dict()
        for devid in execplan.devices():
            bsus = [su for su in execplan.sequence(devid) if su.stype == SUType.Backward]
            for bsu in bsus:
                # bsu has only one node
                for input in bsu.inputs():
                    if isinstance(input, IRSubTensor) and input.is_param():
                        if input not in params:
                            params[input] = {devid : list()}
                        grad = input.grad
                        assert grad is not None
                        if grad in params[input][devid]:
                            raise RuntimeError("Already logged grad?")
                        params[input][devid].append(grad)
        return params
