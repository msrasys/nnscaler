#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.execplan import ExecutionPlan, ExecutionPlanType


class PlanPass:

    @staticmethod
    def apply(execplan: ExecutionPlanType) -> ExecutionPlanType:
        raise NotImplementedError
