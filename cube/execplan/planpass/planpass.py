from cube.execplan import ExectuionPlan


class PlanPass:

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        raise NotImplementedError
