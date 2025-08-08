_PLAN_NGPUS = None
_RUNTIME_NGPUS = None


def get_plan_ngpus():
    return _PLAN_NGPUS


def get_runtime_ngpus():
    return _RUNTIME_NGPUS


def initialize_parallel_state(plan_ngpus: int, runtime_ngpus: int):
    global _PLAN_NGPUS, _RUNTIME_NGPUS
    if _PLAN_NGPUS is None:
        _PLAN_NGPUS = plan_ngpus
    else:
        assert _PLAN_NGPUS == plan_ngpus, f"Expected plan_ngpus to be {_PLAN_NGPUS}, but got {plan_ngpus}"
    if _RUNTIME_NGPUS is None:
        _RUNTIME_NGPUS = runtime_ngpus
    else:
        assert _RUNTIME_NGPUS == runtime_ngpus, f"Expected runtime_ngpus to be {_RUNTIME_NGPUS}, but got {runtime_ngpus}"
