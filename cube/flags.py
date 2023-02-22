"""
Environment flags for compiling options
"""

import os


def _to_bool(s: str) -> bool:
    val = os.environ.get(s, default=0)
    return bool(int(val))


def _to_int(s: str, default=0) -> int:
    val = os.environ.get(s, default=default)
    return int(val)


class CompileFlag:

    # ============= loggings ===================
    log_transform = _to_bool('LOG_TRANSFORM')
    log_schedule = _to_bool('LOG_SCHEDULE')

    # ================ compiling ========================
    # worker sleep in seconds
    worker_sleep = _to_int('WORKER_SLEEP')
    disable_intra_rvd = _to_bool('DISABLE_INTRA_RVD')
    disable_inter_rvd =  _to_bool('DISABLE_INTER_RVD')
    disable_comm_fusion = _to_bool('DISABLE_COMM_FUSION')

    visualize_plan = _to_bool('VISUALIZE_PLAN')

    # ============ code generation ===============
    use_nnfusion = _to_bool('USE_NNFUSION')
    use_jit = _to_bool('USE_JIT')

    # ============== runtime ====================
    dev_mode = _to_bool('SINGLE_DEV_MODE')  # allow to use python xx.py

    # maximal reducer weight bytes for one allreduce
    max_reducer_bucket = _to_int('MAX_REDUCER_BUCKET', default=5e8)
    
    # use automate mixture precision training, where weights, gradients
    # and optimizer status are kept in its original data type (can be float32),
    # but some of the forward operators will be converted to float16.
    use_amp = _to_bool('USE_AMP')
