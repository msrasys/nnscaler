"""
Environment flags for compiling options
"""

import os


class CompileFlag:

    # ============= loggings ===================
    log_transform = os.environ.get('LOG_TRANSFORM')
    log_schedule = os.environ.get('LOG_SCHEDULE')

    
    # ================ compiling ========================
    # worker sleep in seconds
    worker_sleep = int(os.environ.get('WORKER_SLEEP')) if os.environ.get('WORKER_SLEEP') is not None else 0
    disable_intra_rvd = os.environ.get('DISABLE_INTRA_RVD')
    disable_inter_rvd = os.environ.get('DISABLE_INTER_RVD')
    disable_comm_fusion = os.environ.get('DISABLE_COMM_FUSION')

    # ============ code generation ===============
    use_nnfusion = os.environ.get('USE_NNFUSION')
    use_jit = os.environ.get('USE_JIT')


    # ============== runtime ====================
    dev_mode = os.environ.get('SINGLE_DEV_MODE')  # allow to use python xx.py

    # maximal reducer weight bytes for one allreduce
    max_reducer_bucket = int(os.environ.get('MAX_REDUCER_BUCKET', default=5e8))
    
    # use automate mixture precision training, where weights, gradients
    # and optimizer status are kept in its original data type (can be float32),
    # but some of the forward operators will be converted to float16.
    use_amp = True if os.environ.get('USE_AMP') else False

