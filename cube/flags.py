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

    # ============ code generation ===============
    use_nnfusion = os.environ.get('USE_NNFUSION')
    use_jit = os.environ.get('USE_JIT')


    # ============== runtime ====================
    dev_mode = os.environ.get('SINGLE_DEV_MODE')  # allow to use python xx.py
    
    # use automate mixture precision training, where weights, gradients
    # and optimizer status are kept in its original type (can be float32),
    # but the forward will be converted to float16 with torch.autocast API.
    use_amp = os.environ.get('USE_AMP')

