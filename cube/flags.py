"""
Environment flags for compiling options
"""

import os


class CompileFlag:

    # ============== runtime ====================
    dev_mode = os.environ.get('SINGLE_DEV_MODE')  # allow to use python xx.py

    # ============= loggings ===================
    log_transform = os.environ.get('LOG_TRANSFORM')

    # ============ code generation ===============
    use_nnfusion = os.environ.get('USE_NNFUSION')
    use_jit = os.environ.get('USE_JIT')

