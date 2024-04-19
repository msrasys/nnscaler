from typing import Optional
import logging
from nnscaler import runtime
from nnscaler import utils

from nnscaler import profiler
from nnscaler.profiler.timer import CudaTimer

from nnscaler.compiler import SemanticModel, compile

from nnscaler.utils import load_model, load_default_schedule, load_eval_schedule
from nnscaler.utils import accum_mode

from nnscaler.flags import CompileFlag

from .version import __version__


def _check_torch_version():
    import torch
    torch_version = str(torch.__version__).split('+')[0]
    torch_version = float('.'.join(torch_version.split('.')[:2]))
    if torch_version < 1.12:
        logging.warn(f"expected PyTorch version >= 1.12 but got {torch_version}")


def init():
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()


def set_logger_level(level):
    """Set the logger level with predefined logging format.
    
    Args:
        level (int): the level of the logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


_check_torch_version()
