from typing import Optional
import logging
from cube import runtime

from cube import profiler
from cube.profiler.timer import CudaTimer

from cube.compiler import SemanticModel, compile

from cube.utils import load_model, load_default_schedule, load_eval_schedule
from cube.utils import accum_mode

from cube.flags import CompileFlag


def _check_torch_version():
    import torch
    torch_version = str(torch.__version__).split('+')[0]
    torch_version = float('.'.join(torch_version.split('.')[:2]))
    if torch_version < 1.12:
        logging.warn(f"expected PyTorch version >= 1.12 but got {torch_version}")


def init():
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()


def _init_logger():
    logging.basicConfig(level=logging.WARN)

    level = lambda flag: logging.INFO if flag else logging.WARN

    logging.getLogger('cube.parser').setLevel(
        level(CompileFlag.log_parser)
    )
    logging.getLogger('cube.prim').setLevel(
        level(CompileFlag.log_transform)
    )
    logging.getLogger('cube.adapter').setLevel(
        level(CompileFlag.log_adapter)
    )
    logging.getLogger('cube.execplan').setLevel(
        level(CompileFlag.log_execplan)
    )
    logging.getLogger('cube.codegen').setLevel(
        level(CompileFlag.log_codegen)
    )
    logging.getLogger('cube.runtime').setLevel(
        level(CompileFlag.log_runtime)
    )
    logging.getLogger('cube.profiler').setLevel(
        level(CompileFlag.log_profiler)
    )
    logging.getLogger('cube.compiler').setLevel(
        logging.INFO
    )


def set_logger_level(name: Optional[str], level):
    """Set the logger level of cube.
    
    Args:
        name (Optional[str]): the name of the logger, can be one of
            'cube.parser', 'cube.policy', 'cube.adapter',
            'cube.execplan', 'cube.compiler'. Or None to set all.
        level (int): the level of the logger, can be one of
            logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR.
    """

    if name is None:
        logger_names = list(logging.root.manager.loggerDict.keys())
        logger_names = [name for name in logger_names if name.startswith('cube')]
        loggers = [logging.getLogger(name) for name in logger_names]
        for logger in loggers:
            logger.setLevel(level)
    elif name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(level)


_check_torch_version()
_init_logger()