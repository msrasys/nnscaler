import warnings
from cube import runtime
from cube import profiler

from cube.compiler import SemanticModel, compile
from cube.utils import load_model, load_default_schedule, load_eval_schedule


def _check_torch_version():
    import torch
    torch_version = str(torch.__version__).split('+')[0]
    torch_version = float('.'.join(torch_version.split('.')[:2]))
    if torch_version < 1.11:
        warnings.warn(f"Expected PyTorch version >= 1.11 but got {torch_version}")


def init():
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()


_check_torch_version()
