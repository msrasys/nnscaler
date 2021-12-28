from cube import logics
from cube import runtime

from cube.compiler import SemanticModel, compile


def init():
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()
