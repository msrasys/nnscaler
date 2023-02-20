import warnings
from cube import runtime
from cube import profiler

from cube.compiler import SemanticModel, compile


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


# ================== Experimental Feature =======================

# import threading

# _message_context = None

# def handle_request():
#     manager = runtime.executor.MessageManager()
#     while True:
#         req = manager.pull()
#         if isinstance(req, int):
#             break
#         req.wait()

# def init_manager():
#     global _message_context
#     _ = runtime.executor.MessageManager()
#     _message_context = threading.Thread(target=handle_request)
#     _message_context.start()


# def finish_manager():
#     """
#     Clear message manager
#     """
#     global _message_context
#     manager = runtime.executor.MessageManager()
#     manager.push(-1)
#     _message_context.join()
