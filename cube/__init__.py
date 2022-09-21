from cube import runtime

from cube.compiler import SemanticModel, compile


def init():
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()



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
