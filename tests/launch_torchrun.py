#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Callable
import uuid
import torch
import os
import socket

from torch.distributed.run import elastic_launch, LaunchConfig
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

def _find_free_port() -> int:
    """Find an available local TCP port for rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _is_retryable_rdzv_error(exc: BaseException) -> bool:
    retryable_markers = (
        "The server socket has failed to listen on any local network address.",
        "The connection to the C10d store has failed.",
        "The client socket has timed out",
        "Address already in use",
    )
    cur = exc
    messages = []
    while cur is not None:
        messages.append(str(cur))
        cur = cur.__cause__
    error_text = " ".join(messages)
    return any(marker in error_text for marker in retryable_markers)


def launch_torchrun(nproc_per_node, worker_fn, *args, **kwargs):
    explicit_master_port = os.environ.get("MASTER_PORT")

    # LaunchConfig may transiently fail if a selected port is no longer
    # available. Retry with a fresh local port when c10d rendezvous errors.
    for attempt in range(3):
        if attempt == 0 and explicit_master_port is not None:
            master_port = explicit_master_port
        else:
            master_port = str(_find_free_port())

        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=nproc_per_node,
            rdzv_backend="c10d",
            rdzv_endpoint=f"127.0.0.1:{master_port}",
            rdzv_configs={"is_host": True},
            run_id=str(uuid.uuid4()),
            monitor_interval=0.1,
            max_restarts=0,
        )

        try:
            return elastic_launch(launch_config, worker_fn)(*args, **kwargs)
        except ChildFailedError:
            raise
        except Exception as exc:
            if attempt == 2 or not _is_retryable_rdzv_error(exc):
                raise
            print(
                f"retrying launch_torchrun with new rendezvous port after error: {exc}"
            )

    raise RuntimeError("launch_torchrun failed after retries")


def torchrun(nproc_per_node: int, test_fn: Callable, *args, **kwargs):
    """Test utility for torchrun

    Example usage:

    ```python
    from functools import partial
    test_function_name = partial(torchrun, 2, function_to_test)
    ```

    Args:
        nproc_per_node (int): number of gpus
        test_fn (function): test function, which should return None
        *args: args for worker_fn
        **kwargs: kwargs for worker_fn

    Returns:
        None
    """

    if not torch.cuda.is_available() or torch.cuda.device_count() < nproc_per_node:
        print(f"skip test on {nproc_per_node} gpus due to lack of cuda devices")
        return
    launch_torchrun(nproc_per_node, test_fn, *args, **kwargs)


def clone_to_cpu(tensor: torch.Tensor):
    # when you use launch_torchrun
    # you can't directly return a cuda tensor
    #   Error message: Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
    # So you can use this function to clone a tensor to cpu
    cloned_tensor = tensor.cpu().clone().detach().requires_grad_(tensor.requires_grad)
    if tensor.is_leaf and tensor.grad is not None:
        cloned_tensor.grad = tensor.grad.cpu().clone()
    return cloned_tensor


def clone_to_cpu_recursively(data):
    if isinstance(data, torch.Tensor):
        return clone_to_cpu(data)
    elif isinstance(data, dict):
        return {k: clone_to_cpu_recursively(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clone_to_cpu_recursively(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(clone_to_cpu_recursively(v) for v in data)
    else:
        return data
