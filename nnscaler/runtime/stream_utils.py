#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CUDA stream lifetime helpers."""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import itertools
import queue
import threading
from typing import Any, Optional

import torch


_CUDA_SUCCESS = 0
_HOST_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_cudart = None
_cuda_launch_host_func = None
_cuda_get_error_string = None

_pending_refs: dict[int, tuple[str, tuple[torch.Tensor, ...]]] = {}
_pending_lock = threading.Lock()
_done_tokens = queue.SimpleQueue()
_release_thread: Optional[threading.Thread] = None
_release_thread_lock = threading.Lock()
_token_counter = itertools.count(1)


def _load_cudart():
    global _cudart, _cuda_launch_host_func, _cuda_get_error_string
    if _cuda_launch_host_func is not None:
        return _cuda_launch_host_func

    lib_names = []
    found = ctypes.util.find_library('cudart')
    if found:
        lib_names.append(found)
    lib_names.extend(['libcudart.so', 'libcudart.so.13', 'libcudart.so.12'])

    last_error = None
    for lib_name in lib_names:
        try:
            _cudart = ctypes.CDLL(lib_name)
            break
        except OSError as exc:
            last_error = exc
    else:
        raise RuntimeError("Unable to load libcudart for cudaLaunchHostFunc") from last_error

    _cuda_launch_host_func = _cudart.cudaLaunchHostFunc
    _cuda_launch_host_func.argtypes = [ctypes.c_void_p, _HOST_FUNC, ctypes.c_void_p]
    _cuda_launch_host_func.restype = ctypes.c_int

    _cuda_get_error_string = getattr(_cudart, 'cudaGetErrorString', None)
    if _cuda_get_error_string is not None:
        _cuda_get_error_string.argtypes = [ctypes.c_int]
        _cuda_get_error_string.restype = ctypes.c_char_p
    return _cuda_launch_host_func


def _cuda_error_message(code: int) -> str:
    if _cuda_get_error_string is None:
        return f'CUDA error {code}'
    message = _cuda_get_error_string(code)
    if not message:
        return f'CUDA error {code}'
    return message.decode('utf-8', errors='replace')


def _flatten_cuda_tensors(values: Any) -> tuple[torch.Tensor, ...]:
    tensors = []
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.is_cuda and id(value) not in seen:
                seen.add(id(value))
                tensors.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    visit(values)
    return tuple(tensors)


def _release_worker():
    while True:
        token = _done_tokens.get()
        if token is None:
            return
        _release_token(token)


def _release_token(token: int):
    with _pending_lock:
        payload = _pending_refs.pop(token, None)
    if payload is None:
        return
    action, refs = payload
    if action == 'resize_storage':
        for tensor in refs:
            tensor.untyped_storage().resize_(0)


def _ensure_release_worker():
    global _release_thread
    if _release_thread is not None and _release_thread.is_alive():
        return
    with _release_thread_lock:
        if _release_thread is None or not _release_thread.is_alive():
            _release_thread = threading.Thread(
                target=_release_worker,
                name='nnscaler-cuda-stream-release',
                daemon=True,
            )
            _release_thread.start()


@_HOST_FUNC
def _stream_done_callback(user_data):
    token = int(user_data) if user_data else 0
    if token:
        _done_tokens.put(token)


def _defer_until_stream_done(action: str, tensors: Any, stream: Optional[torch.cuda.Stream]):
    refs = _flatten_cuda_tensors(tensors)
    if not refs:
        return
    if stream is None:
        stream = torch.cuda.current_stream()

    token = next(_token_counter)
    with _pending_lock:
        _pending_refs[token] = (action, refs)

    _ensure_release_worker()
    launch_host_func = _load_cudart()
    err = launch_host_func(
        ctypes.c_void_p(stream.cuda_stream),
        _stream_done_callback,
        ctypes.c_void_p(token),
    )
    if err != _CUDA_SUCCESS:
        with _pending_lock:
            _pending_refs.pop(token, None)
        raise RuntimeError(f'cudaLaunchHostFunc failed: {_cuda_error_message(err)}')


def keep_alive_until_stream_done(tensors: Any, stream: Optional[torch.cuda.Stream] = None):
    """Keep CUDA tensors alive until all currently enqueued stream work finishes.

    This is an alternative to ``Tensor.record_stream`` for scheduler-owned
    cross-stream hand-offs. It holds Python references to the tensors and
    enqueues a CUDA host function at the tail of ``stream``. Once the stream
    reaches that host function, a normal Python worker thread drops the
    references, allowing the caching allocator to reclaim the storage without
    adding the tensors to another stream's delayed-free list.
    """
    _defer_until_stream_done('keep_alive', tensors, stream)


def resize_storage_after_stream_done(tensors: Any, stream: Optional[torch.cuda.Stream] = None):
    """Resize tensor storages to zero after current stream work finishes."""
    _defer_until_stream_done('resize_storage', tensors, stream)


def flush_deferred_stream_refs():
    """Release references whose host callbacks have already fired."""
    while True:
        try:
            token = _done_tokens.get_nowait()
        except queue.Empty:
            return
        if token is None:
            _done_tokens.put(None)
            return
        _release_token(token)


def _shutdown_release_worker():
    _done_tokens.put(None)


atexit.register(_shutdown_release_worker)
