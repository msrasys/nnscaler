# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pre-enqueued CUDA work with a host-releasable progress gate.

The native watchdog can release a wait without the GIL or a runnable CUDA
queue. Caller event dependencies establish data readiness independently.
"""
import ctypes
from collections import Counter
from pathlib import Path
from threading import RLock
import torch

_u64, _int, _ptr = ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p

_library = None
_library_lock = RLock()


def _native():
    global _library
    with _library_lock:
        if _library is not None:
            return _library
        from torch.utils.cpp_extension import CUDA_HOME, load
        if CUDA_HOME is None:
            raise RuntimeError('Staged prefetch requires CUDA toolkit headers and a C++ compiler')
        source = Path(__file__).with_name('_cuda_gate.cpp')
        import hashlib
        name = 'nnscaler_cuda_gate_' + hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        binary = load(name=name, sources=[str(source)], with_cuda=False,
                      is_python_module=False, verbose=False,
                      extra_include_paths=[str(Path(CUDA_HOME) / 'include')],
                      extra_cflags=['-O2', '-pthread'],
                      extra_ldflags=['-L' + str(Path(CUDA_HOME) / 'lib64' / 'stubs'),
                                     '-lcuda', '-pthread'])
        library = ctypes.CDLL(binary)
        for name, arguments, result in (
            ('gate_map', [_u64, ctypes.POINTER(_u64)], _int),
            ('gate_wait', [_u64, _u64], _int),
            ('gate_write', [_u64, _u64], _int),
            ('gate_host_open', [_u64], None),
            ('gate_host_is_open', [_u64], _int),
            ('gate_pool_internal', [_int, ctypes.POINTER(_int)], _int),
            ('gate_watchdog', [_u64, ctypes.c_uint], _ptr),
            ('gate_watchdog_state', [_ptr], _int),
            ('gate_watchdog_close', [_ptr], None),
        ):
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, result
        _library = library
        return library

counts = Counter()
_pools, _lock = {}, RLock()


def check(code):
    if code:
        raise RuntimeError(f'CUDA driver gate error {code}')


class PoolLease:
    def __init__(self, device):
        self.device, self.active = device, True
        with _lock, torch.cuda.device(device):
            record = _pools.get(device.index)
            if record is None:
                previous = _int()
                check(_native().gate_pool_internal(0, ctypes.byref(previous)))
                record = _pools[device.index] = [0, previous.value]
            record[0] += 1

    def release(self):
        if not self.active:
            return
        with _lock, torch.cuda.device(self.device):
            record = _pools[self.device.index]
            record[0] -= 1
            if record[0] == 0:
                previous = _int()
                check(_native().gate_pool_internal(record[1], ctypes.byref(previous)))
                del _pools[self.device.index]
            self.active = False


class Gate:
    def __init__(self, device, timeout_ms=200):
        allocator = torch.cuda.get_allocator_backend()
        if allocator not in ('native', 'cudaMallocAsync'):
            raise RuntimeError('Unsupported gate allocator')
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.native = _native()
        self.ticket = None
        self.queued = False
        self.pool = None
        pointer = _u64()
        with torch.cuda.device(self.device):
            self.flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self.host = self.flag.data_ptr()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Staged prefetch gates require eager CUDA execution')
            check(self.native.gate_map(self.host, ctypes.byref(pointer)))
            self.pointer = pointer.value
            self.pool = PoolLease(self.device) if allocator == 'cudaMallocAsync' else None
            self.ticket = self.native.gate_watchdog(self.host, timeout_ms)
            if not self.ticket:
                if self.pool is not None:
                    self.pool.release()
                raise RuntimeError('Could not start the native gate watchdog')
        counts['created'] += 1

    def hold(self, stream):
        from . import stream as lifetime
        event = torch.cuda.Event()
        event.record(stream)
        with lifetime._holds_lock:
            lifetime._pending_holds.append((event, (self,)))

    def wait(self, stream):
        try:
            check(self.native.gate_wait(stream.cuda_stream, self.pointer))
        finally:
            # The flag is needed until the GPU has actually passed the wait.
            self.hold(stream)

    def open(self, stream=None):
        if stream is None:
            self.native.gate_host_open(self.host)
            counts['host_release'] += 1
            self.queued = True
            return
        if self.queued or self.native.gate_host_is_open(self.host):
            return
        # A completed caller already satisfies the release ordering. Host
        # release avoids a redundant GPU memory fence and queue round trip.
        if stream.query():
            self.native.gate_host_open(self.host)
            counts['host_ready_release'] += 1
            self.queued = True
            return
        try:
            check(self.native.gate_write(stream.cuda_stream, self.pointer))
            self.hold(stream)
        except BaseException:
            self.native.gate_host_open(self.host)
            raise
        counts['device_release'] += 1
        self.queued = True

    def fallback(self):
        return self.ticket is not None and self.native.gate_watchdog_state(self.ticket) == 2

    def finish(self):
        if self.pool is not None:
            self.pool.release()

    def __del__(self):
        # Stop the native reader before Python can free the pinned flag, even
        # during interpreter teardown when diagnostic globals may be gone.
        fallback = False
        try:
            ticket = getattr(self, 'ticket', None)
            if ticket is not None:
                fallback = self.native.gate_watchdog_state(ticket) == 2
                self.native.gate_host_open(self.host)
                self.native.gate_watchdog_close(ticket)
                self.ticket = None
        finally:
            try:
                self.finish()
                if fallback:
                    counts['watchdog_fallback'] += 1
            except Exception:
                pass

