#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Optional
import math
import os
import time
import warnings

import torch


GPU_METRIC_NAMES = (
    'gpu_util_avg',
    'gpu_util_min',
    'gpu_mem_util_avg',
    'gpu_sm_clock_avg_mhz',
    'gpu_sm_clock_min_mhz',
    'gpu_power_avg_w',
    'gpu_power_max_w',
    'gpu_temp_max_c',
    'gpu_throttle_reasons',
)

IB_METRIC_NAMES = (
    'ib_tx_mib_s',
    'ib_rx_mib_s',
    'ib_error_delta',
    'ib_xmit_wait_delta',
)


def nan_metrics(names) -> Dict[str, float]:
    return {name: math.nan for name in names}


def finite_statistics(values: torch.Tensor) -> Optional[Dict[str, float]]:
    """Return rank statistics while ignoring unavailable (NaN/Inf) samples."""
    valid_mask = torch.isfinite(values)
    if not torch.any(valid_mask).item():
        return None
    valid = values[valid_mask]
    valid_ranks = torch.nonzero(valid_mask, as_tuple=False).flatten()
    max_index = torch.argmax(valid)
    return {
        'p50': torch.quantile(valid, 0.50).item(),
        'p95': torch.quantile(valid, 0.95).item(),
        'max': valid[max_index].item(),
        'min': valid.min().item(),
        'max_rank': int(valid_ranks[max_index].item()),
        'count': int(valid.numel()),
    }


def finite_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation for finite pairs, or NaN when undefined."""
    valid = torch.isfinite(x) & torch.isfinite(y)
    if valid.sum().item() < 2:
        return math.nan
    x_valid = x[valid].double()
    y_valid = y[valid].double()
    x_centered = x_valid - x_valid.mean()
    y_centered = y_valid - y_valid.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if denominator.item() == 0:
        return math.nan
    return (torch.dot(x_centered, y_centered) / denominator).item()


class GpuTelemetrySampler:
    """Sample NVML in the background without launching ``nvidia-smi`` processes."""

    def __init__(self, local_rank: int, interval_s: float):
        self._interval_s = interval_s
        self._samples = []
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._pynvml = None
        self._handle = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = self._get_handle(pynvml, local_rank)
        except Exception:
            if self._pynvml is not None:
                try:
                    self._pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._pynvml = None
            self._handle = None
            return
        self._thread = Thread(
            target=self._run,
            name=f'nnscaler-gpu-telemetry-{local_rank}',
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _get_handle(pynvml, local_rank: int):
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        visible = [value.strip() for value in visible_devices.split(',') if value.strip()]
        if local_rank < len(visible):
            device = visible[local_rank]
            if device.startswith(('GPU-', 'MIG-')):
                return pynvml.nvmlDeviceGetHandleByUUID(device)
            if device.isdigit():
                return pynvml.nvmlDeviceGetHandleByIndex(int(device))
        return pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    def _sample(self):
        pynvml = self._pynvml
        handle = self._handle
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        throttle_reasons = 0
        try:
            throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        except Exception:
            pass
        return (
            float(utilization.gpu),
            float(utilization.memory),
            float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
            float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            int(throttle_reasons),
        )

    def _run(self):
        while not self._stop.is_set():
            try:
                sample = self._sample()
                with self._lock:
                    self._samples.append(sample)
            except Exception:
                # A transient NVML failure should not terminate training.
                pass
            self._stop.wait(self._interval_s)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def collect(self) -> Dict[str, float]:
        with self._lock:
            samples = self._samples
            self._samples = []
        if not samples:
            return nan_metrics(GPU_METRIC_NAMES)
        gpu_util, mem_util, sm_clock, power, temperature, throttle = zip(*samples)
        throttle_reasons = 0
        for value in throttle:
            throttle_reasons |= value
        return {
            'gpu_util_avg': sum(gpu_util) / len(gpu_util),
            'gpu_util_min': min(gpu_util),
            'gpu_mem_util_avg': sum(mem_util) / len(mem_util),
            'gpu_sm_clock_avg_mhz': sum(sm_clock) / len(sm_clock),
            'gpu_sm_clock_min_mhz': min(sm_clock),
            'gpu_power_avg_w': sum(power) / len(power),
            'gpu_power_max_w': max(power),
            'gpu_temp_max_c': max(temperature),
            'gpu_throttle_reasons': float(throttle_reasons),
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self._interval_s))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class InfiniBandTelemetrySampler:
    """Read node-level InfiniBand counters from sysfs on local rank zero."""

    _ERROR_COUNTERS = (
        'excessive_buffer_overrun_errors',
        'link_downed',
        'link_error_recovery',
        'local_link_integrity_errors',
        'port_rcv_errors',
        'port_rcv_remote_physical_errors',
        'port_xmit_discards',
        'symbol_error',
        'VL15_dropped',
    )

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._start: Optional[Dict[str, int]] = None
        self._start_at: Optional[float] = None

    @staticmethod
    def _read_counter(path: Path) -> int:
        try:
            return int(path.read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return 0

    def _snapshot(self) -> Optional[Dict[str, int]]:
        if not self._enabled:
            return None
        ports = tuple(Path('/sys/class/infiniband').glob('*/ports/*/counters'))
        if not ports:
            return None
        result = {'tx_data': 0, 'rx_data': 0, 'errors': 0, 'xmit_wait': 0}
        for counters in ports:
            result['tx_data'] += self._read_counter(counters / 'port_xmit_data')
            result['rx_data'] += self._read_counter(counters / 'port_rcv_data')
            result['xmit_wait'] += self._read_counter(counters / 'port_xmit_wait')
            result['errors'] += sum(
                self._read_counter(counters / name) for name in self._ERROR_COUNTERS
            )
        return result

    def begin(self) -> None:
        self._start = self._snapshot()
        self._start_at = time.perf_counter()

    def collect(self) -> Dict[str, float]:
        end = self._snapshot()
        end_at = time.perf_counter()
        if self._start is None or self._start_at is None or end is None:
            return nan_metrics(IB_METRIC_NAMES)
        elapsed = max(end_at - self._start_at, 1e-9)
        # InfiniBand port_{xmit,rcv}_data counters count four-byte words.
        tx_mib = max(0, end['tx_data'] - self._start['tx_data']) * 4 / (1024 ** 2)
        rx_mib = max(0, end['rx_data'] - self._start['rx_data']) * 4 / (1024 ** 2)
        return {
            'ib_tx_mib_s': tx_mib / elapsed,
            'ib_rx_mib_s': rx_mib / elapsed,
            'ib_error_delta': float(max(0, end['errors'] - self._start['errors'])),
            'ib_xmit_wait_delta': float(max(0, end['xmit_wait'] - self._start['xmit_wait'])),
        }
