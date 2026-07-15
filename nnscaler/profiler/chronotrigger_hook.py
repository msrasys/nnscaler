#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""ChronoTrigger lifecycle integration for the nnScaler Trainer."""

from __future__ import annotations

import logging
import os
from math import prod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import chronotrigger.trace as ct

from nnscaler.cli.train_hook import TrainHook

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer


logger = logging.getLogger(__name__)

_TRACE_AXES = ("pp", "ep", "tp", "cp")
_DEFAULT_AXIS_ORDER = _TRACE_AXES
_LEGACY_TRACE_GATE_ENV = "NNSCALER_NVTX_TRACE"


class ChronoTriggerTrainHook(TrainHook):
    """Own ChronoTrigger initialization, step context, and delayed capture."""

    def __init__(self) -> None:
        self.capture_window: Optional[Tuple[int, int]] = None
        self.capture_started = False
        self.capture_stopped = False

    def after_setup(self, trainer: "Trainer") -> None:
        tracing_enabled = _tracing_enabled()
        rank_layout = _rank_layout(trainer) if tracing_enabled else None
        ct.init(
            profile="nnscaler",
            schema="nnscaler.v1",
            rank_layout=rank_layout,
            enabled=tracing_enabled,
        )
        self.capture_window = _capture_window()

    def on_train_step_start(self, trainer: "Trainer", batches: list) -> None:
        current_step = trainer.train_status.finished_train_steps
        ct.set_step(current_step)
        if (
            self.capture_started
            or self.capture_window is None
            or current_step != self.capture_window[0]
        ):
            return

        _distributed_barrier()
        torch.cuda.synchronize()
        _cuda_profiler_call("cudaProfilerStart")
        self.capture_started = True
        ct.emit_metadata()
        if trainer.rank == 0:
            logger.info(
                "Started Nsight Systems CUDA-profiler capture at train step %s; stop step %s",
                *self.capture_window,
            )

    def on_train_step_end(self, trainer: "Trainer", outputs: list) -> None:
        if (
            not self.capture_started
            or self.capture_stopped
            or self.capture_window is None
        ):
            return

        completed_step = trainer.train_status.finished_train_steps + 1
        if completed_step < self.capture_window[1]:
            return

        torch.cuda.synchronize()
        _distributed_barrier()
        _cuda_profiler_call("cudaProfilerStop")
        self.capture_stopped = True
        if trainer.rank == 0:
            logger.info(
                "Stopped Nsight Systems CUDA-profiler capture after train step %s",
                completed_step,
            )


def _rank_layout(trainer: "Trainer") -> Dict[str, int]:
    rank = int(trainer.rank)
    world = int(trainer.world_size)
    compute_config = trainer.train_args.compute_config
    plan_size = int(compute_config.plan_ngpus)
    if plan_size <= 0 or world % plan_size != 0:
        raise ValueError(
            f"Invalid ChronoTrigger layout: world={world}, plan_ngpus={plan_size}"
        )

    pas_config = compute_config.pas_config
    default_pp_size = int(
        pas_config.get("pp_size", pas_config.get("pipeline_size", 1))
    )
    axis_sizes = {
        "pp": _positive_env_int(
            "NNSCALER_TRACE_PP_SIZE",
            _env_int("PIPELINE_VPP_PP_SIZE", default_pp_size),
        ),
        "ep": _positive_env_int(
            "NNSCALER_TRACE_EP_SIZE",
            _env_int("PIPELINE_VPP_EP_SIZE", 1),
        ),
        "tp": _positive_env_int("NNSCALER_TRACE_TP_SIZE", 1),
        "cp": _positive_env_int("NNSCALER_TRACE_CP_SIZE", 1),
    }

    pp_size = axis_sizes["pp"]
    if plan_size % pp_size != 0:
        raise ValueError(
            f"Invalid ChronoTrigger layout: plan_ngpus={plan_size}, pp_size={pp_size}"
        )

    rank_in_plan = rank % plan_size
    named_plan_size = prod(axis_sizes.values())
    has_inner_hint = any(axis_sizes[axis] > 1 for axis in ("ep", "tp", "cp"))
    coordinates = {axis: 0 for axis in _TRACE_AXES}
    if has_inner_hint:
        if named_plan_size != plan_size:
            raise ValueError(
                "Invalid ChronoTrigger layout: explicit PP/EP/TP/CP sizes must "
                f"multiply to plan_ngpus ({named_plan_size} != {plan_size})"
            )
        axis_order = _axis_order()
        remaining = rank_in_plan
        for index, axis in enumerate(axis_order):
            stride = prod(axis_sizes[name] for name in axis_order[index + 1 :])
            coordinates[axis] = remaining // stride
            remaining %= stride
    else:
        coordinates["pp"] = rank_in_plan // (plan_size // pp_size)

    return {
        "rank": rank,
        "world": world,
        "dp": rank // plan_size,
        **coordinates,
    }


def _axis_order() -> Tuple[str, ...]:
    value = os.environ.get("NNSCALER_TRACE_AXIS_ORDER")
    if not value:
        return _DEFAULT_AXIS_ORDER
    axes = tuple(axis.strip().lower() for axis in value.split(","))
    if len(axes) != len(_TRACE_AXES) or set(axes) != set(_TRACE_AXES):
        raise ValueError(
            "NNSCALER_TRACE_AXIS_ORDER must contain pp,ep,tp,cp exactly once"
        )
    return axes


def _capture_window() -> Optional[Tuple[int, int]]:
    start_step = _env_int("NNSCALER_NSYS_CAPTURE_START_STEP", -1)
    if start_step < 0:
        return None

    end_step = _env_int("NNSCALER_NSYS_CAPTURE_END_STEP", -1)
    if end_step <= start_step:
        capture_steps = _env_int("NNSCALER_NSYS_CAPTURE_STEPS", 0)
        if capture_steps <= 0:
            return None
        end_step = start_step + capture_steps
    return start_step, end_step


def _tracing_enabled() -> bool:
    if "CT_TRACE" in os.environ:
        return _env_enabled("CT_TRACE")
    return _env_enabled(_LEGACY_TRACE_GATE_ENV)


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in ("", "0", "false", "no", "off")


def _positive_env_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _cuda_profiler_call(name: str) -> None:
    result = getattr(torch.cuda.cudart(), name)()
    status = result[0] if isinstance(result, tuple) else result
    if status not in (None, 0):
        raise RuntimeError(f"{name} failed with status {status}")