# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
import tempfile

import torch


def default_output_dir(model_name: str) -> Path:
    return Path(tempfile.gettempdir()) / "nnscaler-examples" / model_name


def require_generated_output(output_dir: Path) -> None:
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        raise RuntimeError(
            f"No generated nnScaler module found in {output_dir}. "
            "Run --mode compile before launching --mode run."
        )


def init_distributed(runtime_ngpus: int) -> int:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run mode must be launched with torchrun")
    if int(os.environ.get("WORLD_SIZE", "1")) != runtime_ngpus:
        raise RuntimeError("torchrun world size must match --runtime-ngpus")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return torch.distributed.get_rank()


def assert_finite_tensors(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tensors(item, f"{name}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tensors(item, f"{name}[{key!r}]")


def print_rank0(message: str) -> None:
    if torch.distributed.get_rank() == 0:
        print(message)


def finish_distributed() -> None:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
