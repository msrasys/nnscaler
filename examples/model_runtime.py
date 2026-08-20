# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
import tempfile

import torch


def default_output_dir(model_name: str) -> Path:
    return Path(tempfile.gettempdir()) / "nnscaler-examples" / model_name


def parse_pipeline_stages(value: str):
    if value == "auto":
        return value
    stages = int(value)
    if stages < 1:
        raise ValueError("pipeline stages must be positive or 'auto'")
    return stages


def build_compute_config(
    *,
    plan_ngpus: int,
    runtime_ngpus: int,
    inference_only: bool,
    use_end2end: bool,
    use_zero: bool,
    zero_use_reduce_scatter: bool,
    use_async_reducer: bool,
    reducer_replicated_params: bool,
    microbatches: int,
    pipeline_stages,
    pipeline_pivot: str,
    max_partition_degree: int | None,
    partition_constraints_path: Path | None = None,
):
    import nnscaler

    pas_config = {
        "mem_constraint": 40,
        "update_freq": microbatches,
    }
    if max_partition_degree is not None:
        pas_config["max_partition_degree"] = max_partition_degree
    if partition_constraints_path is not None:
        pas_config["partition_constraints_path"] = str(partition_constraints_path)
    if pipeline_stages != 1:
        if not pipeline_pivot:
            raise ValueError("a pipeline pivot is required when pipeline is enabled")
        pas_config.update(
            pipeline_pivots=pipeline_pivot,
            pipeline_nstages=pipeline_stages,
            max_pipeline_bubble_ratio=0.99,
            max_pipeline_unbalance_ratio=0.01,
        )

    return nnscaler.ComputeConfig(
        plan_ngpus=plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        inference_only=inference_only,
        use_zero=1 if use_zero else 0,
        zero_use_reduce_scatter=zero_use_reduce_scatter,
        use_end2end=use_end2end,
        use_async_reducer=use_async_reducer,
        reducer_replicated_params=reducer_replicated_params,
        pas_config=pas_config,
    )


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


def create_replica_group(plan_ngpus: int, runtime_ngpus: int):
    if runtime_ngpus == plan_ngpus:
        return None
    local_position = torch.distributed.get_rank() % plan_ngpus
    selected_group = None
    for position in range(plan_ngpus):
        ranks = list(range(position, runtime_ngpus, plan_ngpus))
        group = torch.distributed.new_group(ranks)
        if position == local_position:
            selected_group = group
    return selected_group


def assert_tensors_synced(tensors, group, name: str) -> None:
    if group is None:
        return
    for index, tensor in enumerate(tensors):
        local = tensor.detach()
        minimum = local.clone()
        maximum = local.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        if not torch.equal(minimum, maximum):
            difference = (maximum.float() - minimum.float()).abs().max().item()
            raise RuntimeError(
                f"{name}[{index}] is not synchronized across replicas "
                f"(max difference: {difference})"
            )


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
