#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Protocol, Type, Union, TypeVar
from typing_extensions import get_args
from pathlib import Path
import logging
import inspect
import contextlib
import os

import torch
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import yaml
import torch

import nnscaler
from nnscaler.utils import enforce_zero_num_worker, fields, transform_recursively, load_type, copy_dynamic
from nnscaler.parallel import ComputeConfig, build_optimizer, ReuseType, BroadcastGenFilesStrategy, _PREDEFINED_POLICIES

from .arg_parser import (
    deserialize_dataclass,
    deserialize_value_type,
    merge_args, parse_args, fn_field,
    _TYPE_KEY, _VALUE_TYPE_KEY, _VALUE_KEY,
    resolve_args
)
from .loggers.logger_base import LoggerBase
from .train_hook import TrainHook
from .serialization import Checkpointer

if TYPE_CHECKING:
    from .trainer import Trainer


logger = logging.getLogger(__name__)


_TENSOR_TYPE = Literal['param', 'buffer', 'input']
_PRECISION_TYPE = Literal['fp32', 'fp16', 'bf16', 'none']
_PRECISION_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'none': None  # as it is. no conversion will happen.
}
_SELF_ARG_VALUE = 'self'
_LOSS_TYPE = TypeVar('_LOSS_TYPE')


def _get_tensor_dtype(precision: Dict[_TENSOR_TYPE, _PRECISION_TYPE], tensor_type: _TENSOR_TYPE) -> torch.dtype:
    return _PRECISION_MAP[precision[tensor_type]]


def _to_precision(module: torch.nn.Module, precision: Dict[_TENSOR_TYPE, _PRECISION_TYPE]):
    param_dtype = _get_tensor_dtype(precision, 'param')
    buffer_dtype = _get_tensor_dtype(precision, 'buffer')

    if param_dtype == buffer_dtype:
        if param_dtype is not None:
            module = module.to(param_dtype)
    else:
        # separate param and buffer dtype
        # TODO: a little hacky. A better way?
        # 3 kinds of tensors are converted in Module._apply:
        # model parameters, its grad, and buffer
        # param_dtype controls the first two, (but grad is `None` here)
        # and buffer_dtype controls the last one
        buf_ids = { id(buf) for buf in module.buffers(recurse=True) }
        if param_dtype is not None:
            module._apply(
                lambda t: t.to(param_dtype)
                    if t.is_floating_point() and id(t) not in buf_ids
                    else t)
        if buffer_dtype is not None:
            module._apply(
                lambda t: t.to(buffer_dtype)
                    if t.is_floating_point() and id(t) in buf_ids
                    else t)

    return module


def _resolve_precision(precision: Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE]]):
    supported_precision_type = get_args(_PRECISION_TYPE)
    supported_tensor_type = get_args(_TENSOR_TYPE)
    if not precision:
        precision = 'none'
    if isinstance(precision, str):
        precision = {k: precision for k in supported_tensor_type}
    for tensor_type in supported_tensor_type:
        if tensor_type not in precision:
            precision[tensor_type] = 'none'
        if precision[tensor_type] not in supported_precision_type:
            raise ValueError(f"Invalid precision {precision[tensor_type]} for {tensor_type}")
    if any(k not in supported_tensor_type for k in precision):
        raise ValueError(f"Invalid tensor type found in {precision.keys()}")

    return precision


def fix_input(input, input_dtype=None):
    if isinstance(input, dict):
        return {k: fix_input(v, input_dtype) for k, v in input.items()}
    elif isinstance(input, list):
        return [fix_input(v, input_dtype) for v in input]
    elif isinstance(input, tuple):
        return tuple(fix_input(v, input_dtype) for v in input)
    elif isinstance(input, torch.Tensor):
        if input.is_floating_point() and input_dtype is not None:
            return copy_dynamic(input, input.to(input_dtype).cuda())
        else:
            return copy_dynamic(input, input.cuda())
    return input


class PrecisionMixin:
    @property
    def param_dtype(self):
        return _get_tensor_dtype(self.precision, 'param')

    @property
    def buffer_dtype(self):
        return _get_tensor_dtype(self.precision, 'buffer')

    @property
    def input_dtype(self):
        return _get_tensor_dtype(self.precision, 'input')

    def fix_input(self, input):
        return fix_input(input, input_dtype=self.input_dtype)

    def to_precision(self, module):
        return _to_precision(module, self.precision)


class PolicyMixin:
    @property
    def resolved_pas_policy(self):
        if self.pas_policy in _PREDEFINED_POLICIES:
            return self.pas_policy
        return load_type(self.pas_policy)


@dataclass
class AggregatedOutputs:
    """
    Aggregated outputs from all micro-batches
    """
    # the aggregated loss as a sum
    loss_sum: float = None
    # number of mini batches
    num_batches: int = None
    # number of tokens (only used when grad_reduction is 'per-token-mean')
    num_tokens: Optional[int] = None
    # any other custom outputs
    aggregated_outputs: Any = None

    @classmethod
    def aggregate(cls,
        loss_outputs: list[_LOSS_TYPE],
        sync_group: torch.distributed.ProcessGroup,
        loss_fn: Callable[[_LOSS_TYPE], torch.Tensor],
        ntokens_fn: Callable[[_LOSS_TYPE], torch.Tensor] | None = None,
    ) -> 'AggregatedOutputs':
        losses, ntokens = [], []
        for output in loss_outputs:
            losses.append(loss_fn(output))
            if ntokens_fn is not None:
                ntokens.append(ntokens_fn(output))

        loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
        torch.distributed.all_reduce(loss_sum, group=sync_group)

        if ntokens_fn is not None:
            ntokens_sum = torch.sum(torch.tensor(ntokens, dtype=torch.int64, device=torch.cuda.current_device()))
            torch.distributed.all_reduce(ntokens_sum, group=sync_group)
        else:
            ntokens_sum = None

        num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
        torch.distributed.all_reduce(num_batches, group=sync_group)

        return AggregatedOutputs(
            loss_sum=loss_sum.item(),
            num_batches=num_batches.item(),
            num_tokens=ntokens_sum.item() if ntokens_sum is not None else None,
        )


@dataclass(frozen=True)
class OptionalComputeConfig:
    constant_folding: Optional[bool] = None
    trace_strategy: Optional[str] = None
    use_zero: Optional[bool] = None
    zero_ngroups: Optional[int] = None
    zero_use_reduce_scatter: Optional[bool] = None
    use_async_reducer: Optional[bool] = None
    reducer_bucket_cap_mb: Optional[float] = None

    pas_config: Optional[Dict[str, Any]] = None
    user_config: Optional[Dict[str, Any]] = None

    def resolve(self, compute_config: ComputeConfig) -> ComputeConfig:
        replace_values = {
            k: v for k, v in asdict(self).items()
            if v is not None
        }
        resolved_values = asdict(compute_config)
        resolved_values.update(replace_values)
        resolved_values[fields(ComputeConfig).use_end2end] = False
        return ComputeConfig(**resolved_values)


@dataclass
class ModuleParallelizeConfig:
    # The type to be parallelized
    # Please note if you specify this
    # pipeline parallelism will be disabled, and you must ensure ComputeConfig.use_end2end is False
    type: str = None
    # the module args to be used for creating the module
    # If run_mode is 'compile' and `args` is not None
    # we can parallelize submodules instead of creating whole model.
    # This is useful sometimes.
    args: Optional[Dict[str, Any]] = None
    # the full qualified name of the function to generate dummy inputs for forward
    # Its type should be `Callable[[TrainerArgs], dict[str, Any]]`
    # where the output dict is the kwargs for forward function of the module
    # The tensors in the sample will be moved to GPU and converted to input_dtype by trainer.
    forward_args_gen_fn: Optional[Callable[['TrainerArgs'], dict[str, Any]]] = fn_field(default=None)
    # the full qualified name of the function to post process the dummy inputs for forward
    # Note the tensors in the inputs have been moved to GPU and converted to input_dtype
    # But you can still further process the sample,
    # for example, mark some dims of tensors as dynamic
    # (you can do it in `forward_args_gen_fn` as well)
    forward_args_post_process_fn: Optional[Callable[['TrainerArgs', dict[str, Any]], dict[str, Any]]] = fn_field(default=None)
    # the model state dict file for tracing.
    # It is only used in tracing to serve as the initial state dict of the model.
    tracing_from_weights: str = None
    # the prefix in the state dict (loaded from trainer_args.tracing_from_weights) to be used for tracing
    tracing_from_weights_prefix: str = None

    # For the following config, If None, the config of the trainer_args will be used
    compute_config: Optional[OptionalComputeConfig] = None
    gen_savedir: Optional[str] = None
    gen_reuse: Optional[str] = None
    pas_policy: Optional[str] = None
    broadcast_strategy: Optional[str] = None
    # sometimes you want to dynamically set the instance name
    # for example, you can set it to the hash of related files
    # In that case, we can pass a dict with callable __type field.
    instance_name: Optional[str] = None
    precision: Optional[Dict[_TENSOR_TYPE, _PRECISION_TYPE]] = field(default=None, metadata={
        'skip_deserialization': True,
    })

    def __post_init__(self):
        if not self.type:
            raise ValueError("type is required")
        if not self.forward_args_gen_fn:
            raise ValueError("forward_args_gen_fn is required")

        if self.tracing_from_weights and self.tracing_from_weights_prefix:
            raise ValueError("tracing_from_weights and tracing_from_weights_prefix must not be used together")

        if self.precision is not None:
            self.precision = _resolve_precision(self.precision)

    @property
    def model_type(self):
        return load_type(self.type)

    def create_model(self, trainer_args: 'TrainerArgs', module_args: Optional[tuple[tuple, dict]]=None) -> torch.nn.Module:
        if self.args:
            args, kwargs = (), trainer_args.create_kwarg(self.args)
        elif module_args:
            args, kwargs = module_args
        else:
            raise ValueError("`module_args` or `args` must be provided")
        return self.model_type(*args, **kwargs)

    def create_dummy_forward_args(self, trainer_args: 'TrainerArgs') -> dict[str, Any]:
        return self.forward_args_gen_fn(trainer_args)


@dataclass
class ModelConfig:
    type: str = None
    args: dict[str, Any] = field(default_factory=dict)
    # if parallel_modules is not empty,
    # these modules will be parallelized instead of the whole model
    # and sub modules (in the list of `parallel_modules`) in the model
    # will be replaced with parallelized version
    parallel_modules: list[ModuleParallelizeConfig] = field(default_factory=list)

    def __post_init__(self):
        if len(set(m.type for m in self.parallel_modules)) != len(self.parallel_modules):
            raise ValueError(f"parallelized sub modules must be unique by type")


@dataclass
class OptimizerConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)
    clip_gnorm: float = 0.0

    param_clss_fn: Optional[Callable[[str], Any]] = fn_field(default=None)
    # loss reduction method
    # mean: average the loss over all micro-batches
    # sum: sum the loss of all micro-batches
    # per-token-mean: average the gradients over all tokens
    #    you must specify `aggregate_outputs_fn` and return the number of tokens
    # Please note in validation stage, this configuration is ignored
    # the loss is always averaged over all batches
    loss_reduction: str = 'mean'
    # different ways of calculating grad
    # sum: sum the gradients of all micro-batches
    # mean: average the gradients over all micro-batches
    # per-token-mean: average the gradients over all tokens
    #    you must specify `aggregate_outputs_fn` and return the number of tokens
    grad_reduction: str = 'mean'
    # the divisor applied to gradients before all-reduce. If not set, the default
    # divisor is `runtime_ngpus / plan_ngpus`. We divide the gradients to avoid overflow.
    # However, if the gradients are in high precision or the user has known the range of
    # the gradients, he/she can set a smaller divisor to improve the accuracy. Note that
    # the gradients will be recovered by multiplying the divisor after all-reduce and before
    # optimizer step.
    grad_reduce_divisor: Optional[float] = None
    # the function to aggregate the outputs from all micro-batches
    # inputs: (list of local outputs, torch group)
    # output: AggregateOutputs
    # you can use `torch.distributed.*` functions to do the work
    aggregate_outputs_fn: str = None

    def __post_init__(self):
        if self.grad_reduction not in ('sum', 'mean', 'per-token-mean'):
            raise ValueError(f"Invalid gradient_accumulation {self.grad_reduction}")
        if self.grad_reduction == 'per-token-mean' and not self.aggregate_outputs_fn:
            raise ValueError("aggregate_outputs_fn is required when grad_reduction is 'per-token-mean'")
        if self.loss_reduction == 'per-token-mean' and not self.aggregate_outputs_fn:
            raise ValueError("aggregate_outputs_fn is required when loss_reduction is 'per-token-mean'")
        if self.loss_reduction not in ('mean', 'sum', 'per-token-mean'):
            raise ValueError(f"Invalid loss_reduction {self.loss_reduction}")


@dataclass
class DatasetConfig:
    type: str = None
    train_args: Dict[str, Any] = field(default_factory=dict)
    val_args: Dict[str, Any] = field(default_factory=dict)
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataloaderConfig:
    type: str = 'torch.utils.data.DataLoader'
    train_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_args
    val_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_args
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSamplerConfig:
    type: str = 'torch.utils.data.DistributedSampler'
    train_args: Dict[str, Any] = field(default_factory=dict)
    val_args: Dict[str, Any] = field(default_factory=dict)
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRSchedulerConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)
    interval: str = 'epoch'

    def __post_init__(self):
        if self.interval not in ('epoch', 'step'):
            raise ValueError(f"Invalid interval {self.interval}")


@dataclass
class ResumeOptions:
    # sometimes you want to dynamically set checkpoint path
    # for example, you can set it to finetune model if no `last` checkpoint exists
    checkpoint: str = 'last'
    # the full qualified name of the function to
    # convert the checkpoint to nnscaler format
    # It should be `Callable[[Dict[str, Any]], Dict[str, Any]]`
    # Only applied when `checkpoint` is a file.
    # Please note you should handle the case
    # when checkpoint file comes from a factory method
    convert_fn: Optional[str] = None
    # whether to merge the checkpoint files
    # Only used when `checkpoint` is a directory.
    # `True` means will load the merged checkpoint (without saving)
    # `False` means will load the sharded checkpoint files
    # `None` means will load the sharded checkpoint files if the world size is not changed.
    #    and will load merged checkpoint if the world size is changed.
    with_merged: Optional[bool] = None
    # If the memory is limited, we can save memory by only loading merged state dict in GPU 0 of each node
    # and broadcast trimmed state dict to other ranks in the same node
    # although this will be slower
    # Only used when resuming from a merged checkpoint.
    save_memory: bool = True


@dataclass
class SerializerOptions:
    # the serialization runner to be used
    # It should be a name of registered SerializationRunners
    name: str = ''

    # the full qualified name of the function to create the serialization runner
    # Currently we do not support this way
    # to make sure all serialization runners are registered and can be used in other places
    # (like nnscaler.cli.Trainer.merge_checkpoint)
    # type: str = None

    # arguments for the serialization runner
    # Note You should be able to load for any arguments
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    save_dir: str = './checkpoints'
    no_save: bool = False

    # `"pt"`: PyTorch native format
    # `"safetensors"`: Safetensors format
    # You can also register new formats via `nnscaler.cli.serialization.register_format`
    # or specify a custom format here by providing a CheckpointFormat subclass
    format: str = 'pt'

    # the serialization runner to be used
    # It should be a name of registered SerializationRunners
    # If None, the default serializer will be used
    serializer: Optional[SerializerOptions] = field(default=None, metadata={
        'normalize': lambda x: {'name': x} if isinstance(x, str) else x
    })

    # `"sharded"`: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    # `"deduped"`: Each rank saves its deduped shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    # `"merged"`: everything has been merged into a single file.
    #   Used internally only when you merge the checkpoint files via `Trainer.merge_checkpoints`
    save_type: str = 'sharded'

    save_last: bool = True
    save_best: bool = True
    symlink_best_and_last: bool = True

    # save the checkpoint every n train steps
    # Please note we always run validation before saving the checkpoint
    every_n_train_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    keep_last_n_checkpoints: Optional[int] = None

    # resume training from a checkpoint folder/file
    # can be 'last'/'best'/a specific folder/file
    # we will not resume if resume_from is last or best but the corresponding checkpoint does not exist
    resume_from: Optional[ResumeOptions] = field(default=None, metadata={
        'normalize': lambda x: {'checkpoint': x} if isinstance(x, str) else x
    })

    def get_resume_checkpoint(self) -> Optional[Path]:
        if not self.resume_from or not self.resume_from.checkpoint:
            return None
        if self.resume_from.checkpoint in ['last', 'best']:
            d = Path(self.save_dir) / self.resume_from.checkpoint
            if not d.exists():
                return None
            return d
        return Path(self.resume_from.checkpoint)

    @property
    def resolved_convert_fn(self) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
        if not self.resume_from or not self.resume_from.convert_fn:
            return None
        return load_type(self.resume_from.convert_fn)

    def __post_init__(self):
        # backward compatibility
        if isinstance(self.resume_from, str):
            self.resume_from = ResumeOptions(checkpoint=self.resume_from)

        if isinstance(self.serializer, str):
            self.serializer = SerializerOptions(name=self.serializer)

        if self.resume_from and self.resume_from.checkpoint:
            if self.resume_from.checkpoint in ['last', 'best']:
                if not self.save_dir:
                    raise ValueError("save_dir is required when resume_from is 'last'/'best'")
                if not (Path(self.save_dir) / self.resume_from.checkpoint).exists():
                    logger.warning(f"`{self.resume_from.checkpoint}` checkpoint does not exist. Will train from scratch.")
            elif not Path(self.resume_from.checkpoint).exists():
                raise ValueError(f"resume_from {self.resume_from.checkpoint} does not exist")
        if self.no_save:
            return

        if self.save_type not in ('sharded', 'deduped', 'merged'):
            raise ValueError(f"Invalid save_type {self.save_type}")
        if not self.save_dir:
            raise ValueError("save_dir is required")

        if self.format not in Checkpointer.NAME_MAP:
            raise ValueError(f"Invalid format {self.format}")

        if self.serializer and self.serializer.name not in Checkpointer.REGISTERED_RUNNERS:
            raise ValueError(f"Invalid Serialization runner {self.serializer.name}")

        if self.every_n_epochs is not None and self.every_n_train_steps is not None:
            raise ValueError("Cannot specify both every_n_epochs and every_n_train_steps")
        if self.every_n_epochs is None and self.every_n_train_steps is None:
            self.every_n_epochs = 1  # default to 1 epoch

        if self.every_n_train_steps is not None and self.every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be positive")
        if self.every_n_epochs is not None and self.every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if self.keep_last_n_checkpoints is not None and self.keep_last_n_checkpoints < 1:
            raise ValueError("keep_last_n_checkpoints must be positive")


@dataclass
class LogConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.type:
            raise ValueError("type is required")
        if isinstance(self.type, str) and '.' not in self.type:
            # assume it is a built-in logger
            self.type = f'nnscaler.cli.loggers.{self.type}'


@dataclass
class ProfileScheduleConfig:
    # schedule configuration for the profiler.
    # The profiler will skip
    # the first ``skip_first`` steps, then wait for ``wait`` steps,
    # then do the warmup for the next ``warmup`` steps,
    # then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    # The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    # the cycles will continue until the profiling is finished.

    # The ``skip_first_wait`` parameter controls whether the first ``wait`` stage should be skipped.
    # This can be useful if a user wants to wait longer than ``skip_first`` between cycles, but not
    # for the first profile. For example, if ``skip_first`` is 10 and ``wait`` is 20, the first cycle will
    # wait 10 + 20 = 30 steps before warmup if ``skip_first_wait`` is zero, but will wait only 10
    # steps if ``skip_first_wait`` is non-zero. All subsequent cycles will then wait 20 steps between the
    # last active and warmup.

    wait: int = 0
    warmup: int = 0
    active: int = 0
    repeat: int = 0
    skip_first: int = 0
    skip_first_wait: int = 0


@dataclass
class ProfileDefaultTraceHandlerArgs:
    # the file to save chrome trace,
    # must contain `{step_num}`/`{rank}` to avoid overwriting traces of different steps/ranks
    export_chrome_trace: Optional[str] = None
    # the file to save stacks
    # must contain `{step_num}`/`{rank}` to avoid overwriting stacks of different steps/ranks
    # this option will be ignore if `with_stack` is False, as stacks will not be recorded in that case
    export_stacks: Optional[str] = None
    export_stacks_metric: str = 'self_cuda_time_total'

    def __post_init__(self):
        if self.export_chrome_trace:
            if '{step_num}' not in self.export_chrome_trace:
                raise ValueError("export_chrome_trace must contain '{step_num}' to avoid overwriting traces")
            if '{rank}' not in self.export_chrome_trace:
                raise ValueError("export_chrome_trace must contain '{rank}' to avoid overwriting traces.")
        if self.export_stacks:
            if self.export_stacks_metric not in (
                    "self_cpu_time_total",
                    "self_cuda_time_total",
                    "self_xpu_time_total",
            ):
                raise ValueError(f"export_stacks_metric must be one of 'self_cpu_time_total', 'self_cuda_time_total', or 'self_xpu_time_total', but got {self.export_stacks_metric}")
            if '{step_num}' not in self.export_stacks:
                raise ValueError("export_stacks must contain '{step_num}' to avoid overwriting traces")
            if '{rank}' not in self.export_stacks:
                raise ValueError("export_stacks must contain '{rank}' to avoid overwriting traces.")


@dataclass
class ProfileTensorBoardTraceHandlerArgs:
    dir_name: str
    worker_name: Optional[str] = None
    use_gzip: bool = False

    def __post_init__(self):
        if not self.dir_name:
            raise ValueError("dir_name is required for ProfileTensorBoardTraceHandlerArgs")

        if self.worker_name and '{rank}' not in self.worker_name:
            raise ValueError('worker_name must contain "{rank}" to make it unique across different ranks')


@dataclass
class ProfileTraceHandlerConfig:
    # currently we support two trace handlers:
    # "default": the default trace handler that can export chrome trace and stacks
    # "tensorboard": the trace handler that can export trace to tensorboard
    name: str = 'default'
    args: Union[ProfileDefaultTraceHandlerArgs, ProfileTensorBoardTraceHandlerArgs, None] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileTraceHandlerConfig':
        if not data:
            data = {}
        if 'name' not in data:
            data['name'] = 'default'

        name = data['name']
        if name == 'default':
            return cls(
                name=name,
                args=deserialize_dataclass(data.get('args', {}), ProfileDefaultTraceHandlerArgs)
            )
        elif name == 'tensorboard':
            return cls(
                name=name,
                args=deserialize_dataclass(data.get('args', {}), ProfileTensorBoardTraceHandlerArgs)
            )
        else:
            raise ValueError(f"Unsupported trace handler {name}")

@dataclass
class ProfileConfig:
    # list of activity groups (CPU, CUDA, XPU) to use in profiling
    activities: List[str] = field(default_factory=list)

    schedule: ProfileScheduleConfig = field(default_factory=ProfileScheduleConfig)

    # whether to record tensor shapes
    record_shapes: bool = True
    # whether to profile memory usage
    profile_memory: bool = True
    # whether to add stack traces
    with_stack: bool = False
    # whether to calculate FLOPs.
    with_flops: bool = False
    # record module hierarchy (including function names) corresponding to the callstack of the op
    with_modules: bool = False

    trace_handler: ProfileTraceHandlerConfig = field(
        default_factory=lambda: ProfileTraceHandlerConfig(name='default'),
        metadata={
            'deserialize': ProfileTraceHandlerConfig.from_dict
        }
    )

    def resolve_activities(self):
        activity_map = {
            'CPU': torch.profiler.ProfilerActivity.CPU,
            'CUDA': torch.profiler.ProfilerActivity.CUDA,
            'XPU': torch.profiler.ProfilerActivity.XPU,
        }
        activities = []
        for a in self.activities or []:
            a = a.upper()
            if a not in activity_map:
                raise ValueError(f"Unsupported activity {a} in ProfileConfig.activities")
            activities.append(activity_map[a])
        if not activities:
            return None
        return activities

    def __post_init__(self):
        if self.activities and any(a.upper() not in ('CPU', 'CUDA', 'XPU') for a in self.activities):
            raise ValueError(f"Invalid activity found in activities {self.activities}")

@dataclass
class DebugConfig:
     # before gradient clip norm, check the gradient sync for the same parameter is consistent cross devices,
     # if ZeRO is enabled, will check the gradient cross each ZeRO group,
     # if ZeRO is not enabled, will check the gradient cross each nnscaler scale unit.
     # this helps to find bugs related to gradient updates during training.
    check_gradient_sync_cross_devices: bool = True
    # profiling configuration using torch.profiler.profile
    profile: Optional[ProfileConfig] = None


@dataclass
class HookConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookMapConfig:
    after_setup: str = None
    on_finalize: str = None

    on_train_start: str = None
    on_train_end: str = None
    on_val_start: str = None
    on_val_end: str = None

    on_epoch_start: str = None
    on_epoch_end: str = None

    on_step_start: str = None
    on_step_end: str = None

    on_train_step_start: str = None
    on_train_step_end: str = None
    on_val_step_start: str = None
    on_val_step_end: str = None

    after_aggregate_train_step_outputs: str = None
    after_aggregate_val_step_outputs: str = None

    before_zero_grad: str = None
    after_zero_grad: str = None

    before_sync_grad: str = None
    after_sync_grad: str = None

    before_gnorm_clip: str = None
    after_gnorm_clip: str = None

    before_optimizer_step: str = None
    after_optimizer_step: str = None

    before_log_train_metrics: str = None
    before_log_val_metrics: str = None

    on_load_checkpoint: str = None
    after_load_checkpoint: str = None
    on_save_checkpoint: str = None
    on_expire_checkpoint: str = None


class ArgsTrainHook(TrainHook):
    def __init__(self, hook_config: HookMapConfig):
        self.config = hook_config
        for k, v in asdict(hook_config).items():
            if v:
                setattr(self, k, load_type(v))


def _deserialize_hook_config(hook) -> Union[HookConfig, HookMapConfig]:
    if isinstance(hook, dict):
        if 'type' in hook:
            return deserialize_dataclass(hook, HookConfig)
        else:
            # treat hook map as a dict. this is for backward compatibility
            # don't use `deserialize_dataclass` here
            # because hooks can be functions (not str)
            return HookMapConfig(**hook)
    raise ValueError(f"Invalid hook config {hook}.")


class _StepableContext(contextlib.AbstractContextManager):
    def step(self):
        ...


@dataclass
class TrainerArgs(PrecisionMixin, PolicyMixin):
    init_module: Optional[str] = None
    vars: Dict[str, Any] = field(default_factory=dict)
    compute_config: ComputeConfig = None

    gen_savedir: str = './.nnscaler'
    # the reuse strategy of the generated code
    # auto: automatically decide the reuse strategy (moo for compile, match for run)
    # Or one of match/override/moo/graph (see `nnscaler.ReuseType`)
    gen_reuse: str = 'auto'
    pas_policy: str = 'autodist'
    broadcast_strategy: str = 'all'
    # sometimes you want to dynamically set the instance name
    # for example, you can set it to the hash of related files
    # In that case, we can pass a dict with callable __type field.
    instance_name: Optional[str] = None
    # compile: compile the model but not training
    # run: compile and run the model
    run_mode: str = 'run'
    # the full qualified name of the function to generate dummy sample
    # Its type should be `Callable[[TrainerArgs], Any]`
    # The tensors in the sample will be moved to GPU and converted to input_dtype by trainer.
    dummy_sample_gen_fn: Optional[Callable[['TrainerArgs'], Any]] = fn_field(default=None)
    # the full qualified name of the function to post process the dummy sample
    # Note the tensors in the sample have been moved to GPU and converted to input_dtype
    # But you can still further process the sample,
    # for example, you can use this function to mark some dims of tensors as dynamic
    # when you don't use `dummy_sample_gen_fn` or don't handle dynamic dims in it,
    dummy_sample_post_process_fn: Optional[Callable[['TrainerArgs', Any], Any]] = fn_field(default=None)
    # the model state dict file for tracing.
    # It is only used in tracing to serve as the initial state dict of the model.
    tracing_from_weights: str = None

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset_sampler: Optional[DatasetSamplerConfig] = None
    lr_scheduler: Optional[LRSchedulerConfig] = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: List[LogConfig] = field(default_factory=list)
    # It can be `HookConfig` or `HookMapConfig`
    hook: Union[HookConfig, HookMapConfig, None] = field(default=None, metadata={
        'deserialize': _deserialize_hook_config
    })

    debug: DebugConfig = field(default_factory=DebugConfig)

    # None value will be resolved in __post_init__
    precision: Dict[_TENSOR_TYPE, _PRECISION_TYPE] = field(default=None, metadata={
        'skip_deserialization': True,
    })

    micro_batch_size: int = 1
    # You can set one of `global_batch_size` and `grad_accumulation_steps` option.
    # Please note if both are set, they must be consistent.
    # default is
    # global_batch_size = self.micro_batch_size*self.scaling_factor
    # grad_accumulation_steps = 1
    global_batch_size: Optional[int] = None
    grad_accumulation_steps: Optional[int] = None

    max_epochs: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None

    # validation frequency
    val_every_n_train_steps: Optional[int] = None
    val_every_n_epochs: Optional[int] = 1

    enable_progress_bar: bool = True
    # if progress_bar is disabled (enable_progress_bar is False),
    # the frequency to print the training progress
    # validation metrics will also be printed if it is not None.
    log_progress_every_n_train_steps: Optional[int] = 100

    seed: Optional[int] = None
    # environment initialization function
    # you can put your environment initialization code here
    init_env_fn: str = None

    def __post_init__(self):
        if not self.compute_config:
            raise ValueError("compute_config is required")
        if not self.compute_config.use_end2end:
            raise ValueError("use_end2end must be True")

        if not self.global_batch_size and not self.grad_accumulation_steps:
            self.global_batch_size = self.micro_batch_size*self.scaling_factor
            self.grad_accumulation_steps = 1
        elif not self.global_batch_size:
            self.global_batch_size = self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps
        elif not self.grad_accumulation_steps:
            self.grad_accumulation_steps = self.global_batch_size // (self.micro_batch_size*self.scaling_factor)

        if self.global_batch_size != self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps:
            raise ValueError(f"`global_batch_size` {self.global_batch_size} is not equal to `micro_batch_size*scaling_factor*grad_accumulation_steps` "
                             f"{self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps}")

        if self.run_mode not in ('compile', 'run'):
            raise ValueError(f"Invalid run_mode {self.run_mode}")

        if self.gen_reuse != 'auto':
            if self.gen_reuse not in [e.value for e in ReuseType]:
                raise ValueError(f"Invalid gen_reuse {self.gen_reuse}")
        else:
            self.gen_reuse = 'moo' if self.run_mode == 'compile' else 'match'

        if self.broadcast_strategy not in [e.value for e in BroadcastGenFilesStrategy]:
            raise ValueError(f"Invalid broadcast_strategy {self.broadcast_strategy}")

        self.precision = _resolve_precision(self.precision)

        if not self.max_epochs and not self.max_train_steps:
            raise ValueError("max_epochs or max_train_steps is required")

        if not self.model.type:
            raise ValueError("model type is required")

        for m in self.model.parallel_modules:
            if m.compute_config:
                # will raise ValueError if m.compute_config is invalid when combining with the global compute_config
                m.compute_config.resolve(self.compute_config)

            if load_type(m.type) == self.model_type:
                raise ValueError(f"parallelized sub module {m.type} cannot be the same as the model type in trainer args")

            if m.tracing_from_weights_prefix and not self.tracing_from_weights:
                raise ValueError("`tracing_from_weights` is required when `tracing_from_weights_prefix` is specified")

        if not self.optimizer.type:
            raise ValueError("optimizer type is required")
        if not self.dataset.type:
            raise ValueError("dataset type is required")
        if not self.dataloader.type:
            raise ValueError("dataloader type is required")
        if self.dataset_sampler and not self.dataset_sampler.type:
            raise ValueError("dataset_sampler type is required")
        if self.lr_scheduler and not self.lr_scheduler.type:
            raise ValueError("lr_scheduler type is required")

        if isinstance(self.hook, dict):
            # if it is a dict, we will deserialize it to HookMapConfig
            # This is for backward compatibility
            self.hook = _deserialize_hook_config(self.hook)

        if self.seed is None and self.init_env_fn is None:
            logger.warning(
                "Neither `seed` nor `init_env_fn` is not provided. "
                "The training may not be reproducible "
                "and the model weights on different devices can be different."
            )

        self._vars = self.create_kwarg(self.vars)
        # will be initialized lazily
        # because it is heavy, and may not be used in some cases
        # and it looks weird to initialize it eagerly in __post_init__
        self._dummy_input = None

    @classmethod
    def from_cli(cls, argv: List[str]) -> 'TrainerArgs':
        d = {}
        if argv[0] == '-f':
            with open(argv[1], 'r') as f:
                d = yaml.safe_load(f)
            argv = argv[2:]

        merge_args(d, argv)
        resolve_args(d)

        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainerArgs':
        if init_module := d.get('init_module', None):
            importlib.import_module(init_module)
        ta = deserialize_dataclass(d, TrainerArgs)
        return ta

    def to_dict(self):
        # replace all callable with their full qualified name
        # please note it is not reversible if local functions are used
        return transform_recursively(
            asdict(self),
            lambda class_or_func: f'{class_or_func.__module__}.{class_or_func.__qualname__}',
            callable,
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainerArgs':
        return cls.from_cli(['-f', path])

    def create_kwarg(self, value: Any) -> Any:
        if isinstance(value, dict):
            value = {k: self.create_kwarg(v) for k, v in value.items()}
            if _TYPE_KEY in value:
                value_type = load_type(value.pop(_TYPE_KEY))
                return value_type(**value)
            elif _VALUE_TYPE_KEY in value:
                return deserialize_value_type(value)
            else:
                return value
        elif isinstance(value, list):
            return [self.create_kwarg(i) for i in value]
        elif isinstance(value, tuple):
            return tuple(self.create_kwarg(i) for i in value)
        elif isinstance(value, str):
            # resolved reference
            # Note: resolved reference can only be used in various args
            # (train/optimizer/dataloader/etc args).
            # Use $$!(...) or $$!{...} to produce a literal $!(...) or $!{...} string.
            if value.startswith('$$!(') or value.startswith('$$!{'):
                return value[1:]  # strip one $ -> literal $!(...) / $!{...}
            if (value.startswith('$!(') and value.endswith(')')) \
                or (value.startswith('$!{') and value.endswith('}')):
                value = value[3:-1]
                if value == 'self':
                    return self
                else:
                    parts = value.split('.')
                    if parts[0] != 'vars':
                        raise ValueError(f"Invalid resolved reference {value}. It must be `self` or start with `vars`.")
                    # resolve self.vars.x.y.z
                    return self.get_resolved_var('.'.join(parts[1:]))
            return value
        else:
            return value

    @property
    def model_type(self):
        m = load_type(self.model.type)
        if not inspect.isclass(m) or not issubclass(m, torch.nn.Module):
            raise ValueError(f"Invalid model type {self.model.type}. It must be a subclass of torch.nn.Module")
        return m

    @property
    def resolved_aggregate_outputs_fn(self):
        if not self.optimizer.aggregate_outputs_fn:
            return None
        return load_type(self.optimizer.aggregate_outputs_fn)

    @property
    def scaling_factor(self):
        return self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus

    @property
    def update_freq(self):
        return self.global_batch_size // self.micro_batch_size // self.scaling_factor

    @property
    def enable_log_progress(self):
        return not self.enable_progress_bar and self.log_progress_every_n_train_steps

    @property
    def compile_mode(self) -> bool:
        return self.run_mode == 'compile'

    def init_env(self, trainer: 'Trainer'):
        if self.seed is not None:
            import random
            import numpy as np
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        if self.init_env_fn is None:
            return
        init_env_fn = load_type(self.init_env_fn)
        init_env_fn(trainer)

    def get_resolved_var(self, fqn: str, *, default: Any = None) -> Any:
        """
        Get a resolved variable from the vars dictionary.
        The fqn is a full qualified name of the variable, e.g. 'x.y.z'.
        """
        parts = fqn.split('.')
        var = self._vars
        for part in parts:
            if part not in var:
                return default
            var = var[part]
        return var

    @property
    def dummy_input(self):
        if self._dummy_input is None:
            self._dummy_input = self._load_dummy_input()
            self._dummy_input = fix_input(self._dummy_input, self.input_dtype)
            if self.dummy_sample_post_process_fn:
                self._dummy_input = self.dummy_sample_post_process_fn(self, self._dummy_input)
        return self._dummy_input

    def _load_dummy_input(self):
        if self.dummy_sample_gen_fn:
            return self.dummy_sample_gen_fn(self)

        with enforce_zero_num_worker(DataLoader):
            dataset = self.create_dataset('train')
            dataloader = self.create_dataloader('train', dataset)
            assert dataloader.num_workers == 0, "The dataloader must have `num_workers=0`."
            value = next(iter(dataloader))
            if close_fn := getattr(dataloader, 'close', None):
                close_fn()
            return value

    def create_model(self) -> torch.nn.Module:
        kwargs = self.create_kwarg(self.model.args)
        return self.model_type(**kwargs)

    def should_delay_bucket_building(self) -> bool:
        return self.optimizer.param_clss_fn is not None

    def create_parallel_optimizer(self, parallel_model: torch.nn.Module):
        kwargs = self.create_kwarg(self.optimizer.args)
        optimizer_class = load_type(self.optimizer.type)
        return build_optimizer(
            parallel_model, optimizer_class, self.compute_config,
            self.optimizer.param_clss_fn,
            **kwargs
        )

    def create_dataset(self, stage='train'):
        dataset_args = getattr(self.dataset, f'{stage}_args')
        # Sometimes a user uses a parameterless dataset class/factory function.
        # To support this case, we will create train dataset even without any arguments.
        # but val/test dataset must have arguments.
        if not dataset_args and stage != 'train':
            logger.info(f"{stage} dataset will not be created because empty arguments are provided.")
            return None
        kwargs = self.create_kwarg(dataset_args)
        dataset_class = load_type(self.dataset.type)
        dataset = dataset_class(**kwargs)
        return dataset

    def create_sampler(self, dataset, stage='train'):
        dataset_sampler = self.dataset_sampler or DatasetSamplerConfig()
        sampler_args = getattr(dataset_sampler, f'{stage}_args')
        sampler_args = sampler_args or dataset_sampler.train_args
        kwargs = self.create_kwarg(sampler_args)
        kwargs['dataset'] = dataset
        kwargs['num_replicas'] = self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus
        # if not distributed, we use the rank 0 sampler
        kwargs['rank'] = int(os.environ.get('RANK', 0)) // self.compute_config.plan_ngpus
        sampler_class = load_type(dataset_sampler.type)
        return sampler_class(**kwargs)

    def create_dataloader(self, stage='train', dataset=None):
        dataloader_args = getattr(self.dataloader, f'{stage}_args')
        dataloader_args = dataloader_args or self.dataloader.train_args
        kwargs = self.create_kwarg(dataloader_args)
        if 'batch_size' in kwargs:
            raise ValueError("`batch_size` should not be specified in dataloader_args. "
                             "You should use `micro_batch_size` instead.")
        kwargs['dataset'] = dataset or self.create_dataset(stage)
        if kwargs['dataset'] is None:
            return None
        if 'collate_fn' in kwargs:
            # special handling for collate_fn as a function
            # here we don't use self.collate_fn to avoid its implementation hacking
            kwargs['collate_fn'] = load_type(kwargs['collate_fn'])
        kwargs['batch_size'] = self.micro_batch_size

        dataloader_class = load_type(self.dataloader.type)
        if isinstance(dataset, torch.utils.data.IterableDataset):
            if self.dataset_sampler:
                raise ValueError("IterableDataset does not support sampler. "
                                 "Please remove dataset_sampler from TrainerArgs.")
        else:
            kwargs['sampler'] = self.create_sampler(kwargs['dataset'], stage)

        return dataloader_class(**kwargs)

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if not self.lr_scheduler:
            return None
        kwargs = self.create_kwarg(self.lr_scheduler.args)
        lr_scheduler_class = load_type(self.lr_scheduler.type)
        return lr_scheduler_class(optimizer, **kwargs)

    def create_loggers(self) -> List['LoggerBase']:
        loggers = []
        for log_config in self.log:
            kwargs = self.create_kwarg(log_config.args)
            logger_class = load_type(log_config.type)
            loggers.append(logger_class(**kwargs))
        return loggers

    def create_hook(self) -> TrainHook:
        if not self.hook:
            return TrainHook()  # empty hook

        hook_config = self.hook

        if isinstance(hook_config, HookConfig):
            kwargs = self.create_kwarg(hook_config.args)
            return load_type(hook_config.type)(kwargs)
        elif isinstance(hook_config, HookMapConfig):
            return ArgsTrainHook(hook_config)
        else:
            raise ValueError(f"Invalid hook_config {hook_config}")

    def create_checkpointer(self) -> Checkpointer:
        if self.checkpoint.serializer:
            return Checkpointer(
                self.checkpoint.format,
                self.checkpoint.serializer.name,
                self.checkpoint.serializer.args
            )
        return Checkpointer(self.checkpoint.format)

    def create_profiler(self) -> _StepableContext:
        """Create a torch.profiler.profile context manager based on DebugConfig.profile settings.

        Returns a profiler context manager if profiling is enabled, otherwise a contextlib.nullcontext.
        """
        profile_config = self.debug.profile
        if not profile_config:
            nc = contextlib.nullcontext()
            nc.step = lambda: None  # add a dummy step method to avoid checking in the training loop
            return nc

        trace_handler_config = profile_config.trace_handler
        schedule_config = profile_config.schedule
        rank = int(os.environ.get('RANK', 0))

        def _trace_handler(prof: torch.profiler.profile):
            if trace_handler_config.args.export_chrome_trace:
                trace_file = trace_handler_config.args.export_chrome_trace.format(
                    step_num=prof.step_num,
                    rank=rank
                )
                prof.export_chrome_trace(trace_file)
            if profile_config.with_stack and trace_handler_config.args.export_stacks:
                stacks_file = trace_handler_config.args.export_stacks.format(
                    step_num=prof.step_num,
                    rank=rank
                )
                prof.export_stacks(stacks_file, metric=trace_handler_config.args.export_stacks_metric)
            return _trace_handler

        def _get_trace_handler():
            if trace_handler_config.name == 'default':
                return _trace_handler
            elif trace_handler_config.name == 'tensorboard':
                args: ProfileTensorBoardTraceHandlerArgs = trace_handler_config.args
                return torch.profiler.tensorboard_trace_handler(
                    args.dir_name,
                    worker_name=args.worker_name.format(rank=rank),
                    use_gzip=args.use_gzip,
                )
            else:
                raise ValueError(f"Unsupported trace handler {trace_handler_config.name}")

        schedule = torch.profiler.schedule(
            skip_first=schedule_config.skip_first,
            skip_first_wait=schedule_config.skip_first_wait,
            wait=schedule_config.wait,
            warmup=schedule_config.warmup,
            active=schedule_config.active,
            repeat=schedule_config.repeat,
        )

        profiler = torch.profiler.profile(
            activities=profile_config.resolve_activities(),
            schedule=schedule,
            on_trace_ready=_get_trace_handler(),
            record_shapes=profile_config.record_shapes,
            profile_memory=profile_config.profile_memory,
            with_stack=profile_config.with_stack,
            with_flops=profile_config.with_flops,
            with_modules=profile_config.with_modules,
        )
        return profiler
