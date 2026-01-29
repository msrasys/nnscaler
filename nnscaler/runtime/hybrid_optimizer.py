#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import defaultdict
from dataclasses import dataclass, field
import types
from typing import Any, Callable, Iterable, Type, Union, TYPE_CHECKING, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.hooks import RemovableHandle

from nnscaler.cli.arg_parser import deserialize_dataclass
from nnscaler.cli.train_hook import TrainHookHost, TrainHook
from nnscaler.utils import fn_field, OptStateDict

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer


@dataclass
class HybridSubOptParamGroupConfig:
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSubOptConfig:
    type: Union[Type[Optimizer], Callable[..., Optimizer]] = fn_field(default=None)
    options: dict[str, Any] = field(default_factory=dict)
    param_groups: list[HybridSubOptParamGroupConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.type:
            raise ValueError("Optimizer type must be specified in HybridSubOptConfig")


@dataclass
class HybridOptConfig:
    optimizers: list[HybridSubOptConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.optimizers:
            raise ValueError("At least one optimizer must be specified in HybridOptConfig")


class HybridRemovableHandle:
    def __init__(self, removable_handles: list[RemovableHandle]):
        self.removable_handles = removable_handles

    def remove(self):
        for removable_handle in self.removable_handles:
            removable_handle.remove()

    def __enter__(self) -> "HybridRemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


class ScaleDelayedOptimizerMixin(TrainHook):
    """
    A mixin class to add scale-delayed optimization support to an optimizer.
    This mixin overrides the `scale_grads`, `clip_gnorm`, and `step` methods
    of the optimizer to delay the scaling of gradients until the `step` method is called.
    """
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    def after_setup(self, trainer: 'Trainer') -> None:
        if trainer.optimizer is self:
            # do nothing if we are in the hybrid optimizer,
            # who is responsible for overriding these methods.
            trainer.optimizer._clip_gnorm =  trainer.optimizer.clip_gnorm
            trainer.optimizer.clip_gnorm = self.overrided_clip_gnorm
            trainer.optimizer._scale_grads = trainer.optimizer.scale_grads
            trainer.optimizer.scale_grads = self.overrided_scale_grads

        # we need to override the step method to apply the scaling factor
        # hybrid optimizer will also call `step` of child optimizers,
        self._step = self.step
        self.step = self.override_step

    def overrided_scale_grads(self, scale: float):
        """
        Scale the gradients by a factor.
        Will override the original scale_grads method in ParallelOptimizer.
        """
        self._multiply_factor *= scale

    def overrided_clip_gnorm(self, max_norm: Optional[float] = None) -> float:
        """
        Will override the original clip_gnorm method in ParallelOptimizer.
        """
        # self._clip_gnorm() is ParallelOptimizer.clip_gnorm
        grad_norm = self._multiply_factor * self._clip_gnorm()
        if max_norm is not None and max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
            self._multiply_factor *= clip_coef
        return grad_norm

    def override_step(self, closure=None):
        """
        Performs a single optimization step.
        """
        # apply the accumulated multiply factor to grads
        if self._multiply_factor != 1.0:
            for pg_idx in range(len(self.param_groups)):
                for p in self.param_groups[pg_idx]['params']:
                    if p.grad is not None:
                        p.grad.mul_(self._multiply_factor)
            self._multiply_factor = 1.0
        # can't use super() here because we need to support applying this mixin to existing optimizers
        self._step(closure)

    @classmethod
    def apply_mixin(cls, obj: Any) -> Any:
        """Apply this mixin to an existing object."""
        obj._multiply_factor = 1.0
        # bind the new methods
        obj.after_setup = types.MethodType(cls.after_setup, obj)
        obj.overrided_scale_grads = types.MethodType(cls.overrided_scale_grads, obj)
        obj.overrided_clip_gnorm = types.MethodType(cls.overrided_clip_gnorm, obj)
        obj.override_step = types.MethodType(cls.override_step, obj)

        return obj


class HybridOptimizer(torch.optim.Optimizer, TrainHookHost, TrainHook):
    """
    A hybrid optimizer that combines multiple optimizers/multiple param groups
    into a single optimizer.

    Please note HybridOptimizer doesn't call super().__init__(),
    So it is actually a duck type for optimizer.
    """

    # Identifier for hybrid optimizer
    is_hybrid = True

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            param_clss: dict[torch.nn.Parameter, tuple[int, int]],
            config: Union[HybridOptConfig, dict[str, Any]]
    ):
        """
        Initialize the hybrid optimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): The parameters to optimize.
            param_clss (dict[torch.nn.Parameter, tuple[int, int]]): The parameter classes for each parameter.
            config (Union[HybridOptConfig, dict[str, Any]]): The configuration for the hybrid optimizer.
        """
        params = list(params)
        if isinstance(config, dict):
            config = deserialize_dataclass(config, HybridOptConfig)
        self.config = config

        self.optimizers = []
        classified_params = defaultdict(list)
        # map from (optimizer_idx, pg_idx, param_pg_idx) to param global param index
        param_loc = {}

        for idx, param in enumerate(params):
            param_cls = param_clss[param]
            assert param_cls[0] < len(self.config.optimizers)
            classified_params[param_cls].append(param)

            loc = *param_cls, len(classified_params[param_cls]) - 1
            param_loc[loc] = idx

        # sort with key i.e. (optimizer idx, param group idx)
        classified_params = dict(sorted(classified_params.items()))

        quick_param_groups = {param_cls: {"params": params} for param_cls, params in classified_params.items()}
        opt_param_groups = defaultdict(dict)
        for param_cls, group in quick_param_groups.items():
            opt_param_groups[param_cls[0]][param_cls[1]] = group

        for idx, opt_config in enumerate(config.optimizers):
            param_groups = opt_param_groups[idx]
            if len(param_groups) > 1:
                if len(param_groups) != len(opt_config.param_groups):
                    raise ValueError(f"Expected {len(opt_config.param_groups)} param groups, got {len(param_groups)}")
                # param group indices must be consecutive.
                if max(param_groups.keys()) != len(opt_config.param_groups) - 1:
                    raise ValueError(f"Param group indices must be consecutive. We have {len(opt_config.param_groups)} groups, got max group id {max(param_groups.keys())}")
                for param_group_idx, param_group in param_groups.items():
                    param_group.update(opt_config.param_groups[param_group_idx].options)
            else:
                if len(opt_config.param_groups) > 1:
                    raise ValueError(f"Expected at most 1 param group, got {len(opt_config.param_groups)}")
                if opt_config.param_groups:
                    param_groups[0].update(opt_config.param_groups[0].options)
            optimizer = opt_config.type(param_groups.values(), **opt_config.options)
            self.optimizers.append(optimizer)

        # map from param global index to (optimizer_idx, param_idx)
        self._param_map: dict[int, tuple[int, int]] = {}
        # map from (optimizer_idx, param_idx) to param global idx
        self._reverse_param_map: dict[tuple[int, int], int] = {}
        for opt_idx, optimizer in enumerate(self.optimizers):
            state_dict: OptStateDict = optimizer.state_dict()
            for pg_idx, pg in enumerate(state_dict['param_groups']):
                for param_idx_in_pg, param_idx in enumerate(pg['params']):
                    # param_idx_in_pg is the index in this param group
                    # param_idx is the index in this optimizer
                    global_idx = param_loc[(opt_idx, pg_idx, param_idx_in_pg)]
                    self._param_map[global_idx] = (opt_idx, param_idx)
                    self._reverse_param_map[(opt_idx, param_idx)] = global_idx

        # Don't call base init
        # So HybridOptimizer is a duck optimizer
        # super().__init__(params, {})

        # simulated param groups
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

        # to support scale-delayed optimizers like mixed-precision f16 optimizer
        self._has_scale_delayed = any(isinstance(opt, ScaleDelayedOptimizerMixin) for opt in self.optimizers)

    def after_setup(self, trainer: 'Trainer') -> None:
        if not self._has_scale_delayed:
            return

        assert trainer.optimizer is self, "HybridOptimizer should not be nested inside another optimizer"
        trainer.optimizer._clip_gnorm = trainer.optimizer.clip_gnorm
        trainer.optimizer._scale_grads = trainer.optimizer.scale_grads

        # if any one of the optimizers is scale-delayed,
        # we must apply the mixin to make sure all optimizers are scale-delayed
        # this is the only way to calculate gnorm correctly.
        for opt in self.optimizers:
            if not isinstance(opt, ScaleDelayedOptimizerMixin):
                ScaleDelayedOptimizerMixin.apply_mixin(opt)
            opt.after_setup(trainer)
            # disable after_setup for child optimizers
            # as we have already handled it here
            opt.after_setup = lambda *args, **kwargs: None

        def overrided_scale_grads(self, scale: float) -> None:
            for optimizer in self.optimizers:
                optimizer.overrided_scale_grads(scale)

        self.scale_grads = types.MethodType(overrided_scale_grads, self)

        def override_clip_gnorm(self, max_norm: Optional[float] = None) -> float:
            # self._clip_gnorm() is ParallelOptimizer.clip_gnorm
            # all optimizers have the same `multiply_factor`
            grad_norm = self.optimizers[0]._multiply_factor * self._clip_gnorm()
            if max_norm is not None and max_norm > 0.0:
                clip_coef = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                # will update all optimizers' multiply_factor
                self.scale_grads(clip_coef)
            return grad_norm

        self.clip_gnorm = types.MethodType(override_clip_gnorm, self)

    def _get_hook_objects(self):
        return self.optimizers

    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        assert closure is None, "Closure is not supported in HybridOptimizer"
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all optimizers.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " [\n"
        format_string += ",\n".join(f"{repr(opt)}" for opt in self.optimizers)
        format_string += "\n]"
        return format_string

    def register_step_pre_hook(self, hook) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_step_pre_hook(hook) for opt in self.optimizers])

    def register_step_post_hook(self, hook) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_step_post_hook(hook) for opt in self.optimizers])

    def register_state_dict_pre_hook(
        self, hook, prepend: bool = False
    ) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_state_dict_pre_hook(hook, prepend=prepend) for opt in self.optimizers])

    def register_state_dict_post_hook(
        self,
        hook,
        prepend: bool = False,
    ) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_state_dict_post_hook(hook, prepend=prepend) for opt in self.optimizers])

    def state_dict(self):
        state_dicts: list[OptStateDict] = [opt.state_dict() for opt in self.optimizers]
        merged_state_dict: OptStateDict = {'state': {}, 'param_groups': [{'children': {}}]}

        for opt_idx, sd in enumerate(state_dicts):
            for param_idx, s in sd['state'].items():
                merged_state_dict['state'][self._reverse_param_map[(opt_idx, param_idx)]] = s
            merged_state_dict['param_groups'][0]['children'][opt_idx] = sd['param_groups']

        merged_state_dict['param_groups'][0]['params'] = list(range(len(self._param_map)))
        merged_state_dict['param_groups'][0]['param_map'] = self._param_map
        merged_state_dict['param_groups'][0]['reverse_param_map'] = self._reverse_param_map
        merged_state_dict['state'] = dict(sorted(merged_state_dict['state'].items()))

        return merged_state_dict

    def register_load_state_dict_pre_hook(
        self,
        hook,
        prepend: bool = False,
    ) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_load_state_dict_pre_hook(hook, prepend=prepend) for opt in self.optimizers])

    def register_load_state_dict_post_hook(
        self, hook, prepend: bool = False
    ) -> HybridRemovableHandle:
        return HybridRemovableHandle([opt.register_load_state_dict_post_hook(hook, prepend=prepend) for opt in self.optimizers])

    def load_state_dict(self, state_dict) -> None:
        child_state_dicts = [{'state': {}, 'param_groups': []} for _ in self.optimizers]

        for idx, sd in enumerate(child_state_dicts):
            # copy param groups from state dict
            sd['param_groups'] = state_dict['param_groups'][0]['children'][idx]
            if len(sd['param_groups']) != len(self.optimizers[idx].param_groups):
                raise ValueError(f"Number of param groups mismatch. Expected {len(self.optimizers[idx].param_groups)} got {len(sd['param_groups'])}")
            # param groups can be changed (for example, the compute config is changed)
            # state_dict for HybridOptimizer is already well organized,
            # here we will carefully dispatch parameters to each optimizer.
            current_state_dict = self.optimizers[idx].state_dict()
            for pg, current_pg in zip(sd['param_groups'], current_state_dict['param_groups']):
                pg['params'] = current_pg['params'][:]  # make a copy

        for param_idx, param_state in state_dict['state'].items():
            opt_idx, param_state_idx = self._param_map[param_idx]
            child_state_dicts[opt_idx]['state'][param_state_idx] = param_state

        for child_state_dict, opt in zip(child_state_dicts, self.optimizers):
            opt.load_state_dict(child_state_dict)

        # after loading from state dict, the param_groups of optimizers are reassigned
        # (instead of updated inplace), so we need to gather them again (as we have done
        # in the constructor).
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # no-op to avoid creating new parameter groups
        # all parameter groups are managed by the individual optimizers
        pass


@dataclass
class HybridSubLRSchedulerConfig:
    type: Union[Type[LRScheduler], Callable[..., LRScheduler]] = fn_field(default=None)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridLRSchedulerConfig:
    schedulers: list[HybridSubLRSchedulerConfig] = field(default_factory=list)


class HybridLRScheduler(LRScheduler, TrainHookHost):
    """
    A hybrid learning rate scheduler that combines multiple schedulers.

    Please note HybridLRScheduler doesn't call super().__init__(),
    So it is actually a duck type for scheduler.
    """

    def __init__(
            self,
            optimizer: HybridOptimizer,
            config: Union[HybridLRSchedulerConfig, dict[str, Any]],
            last_epoch: int = -1,
    ):
        assert isinstance(optimizer, HybridOptimizer), "Optimizer must be an instance of HybridOptimizer"
        if isinstance(config, dict):
            config = deserialize_dataclass(config, HybridLRSchedulerConfig)

        if len(config.schedulers) == 1:
            self.schedulers = [config.schedulers[0].type(optimizer, **config.schedulers[0].options)]
        elif len(config.schedulers) == len(optimizer.optimizers):
            self.schedulers = [sub_config.type(opt, **sub_config.options) for sub_config, opt in zip(config.schedulers, optimizer.optimizers)]
        else:
            raise ValueError(f"Expected {len(optimizer.optimizers)} or 1 schedulers, got {len(config.schedulers)}")

    def _get_hook_objects(self):
        return self.schedulers

    def step(self, epoch=None):
        for scheduler in self.schedulers:
            scheduler.step(epoch)

    def state_dict(self):
        return {idx: scheduler.state_dict() for idx, scheduler in enumerate(self.schedulers)}

    def load_state_dict(self, state_dict):
        for idx, sd in state_dict.items():
            self.schedulers[idx].load_state_dict(sd)
