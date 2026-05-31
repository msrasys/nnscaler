#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/fp16_optimizer.py

import logging
import os
import types
from typing import TYPE_CHECKING

import torch

from nnscaler.runtime.hybrid_optimizer import ScaleDelayedOptimizerMixin
from nnscaler.runtime.utils import get_dparam_meta, get_fparam_meta

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, value, default)
        return default


class MixedPrecisionF16OptimizerMixin(ScaleDelayedOptimizerMixin):
    """
    A mixin class for mixed precision optimizer.
    Support both FP16 and BF16 parameters.

    1. It will create a copy of FP32 parameters and grads,
    and use the FP32 copy for optimization (via `build_fp32_params`).
    2. It will sync FP16 grads to FP32 grads before optimizer.step().
    3. It will sync FP32 params back to FP16 params after optimizer.step().
    4. It will zero FP16 grads and FP32 grads to zero in zero_grad().

    """
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)
        # This flag is used to indicate whether fp32_params are loaded from checkpoint.
        # If not, we will sync from fp16 params to fp32 params in after_load_checkpoint.
        # If the model is trained from scratch, this flag will be None.
        self._fp32_params_loaded = None

    def after_setup(self, trainer: 'Trainer') -> None:
        """
        Here we override the clip_gnorm and scale_grads methods in the optimizer.
        Reason:
        1. The original clip_gnorm and scale_grads methods apply to bf16 grads, which is not what we want.
           We need to apply them to fp32 grads.
        2. Combine the multiply_factors of clip_gnorm and scale_grads. So only one muliply is needed.
           This can mitigate the precision loss caused by multiple multiplications.
        Assumption:
        `clip_gnorm` is called immediately after `scale_grads` in training loop.
        """
        if trainer.optimizer is self:
            # don't override when using HybridOptimizer
            trainer.optimizer._clip_gnorm = trainer.optimizer.clip_gnorm
            trainer.optimizer.clip_gnorm = self.overrided_clip_gnorm
            trainer.optimizer._scale_grads = trainer.optimizer.scale_grads
            trainer.optimizer.scale_grads = self.overrided_scale_grads

        # step method is overrided below to apply the scaling factor

    @classmethod
    def build_fp32_params(cls, params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        # create FP32 copy of parameters and grads
        fp32_params = []
        for p in params:
            if p.data.dtype != torch.float32:
                p32 = torch.nn.Parameter(p.data.float())
            else:
                # make sure the storage is not shared with original parameter
                p32 = torch.nn.Parameter(p.data.clone())
            p32.grad = torch.zeros_like(p32.data)
            fp32_params.append(p32)
        return fp32_params

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_f16_grads_to_fp32()
        super().step(closure)
        self._sync_fp32_params_to_f16()
        # No need to call gather_params here when zero is enabled,
        # as the gathered params are not in the optimizer

    def zero_grad(self, set_to_none: bool = True):
        """
        Clears the gradients of all optimized parameters.
        Will ignore `set_to_none` and always set fp16 grads and fp32 grads to None.
        """
        for p in self.f16_params:
            p.grad = None
        for p32 in self.fp32_params:
            p32.grad = None

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = super().state_dict()

        # called from hybrid optimizer before call `.step` (to get the param_groups of the wrapped optimizer)
        # In this case, state_dict['state'] is empty.
        if not state_dict['state']:
            return state_dict

        # move fp32_params to the same level with 'exp_avg' and 'exp_avg_sq'
        # we do this to handle the merge of sharded checkpoint in nnscaler
        assert 'state' in state_dict, f'state not found in state_dict: {state_dict.keys()}'
        assert isinstance(state_dict['state'], dict), f'state is not a dict: {type(state_dict["state"])}'
        assert len(self.fp32_params) == len(state_dict['state']), \
                f'len(fp32_params) != len(state[state]): {len(self.fp32_params)} != {len(state_dict["state"])}'
        assert 'exp_avg' in state_dict['state'][0], f'currently only verified for adam-like optimizer'
        for key, value in state_dict['state'].items():
            assert self.fp32_params[key].shape == value['exp_avg'].shape, f'Shape mismatch: {value["exp_avg"].shape} vs {self.fp32_params[key].shape}'
            # .detach(): save tensor instead of Parameter.
            value['fp32_params'] = self.fp32_params[key].detach()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict.
        This will also load the fp32_params from the state
        """
        loaded_fp32_params = False
        if 'state' in state_dict and len(state_dict['state']) > 0 and 'fp32_params' in state_dict['state'][0]:
            logger.info('try to load fp32_params from state_dict in f16_optimizer')
            assert isinstance(self.fp32_params, list), f'fp32_params is not a list: {type(self.fp32_params)}'
            loaded_fp32_params = True
            device = torch.cuda.current_device()
            for i, param in enumerate(self.fp32_params):
                ckpt_param = state_dict['state'][i]['fp32_params']
                assert param.shape == ckpt_param.shape, f'Shape mismatch: {param.shape} vs {ckpt_param.shape}'
                logger.info(f'param {i}, fp16 norm: {param.data.detach().norm().item()}, fp32 norm: {ckpt_param.data.detach().norm().item()}')
                param.data = state_dict['state'][i]['fp32_params'].data.to(device)
                # pop to avoid store a redundant copy in the wrapped optimizer
                state_dict['state'][i].pop('fp32_params')
        else:
            logger.warning('fp32_params not found in state_dict, will sync from fp16 params to fp32 params')
            self._sync_fp16_params_to_fp32()

        if len(self.param_groups) != 1:
            raise RuntimeError('only support one param group')

        super().load_state_dict(state_dict)
        if loaded_fp32_params:
            self._log_fp32_model_param_mismatches('f16_optimizer.load_state_dict.before_sync')
        self._sync_fp32_params_to_f16()
        self._fp32_params_loaded = True

    @staticmethod
    def _max_abs_diff_to_model_dtype(model_param: torch.nn.Parameter, fp32_param: torch.nn.Parameter) -> float:
        model_data = model_param.data.detach().view(-1)
        fp32_data = fp32_param.data.detach().view(-1)
        if model_data.numel() == 0:
            return 0.0

        chunk_numel = max(1, _env_int('NNSCALER_RESUME_PARAM_CHECK_CHUNK_NUMEL', 8 * 1024 * 1024))
        max_diff = 0.0
        with torch.no_grad():
            for start in range(0, model_data.numel(), chunk_numel):
                end = min(start + chunk_numel, model_data.numel())
                model_chunk = model_data[start:end]
                fp32_chunk = fp32_data[start:end]
                if fp32_chunk.device != model_chunk.device:
                    fp32_chunk = fp32_chunk.to(model_chunk.device)
                if fp32_chunk.dtype != model_chunk.dtype:
                    fp32_chunk = fp32_chunk.to(model_chunk.dtype)
                diff = (model_chunk - fp32_chunk).float().abs().max().item()
                max_diff = max(max_diff, diff)
        return max_diff

    @staticmethod
    def _describe_param(index: int, param: torch.nn.Parameter) -> str:
        meta = get_dparam_meta(param)
        if meta is not None:
            return f"idx={index} orig={meta.orig_name} local_shape={tuple(param.shape)} dtype={param.dtype}"

        fmeta = get_fparam_meta(param)
        if fmeta is not None:
            names = []
            try:
                embedded_params = fmeta.get_embeded_params()
            except Exception:
                embedded_params = list(fmeta.params_info.keys())
            for embedded_param in embedded_params[:4]:
                embedded_meta = get_dparam_meta(embedded_param)
                if embedded_meta is not None:
                    names.append(embedded_meta.orig_name)
                else:
                    names.append(f"shape={tuple(embedded_param.shape)}")
            suffix = "" if len(embedded_params) <= 4 else f", ... +{len(embedded_params) - 4}"
            return (
                f"idx={index} flattened local_shape={tuple(param.shape)} dtype={param.dtype} "
                f"embedded=[{', '.join(names)}{suffix}]"
            )

        return f"idx={index} local_shape={tuple(param.shape)} dtype={param.dtype}"

    def _log_fp32_model_param_mismatches(self, context: str) -> None:
        threshold = _env_float('NNSCALER_RESUME_PARAM_CHECK_THRESHOLD', 0.0)
        max_logs = max(0, _env_int('NNSCALER_RESUME_PARAM_CHECK_MAX_LOGS', 20))
        mismatch_count = 0
        checked_count = 0
        max_diff = 0.0
        max_diff_index = -1

        for index, (model_param, fp32_param) in enumerate(zip(self.f16_params, self.fp32_params)):
            if not model_param.requires_grad:
                continue
            checked_count += 1
            diff = self._max_abs_diff_to_model_dtype(model_param, fp32_param)
            if diff > max_diff:
                max_diff = diff
                max_diff_index = index
            if diff <= threshold:
                continue

            mismatch_count += 1
            if mismatch_count <= max_logs:
                logger.warning(
                    "[resume param mismatch] context=%s %s cast_diff=%s "
                    "fp32_dtype=%s fp32_shape=%s",
                    context,
                    self._describe_param(index, model_param),
                    diff,
                    fp32_param.dtype,
                    tuple(fp32_param.shape),
                )

        if mismatch_count:
            logger.warning(
                "[resume param mismatch summary] context=%s checked=%s mismatched=%s "
                "threshold=%s max_diff=%s max_diff_index=%s logged=%s",
                context, checked_count, mismatch_count, threshold, max_diff,
                max_diff_index, min(mismatch_count, max_logs),
            )
        else:
            logger.info(
                "[resume param mismatch summary] context=%s checked=%s mismatched=0 "
                "threshold=%s max_diff=%s max_diff_index=%s",
                context, checked_count, threshold, max_diff, max_diff_index,
            )

    def _sync_f16_grads_to_fp32(self):
        # copy FP16 grads to FP32
        for p, p32 in zip(self.f16_params, self.fp32_params):
            if not p.requires_grad:
                continue
            if p.grad is not None:
                if p32.grad is None:
                    p32.grad = p.grad.data.float()
                else:
                    p32.grad.data.copy_(p.grad.data)
            else:
                p32.grad = torch.zeros_like(p.data, dtype=torch.float)
            if self._multiply_factor != 1.0:
                p32.grad.mul_(self._multiply_factor)
        self._multiply_factor = 1.0

    def _sync_fp32_params_to_f16(self):
        # copy FP32 params back into FP16 model
        for p, p32 in zip(self.f16_params, self.fp32_params):
            if not p.requires_grad:
                continue
            p.data.copy_(p32.data)

    def _sync_fp16_params_to_fp32(self):
        # copy FP16 params to FP32
        for p, p32 in zip(self.f16_params, self.fp32_params):
            if not p.requires_grad:
                continue
            p32.data.copy_(p.data)

    def on_load_checkpoint(self, trainer, checkpoint) -> None:
        self._fp32_params_loaded = False
        logger.info('Set _fp32_params_loaded to False in on_load_checkpoint hook')

    def after_load_checkpoint(self, trainer, checkpoint) -> None:
        if not self._fp32_params_loaded:
            logger.info('fp32_params not loaded, will sync from fp16 params to fp32 params')
            self._sync_fp16_params_to_fp32()
            self._fp32_params_loaded = True

    def _unfold_params(self, params) -> tuple[list[torch.nn.Parameter], dict]:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(params[0], dict):
            if len(params) > 1:
                raise ValueError("MixedPrecisionF16OptimizerMixin only supports one param group")
            unfolded_params = list(params[0]['params'])
            unfolded_kwargs = {k: v for k, v in params[0].items() if k != 'params'}
        else:
            if not all(isinstance(p, torch.nn.Parameter) for p in params):
                raise ValueError("optimizer params should be either a list of Parameters or a dict with 'params' key")
            unfolded_params = params
            unfolded_kwargs = {}

        return unfolded_params, unfolded_kwargs


class MixedPrecisionAdam(MixedPrecisionF16OptimizerMixin, torch.optim.Adam):
    def __init__(self, params, **kwargs):
        self.f16_params, unfolded_kwargs = self._unfold_params(params)
        self.fp32_params = self.build_fp32_params(self.f16_params)
        kwargs = {**unfolded_kwargs, **kwargs}
        super().__init__(self.fp32_params, **kwargs)


class MixedPrecisionAdamW(MixedPrecisionF16OptimizerMixin, torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        self.f16_params, unfolded_kwargs = self._unfold_params(params)
        self.fp32_params = self.build_fp32_params(self.f16_params)
        kwargs = {**unfolded_kwargs, **kwargs}
        super().__init__(self.fp32_params, **kwargs)
