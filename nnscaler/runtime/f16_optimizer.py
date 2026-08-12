#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/fp16_optimizer.py

import logging
import types
from typing import TYPE_CHECKING

import torch

from nnscaler.runtime.hybrid_optimizer import ScaleDelayedOptimizerMixin

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer

logger = logging.getLogger(__name__)


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
        fp32_params = self._get_fp32_params_for_state_dict(state_dict)
        for key, fp32_param in fp32_params.items():
            state_dict['state'].setdefault(key, {})['fp32_params'] = fp32_param

        return state_dict

    def _get_fp32_params_for_state_dict(self, state_dict):
        for key, value in state_dict['state'].items():
            assert 'exp_avg' in value, f'currently only verified for adam-like optimizer'
            assert self.fp32_params[key].shape == value['exp_avg'].shape, f'Shape mismatch: {value["exp_avg"].shape} vs {self.fp32_params[key].shape}'
        return {
            key: fp32_param.detach()
            for key, fp32_param in enumerate(self.fp32_params)
        }

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict.
        This will also load the fp32_params from the state
        """
        if not self._load_fp32_params(state_dict):
            logger.warning('fp32_params not found in state_dict, will sync from fp16 params to fp32 params')
            self._sync_fp16_params_to_fp32()

        if len(self.param_groups) != 1:
            raise RuntimeError('only support one param group')

        super().load_state_dict(state_dict)
        self._fp32_params_loaded = True

    def _load_fp32_params(self, state_dict):
        if 'state' not in state_dict or not state_dict['state']:
            return False

        state_count = len(state_dict['state'])
        fp32_param_state_count = sum(1 for state in state_dict['state'].values() if 'fp32_params' in state)
        if fp32_param_state_count == 0:
            return False
        if fp32_param_state_count != state_count:
            raise RuntimeError(
                f'fp32_params found in {fp32_param_state_count} out of {state_count} states, '
                f'but all states should have fp32_params'
            )

        logger.info('try to load fp32_params from state_dict in f16_optimizer')
        assert isinstance(self.fp32_params, list), f'fp32_params is not a list: {type(self.fp32_params)}'
        device = torch.cuda.current_device()
        for i, param in enumerate(self.fp32_params):
            ckpt_param = state_dict['state'][i]['fp32_params']
            assert param.shape == ckpt_param.shape, f'Shape mismatch: {param.shape} vs {ckpt_param.shape}'
            logger.info(f'param {i}, fp16 norm: {param.data.detach().norm().item()}, fp32 norm: {ckpt_param.data.detach().norm().item()}')
            param.data = state_dict['state'][i]['fp32_params'].data.to(device)
            # pop to avoid store a redundant copy in the wrapped optimizer
            state_dict['state'][i].pop('fp32_params')
        return True

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


import nnscaler.runtime.dion_optimizer as _dion_optimizer
_DionMuon = getattr(_dion_optimizer, 'Muon', None)


if _DionMuon is not None:
    class MixedPrecisionDionMuon(MixedPrecisionF16OptimizerMixin, _DionMuon):
        """Dion Muon with BF16/FP16 model parameters and FP32 optimizer state."""

        _fp32_state_key = 'fp32_params'

        def __init__(self, params, **kwargs):
            flattened_f16_params, unfolded_kwargs = self._unfold_params(params)
            kwargs = {**unfolded_kwargs, **kwargs}

            self.f16_params, flat_map = self.unflatten_params(flattened_f16_params)
            self.fp32_params = self.build_fp32_params(self.f16_params)
            super().__init__(
                self.fp32_params,
                flat_map=flat_map,
                **kwargs,
            )

        def _get_fp32_params_for_state_dict(self, state_dict):
            fp32_states = {
                index: {self._fp32_state_key: param.detach()}
                for index, param in enumerate(self.fp32_params)
            }
            flat_states = self._flatten_state(
                fp32_states,
                self._fp32_state_key,
                torch.float32,
            )
            return {
                index: state[self._fp32_state_key]
                for index, state in flat_states.items()
            }

        def _load_fp32_params(self, state_dict):
            flat_states = {
                index: {self._fp32_state_key: state[self._fp32_state_key]}
                for index, state in state_dict['state'].items()
                if self._fp32_state_key in state
            }
            if not flat_states:
                return not self.fp32_params

            fp32_states = self._unflatten_state(
                flat_states,
                self._fp32_state_key,
                torch.float32,
            )
            loaded = not self.fp32_params or super()._load_fp32_params(
                {'state': fp32_states})
            for state in state_dict['state'].values():
                state.pop(self._fp32_state_key, None)
            return loaded
