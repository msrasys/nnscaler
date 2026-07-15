#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""NNScaler checkpoint and mixed-precision adapters for Dion's Muon optimizer."""

import logging
from typing import TYPE_CHECKING

import torch

try:
    from dion.muon import Muon as _Muon
except ImportError as e:
    _MUON_IMPORT_ERROR = e

    class _Muon(torch.optim.Optimizer):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Dion is not installed. Install Dion to use Muon optimizers."
            ) from _MUON_IMPORT_ERROR

from nnscaler.runtime.utils import get_fparam_meta, get_dparam_meta
from nnscaler.utils import OptStateDict

from nnscaler.runtime.f16_optimizer import MixedPrecisionF16OptimizerMixin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nnscaler.runtime.adapter.reducer import FlattenParamInfo


class MuonMixin:
    _momentum_state_key = 'momentum'
    _legacy_momentum_state_key = 'momentum_buffer'

    def __init__(self, params, **kwargs):
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")

        self._flat_map: dict[int, tuple[FlattenParamInfo, list[int]]] = {}
        if isinstance(params[0], dict):
            if len(params) > 1:
                raise ValueError("MuonMixin only supports one param group")
            param_group = dict(params[0])
            param_group['params'] = self._unflatten_params(param_group['params'])
            params = [param_group]
        else:
            params = self._unflatten_params(params)
            if not params:
                params = [{'params': []}]

        super().__init__(params, **kwargs)

    def _unflatten_params(self, params):
        unflattened_params = []
        for idx, p in enumerate(params):
            if fpi := get_fparam_meta(p):
                if fpi.zero > 1:
                    raise ValueError("Muon does not support ZeRO3.")
                p_start = len(unflattened_params)
                unflattened_params.extend(fpi.get_embeded_params())
                self._flat_map[idx] = (fpi, list(range(p_start, len(unflattened_params))))
            else:
                unflattened_params.append(p)
                self._flat_map[idx] = (None, [len(unflattened_params) - 1])

        for p in unflattened_params:
            if dmeta := get_dparam_meta(p):
                # if dmeta.sub_shape != dmeta.shape:
                #     raise ValueError("Muon does not support TP.")
                if dmeta.sub_shape != p.shape:
                    raise ValueError("Muon does not support ZeRO3.")
            else:
                pass  # normal param from non-parallel module

        return unflattened_params

    def _get_momentum_state(self, param_state):
        if self._momentum_state_key in param_state:
            return self._momentum_state_key, param_state[self._momentum_state_key]
        if self._legacy_momentum_state_key in param_state:
            return self._legacy_momentum_state_key, param_state[self._legacy_momentum_state_key]
        return None, None

    def _normalize_state_for_load(self, param_state):
        param_state = dict(param_state)
        legacy_momentum = param_state.pop(self._legacy_momentum_state_key, None)
        if self._momentum_state_key not in param_state and legacy_momentum is not None:
            param_state[self._momentum_state_key] = legacy_momentum
        return param_state

    def state_dict(self):
        """
        Override state_dict to get the flattened states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        state: OptStateDict = super().state_dict()

        # called from hybrid optimizer before call `.step` (to get the param_groups of the wrapped optimizer)
        # In this case, state_dict['state'] is empty.
        if state['state']:
            new_param_states = {}
            for flat_idx, (fpi, param_indices) in self._flat_map.items():
                if fpi is None:
                    assert len(param_indices) == 1
                    if param_indices[0] in state['state']:
                        new_param_states[flat_idx] = self._normalize_state_for_load(state['state'][param_indices[0]])
                elif any(i in state['state'] for i in param_indices):
                    # need to flatten the states
                    embeded_states = []
                    state_key = None
                    for i in param_indices:
                        param_state = state['state'].get(i)
                        if param_state is None:
                            embeded_states.append(None)
                            continue
                        param_state = self._normalize_state_for_load(param_state)
                        cur_state_key, momentum_state = self._get_momentum_state(param_state)
                        if momentum_state is not None and state_key is None:
                            state_key = cur_state_key
                        embeded_states.append(momentum_state)
                    if any(momentum_state is not None for momentum_state in embeded_states):
                        new_param_states[flat_idx] = {state_key: fpi.flatten(embeded_states, device='cpu')}
                #else: no state for this flattened param

            state['state'] = new_param_states

        state['param_groups'][0]['params'] = list(range(len(self._flat_map)))
        return state

    def load_state_dict(self, state_dict: OptStateDict):
        """
        Override load_state_dict to unflatten the states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        new_param_states = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if flat_idx not in state_dict['state']:
                continue
            if fpi is None:
                assert len(param_indices) == 1
                new_param_states[param_indices[0]] = self._normalize_state_for_load(state_dict['state'][flat_idx])
            else:
                # need to unflatten the states
                _, flat_state = self._get_momentum_state(state_dict['state'][flat_idx])
                if flat_state is None:
                    continue
                embeded_states = fpi.unflatten(flat_state, device='cpu')
                for i, param_idx in enumerate(param_indices):
                    new_param_states[param_idx] = {self._momentum_state_key: embeded_states[i]}

        state_dict = dict(state_dict)
        state_dict['state'] = new_param_states
        param_count = sum(len(v[1]) for v in self._flat_map.values())
        state_dict['param_groups'] = list(state_dict['param_groups'])
        state_dict['param_groups'][0] = dict(state_dict['param_groups'][0])
        state_dict['param_groups'][0]['params'] = list(range(param_count))
        super().load_state_dict(state_dict)


class DionMuon(MuonMixin, _Muon):
    pass


class MixedPrecisionDionMuon(MixedPrecisionF16OptimizerMixin, DionMuon):
    def __init__(self, params, **kwargs):
        f16_params, unfolded_kwargs = self._unfold_params(params)
        kwargs = {**unfolded_kwargs, **kwargs}

        # Unflatten on the f16 side so that fparam_meta (lost by build_fp32_params)
        # is consumed before the optimizer ever sees raw 1D bucket chunks.
        self._flat_map = {}
        unflattened_f16 = MuonMixin._unflatten_params(self, f16_params)
        self.f16_params = unflattened_f16
        self.fp32_params = self.build_fp32_params(unflattened_f16)

        # Skip MuonMixin.__init__ (we already unflattened); init ancestor mixin state
        # manually so we don't lose ScaleDelayedOptimizerMixin / F16 mixin invariants.
        opt_params = self.fp32_params if self.fp32_params else [{'params': []}]
        _Muon.__init__(self, opt_params, **kwargs)
        self._multiply_factor = 1.0
        self._fp32_params_loaded = None

    _fp32_state_key = 'fp32_params'

    def state_dict(self):
        """
        Bucket-flattened state_dict that bundles the fp32 master copy alongside
        Muon momentum. The F16 mixin's default state_dict assumes a 1:1 map
        between fp32_params and state entries (and is Adam-specific), which
        breaks once MuonMixin re-flattens entries by `_flat_map`.
        """
        # Raw (unflattened) state from the underlying torch optimizer.
        state: OptStateDict = torch.optim.Optimizer.state_dict(self)

        if state['state']:
            # Optimizer.state_dict() returns shallow state entries. Copy them so
            # serialized fp32_params do not leak into the live optimizer state.
            state['state'] = {i: dict(param_state) for i, param_state in state['state'].items()}

            # Attach fp32 master copies to each unflattened entry first.
            for i, p32 in enumerate(self.fp32_params):
                if i in state['state']:
                    state['state'][i][self._fp32_state_key] = p32.detach()

            new_param_states = {}
            for flat_idx, (fpi, param_indices) in self._flat_map.items():
                if fpi is None:
                    assert len(param_indices) == 1
                    if param_indices[0] in state['state']:
                        new_param_states[flat_idx] = state['state'][param_indices[0]]
                elif any(i in state['state'] for i in param_indices):
                    momenta, fp32s = [], []
                    state_key = None
                    for i in param_indices:
                        ps = state['state'].get(i)
                        if ps is None:
                            momenta.append(None)
                            fp32s.append(None)
                            continue
                        cur_key, m = self._get_momentum_state(ps)
                        if state_key is None and cur_key is not None:
                            state_key = cur_key
                        momenta.append(m)
                        fp32s.append(ps.get(self._fp32_state_key))
                    entry = {}
                    if any(m is not None for m in momenta):
                        entry[state_key or self._momentum_state_key] = fpi.flatten(
                            momenta, dtype=torch.float32, device='cpu'
                        )
                    if any(f is not None for f in fp32s):
                        entry[self._fp32_state_key] = fpi.flatten(
                            fp32s, dtype=torch.float32, device='cpu'
                        )
                    new_param_states[flat_idx] = entry

            state['state'] = new_param_states

        state['param_groups'][0]['params'] = list(range(len(self._flat_map)))
        return state

    def load_state_dict(self, state_dict: OptStateDict):
        """
        Inverse of `state_dict`: unflatten per-bucket entries back into the
        per-parameter layout the underlying optimizer expects, while restoring
        the fp32 master copies onto `self.fp32_params`.
        """
        loaded_fp32 = False

        new_param_states = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if flat_idx not in state_dict['state']:
                continue
            entry = self._normalize_state_for_load(state_dict['state'][flat_idx])
            if fpi is None:
                assert len(param_indices) == 1
                param_idx = param_indices[0]
                p32_tensor = entry.pop(self._fp32_state_key, None)
                if p32_tensor is not None:
                    target = self.fp32_params[param_idx]
                    target.data.copy_(p32_tensor.to(device=target.device, dtype=torch.float32))
                    loaded_fp32 = True
                _, momentum = self._get_momentum_state(entry)
                if momentum is not None:
                    entry[self._momentum_state_key] = momentum.detach().to(dtype=torch.float32)
                new_param_states[param_idx] = entry
            else:
                _, flat_momentum = self._get_momentum_state(entry)
                momenta = fpi.unflatten(flat_momentum, device='cpu') if flat_momentum is not None else [None] * len(param_indices)
                flat_fp32 = entry.get(self._fp32_state_key)
                fp32s = fpi.unflatten(flat_fp32, device='cpu') if flat_fp32 is not None else [None] * len(param_indices)
                for k, param_idx in enumerate(param_indices):
                    sub = {}
                    if momenta[k] is not None:
                        sub[self._momentum_state_key] = momenta[k].detach().to(dtype=torch.float32)
                    if fp32s[k] is not None:
                        target = self.fp32_params[param_idx]
                        target.data.copy_(fp32s[k].to(device=target.device, dtype=torch.float32))
                        loaded_fp32 = True
                    new_param_states[param_idx] = sub

        state_dict = dict(state_dict)
        state_dict['state'] = new_param_states
        param_count = sum(len(v[1]) for v in self._flat_map.values())
        state_dict['param_groups'] = list(state_dict['param_groups'])
        state_dict['param_groups'][0] = dict(state_dict['param_groups'][0])
        state_dict['param_groups'][0]['params'] = list(range(param_count))

        if not loaded_fp32:
            logger.warning('fp32_params not found in state_dict, will sync from fp16 params to fp32 params')
            self._sync_fp16_params_to_fp32()

        torch.optim.Optimizer.load_state_dict(self, state_dict)
        self._fp32_params_loaded = True


try:
    from dion.newton_schulz_triton import newton_schulz_triton
except ImportError as e:
    _DION_TRITON_IMPORT_ERROR = e

    def newton_schulz_triton(*args, **kwargs):
        raise ImportError(
            "DION's newton_schulz_triton is not installed. "
            "Please install DION to use the CAPE optimizer."
        ) from _DION_TRITON_IMPORT_ERROR

def _normalize_frobenius(X: torch.Tensor, epsilon=1e-7, safety: float = 1.0) -> torch.Tensor:
    return X / (X.norm(dim=(-2, -1), keepdim=True) * safety + epsilon)

@torch.no_grad()
def mclip(X: torch.Tensor, P: torch.Tensor, a: float, epsilon=1e-7) -> torch.Tensor:
    transposed = X.size(-2) < X.size(-1)
    if transposed:
        X, P = X.mT, P.mT
    aP = a * P
    A = X.mT @ X
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).expand_as(A)
    S = newton_schulz_triton(A - a * a * I, epsilon=epsilon)
    X_clipped = 0.5 * (X + aP + (aP - X) @ S)
    if transposed:
        X_clipped = X_clipped.mT
    return X_clipped

@torch.no_grad()
def cape_polar(G: torch.Tensor, epsilon=1e-7, c=3.0, iterations: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    dtype = G.dtype
    X = G.float()
    r = min(G.shape[-2], G.shape[-1])
    sqrt_r = r ** 0.5
    X = _normalize_frobenius(X, epsilon=epsilon, safety=1.02)
    a = c / sqrt_r
    P = newton_schulz_triton(X, epsilon=epsilon)
    for _ in range(iterations):
        X = mclip(X, P, a, epsilon)
        X = _normalize_frobenius(X, epsilon=epsilon, safety=1.02)

    # Dion's rms_norm LR adjustment assumes the orthogonalized direction has
    # Frobenius norm sqrt(min(m, n)); keep CAPE on the same update scale.
    X = _normalize_frobenius(X, epsilon=epsilon)
    return (1.2 * sqrt_r * X).to(dtype=dtype)


__all__ = ['DionMuon', 'MixedPrecisionDionMuon', 'cape_polar']
