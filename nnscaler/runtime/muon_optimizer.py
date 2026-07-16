from typing import TYPE_CHECKING

import torch

from nnscaler.runtime.utils import get_fparam_meta, get_dparam_meta
from nnscaler.utils import OptStateDict

if TYPE_CHECKING:
    from nnscaler.runtime.adapter.reducer import FlattenParamInfo


class MuonMixin:
    momentum_buffer_name = 'momentum_buffer'
    momentum_buffer_aliases = ()
    additional_state_keys = ()
    state_key_dtypes = {}

    def __init__(self, params, _nnscaler_flat_map=None, **kwargs):
        params = list(params)
        if not params and _nnscaler_flat_map is None:
            raise ValueError("optimizer got an empty parameter list")

        if _nnscaler_flat_map is not None:
            # Mixed-precision wrappers unflatten the model parameters before
            # creating their FP32 copies. Reuse that mapping instead of trying
            # to rediscover it from the metadata-free FP32 parameters.
            self._flat_map = _nnscaler_flat_map
            if not params:
                params = [{'params': []}]
        else:
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

        self.momentum_buffer_name = getattr(self, 'momentum_buffer_name', 'momentum_buffer')

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
                if dmeta.sub_shape[-2:] != dmeta.shape[-2:]:
                    raise ValueError("Muon does not support TP on last two dimensions.")
                if dmeta.sub_shape != p.shape:
                    raise ValueError("Muon does not support ZeRO3.")
            else:
                pass  # normal param from non-parallel module

        return unflattened_params

    def _state_keys(self):
        return (self.momentum_buffer_name, *self.additional_state_keys)

    def _normalize_param_state(self, param_state):
        param_state = dict(param_state)
        missing = object()
        for alias in self.momentum_buffer_aliases:
            value = param_state.pop(alias, missing)
            if self.momentum_buffer_name not in param_state and value is not missing:
                param_state[self.momentum_buffer_name] = value
        return param_state

    def _flatten_state_dict(self, state: OptStateDict) -> OptStateDict:
        """Map per-parameter optimizer state back to NNScaler flat buckets."""
        state = dict(state)
        new_param_states = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if fpi is None:
                assert len(param_indices) == 1
                if param_indices[0] in state['state']:
                    new_param_states[flat_idx] = self._normalize_param_state(
                        state['state'][param_indices[0]]
                    )
                continue

            param_states = [
                self._normalize_param_state(state['state'][i])
                if i in state['state'] else None
                for i in param_indices
            ]
            if not any(param_state is not None for param_state in param_states):
                continue

            flat_state = {}
            for state_key in self._state_keys():
                tensors = [
                    param_state.get(state_key) if param_state is not None else None
                    for param_state in param_states
                ]
                if any(tensor is not None for tensor in tensors):
                    flat_state[state_key] = fpi.flatten(
                        tensors,
                        dtype=self.state_key_dtypes.get(state_key),
                        device='cpu',
                    )
            if flat_state:
                new_param_states[flat_idx] = flat_state

        state['state'] = new_param_states
        state['param_groups'] = [dict(group) for group in state['param_groups']]
        state['param_groups'][0]['params'] = list(range(len(self._flat_map)))
        return state

    def _unflatten_state_dict(self, state_dict: OptStateDict) -> OptStateDict:
        """Map NNScaler flat-bucket state to the wrapped optimizer layout."""
        new_param_states = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if flat_idx not in state_dict['state']:
                continue
            flat_state = self._normalize_param_state(state_dict['state'][flat_idx])
            if fpi is None:
                assert len(param_indices) == 1
                new_param_states[param_indices[0]] = flat_state
                continue

            unflattened = {}
            for state_key in self._state_keys():
                if state_key in flat_state:
                    unflattened[state_key] = fpi.unflatten(
                        flat_state[state_key], device='cpu'
                    )
            for offset, param_idx in enumerate(param_indices):
                param_state = {
                    state_key: tensors[offset]
                    for state_key, tensors in unflattened.items()
                }
                if param_state:
                    new_param_states[param_idx] = param_state

        state_dict = dict(state_dict)
        state_dict['state'] = new_param_states
        state_dict['param_groups'] = [
            dict(group) for group in state_dict['param_groups']
        ]
        param_count = sum(len(indices) for _, indices in self._flat_map.values())
        state_dict['param_groups'][0]['params'] = list(range(param_count))
        return state_dict

    def state_dict(self):
        """
        Override state_dict to get the flattened states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        state: OptStateDict = super().state_dict()
        return self._flatten_state_dict(state)

    def load_state_dict(self, state_dict: OptStateDict):
        """
        Override load_state_dict to unflatten the states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        super().load_state_dict(self._unflatten_state_dict(state_dict))


if torch.__version__ >= (2, 9, 0):
    from torch.optim import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        pass
