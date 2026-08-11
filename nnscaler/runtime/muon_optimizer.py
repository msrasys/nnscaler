from typing import TYPE_CHECKING

import torch

from nnscaler.runtime.utils import get_fparam_meta, get_dparam_meta
from nnscaler.utils import OptStateDict

if TYPE_CHECKING:
    from nnscaler.runtime.adapter.reducer import FlattenParamInfo


class MuonMixin:
    momentum_buffer_name = 'momentum_buffer'

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
                if dmeta.sub_shape[-2:] != dmeta.shape[-2:]:
                    raise ValueError(
                        "Muon does not support TP on last two dimensions.")
                if dmeta.sub_shape != p.shape:
                    raise ValueError("Muon does not support ZeRO3.")
            else:
                pass  # normal param from non-parallel module

        return unflattened_params

    @staticmethod
    def get_embeded_params(flattened_params):
        embeded_params = []
        for param in flattened_params:
            if fpi := get_fparam_meta(param):
                embeded_params.extend(fpi.get_embeded_params())
            else:
                embeded_params.append(param)
        return embeded_params

    def _flatten_state(self, state, key, dtype=None):
        new_state_dict = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if fpi is None:
                assert len(param_indices) == 1
                if param_indices[0] in state:
                    new_state_dict[flat_idx] = dict(state[param_indices[0]])
                continue

            embeded_states = [state.get(i, {}).get(key) for i in param_indices]
            new_state_dict[flat_idx] = {
                key: fpi.flatten(embeded_states, dtype=dtype, device='cpu')
            }

        return new_state_dict

    def _unflatten_state(self, state, key, dtype=None):
        new_state_dict = {}
        for flat_idx, (fpi, param_indices) in self._flat_map.items():
            if flat_idx not in state:
                continue
            flat_state = state[flat_idx]
            if fpi is None:
                assert len(param_indices) == 1
                param_state = dict(flat_state)
                if dtype is not None and key in param_state:
                    param_state[key] = param_state[key].to(dtype=dtype)
                new_state_dict[param_indices[0]] = param_state
                continue

            if key not in flat_state:
                continue
            tensor = flat_state[key]
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            embeded_states = fpi.unflatten(tensor, device='cpu')
            for param_idx, embeded_state in zip(param_indices, embeded_states):
                new_state_dict[param_idx] = {key: embeded_state}

        return new_state_dict

    def state_dict(self):
        """
        Override state_dict to get the flattened states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        state: OptStateDict = super().state_dict()
        state['state'] = self._flatten_state(
            state['state'],
            self.momentum_buffer_name,
        )
        state['param_groups'] = [
            dict(group) for group in state['param_groups']
        ]
        state['param_groups'][0]['params'] = list(range(len(self._flat_map)))
        return state

    def load_state_dict(self, state_dict: OptStateDict):
        """
        Override load_state_dict to unflatten the states
        This is necessary to be compatible with other state dict related functions
        such as merge_state_dict
        """
        state_dict = dict(state_dict)
        state_dict['state'] = self._unflatten_state(
            state_dict['state'],
            self.momentum_buffer_name,
        )
        state_dict['param_groups'] = [
            dict(group) for group in state_dict['param_groups']
        ]
        param_count = sum(
            len(indices) for _, indices in self._flat_map.values())
        state_dict['param_groups'][0]['params'] = list(range(param_count))
        super().load_state_dict(state_dict)


if torch.__version__ >= (2, 9, 0):
    from torch.optim import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        pass
