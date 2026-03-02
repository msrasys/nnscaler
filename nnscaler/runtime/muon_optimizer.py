from typing import TYPE_CHECKING

import torch

from nnscaler.runtime.utils import get_fparam_meta, get_dparam_meta
from nnscaler.utils import OptStateDict

if TYPE_CHECKING:
    from nnscaler.runtime.adapter.reducer import FlattenParamInfo


class MuonMixin:
    def __init__(self, params, **kwargs):
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")

        self._flat_map: dict[int, tuple[FlattenParamInfo, list[int]]] = {}
        if isinstance(params[0], dict):
            if len(params) > 1:
                raise ValueError("MuonMixin only supports one param group")
            params[0]['params'] = self._unflatten_params(params[0]['params'])
        else:
            params = self._unflatten_params(params)

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
                if dmeta.sub_shape != dmeta.shape:
                    raise ValueError("Muon does not support TP.")
                if dmeta.sub_shape != p.shape:
                    raise ValueError("Muon does not support ZeRO3.")
            else:
                pass  # normal param from non-parallel module

        return unflattened_params

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
                        new_param_states[flat_idx] = state['state'][param_indices[0]]
                elif any(i in state['state'] for i in param_indices):
                    # need to flatten the states
                    embeded_states = [
                        state['state'][i]['momentum_buffer'] if i in state['state'] else None
                        for i in param_indices
                    ]
                    new_param_states[flat_idx] = {'momentum_buffer': fpi.flatten(embeded_states, device='cpu')}
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
                new_param_states[param_indices[0]] = state_dict['state'][flat_idx]
            else:
                # need to unflatten the states
                flat_state = state_dict['state'][flat_idx]['momentum_buffer']
                embeded_states = fpi.unflatten(flat_state, device='cpu')
                for i, param_idx in enumerate(param_indices):
                    new_param_states[param_idx] = {'momentum_buffer': embeded_states[i]}
        state_dict['state'] = new_param_states
        param_count = sum(len(v[1]) for v in self._flat_map.values())
        state_dict['param_groups'][0]['params'] = list(range(param_count))
        super().load_state_dict(state_dict)


if torch.__version__ >= (2, 9, 0):
    from torch.optim import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        pass
