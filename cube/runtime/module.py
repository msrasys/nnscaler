from typing import List, Dict, Tuple, Optional
import torch
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer
import os


class CubeModule(torch.nn.Module):
    """
    The module is responsible for parameter synchronization
    before training
    """

    def __init__(self):
        super().__init__()
        self._reducers: List[Reducer] = list()
        self._fullmap : Dict[str, Tuple[int, Tuple[slice], int]] = dict()
        self._batch_size: Optional[int] = None

    def add_reducer(self, reducer: Reducer):
        if not isinstance(reducer, Reducer):
            raise RuntimeError(f"Expected a Reducer but got {type(reducer)}")
        self._reducers.append(reducer)

    def reduce_grads(self):
        """
        Mannually allreduce gradients on the weight
        """
        for reducer in self._reducers:
            reducer.allreduce()

    def add_full_map(self, attr: str, tid: int, slicers: Tuple[slice], val_chunks: int):
        """
        Add an attribute map.
        The mapping includes current attribute name (str) to logical tensor id,
        and the mapping of logical tensor id including spatial (slice) and val chunks
        
        @param attr str: attribute name of this moudle
        @param tid int: full tensor id
        @param slicers Tuple[slice]: indexing from full tensor
        @param val_chunks int: the number of value chunks.
        """
        assert hasattr(self, attr), f"{attr} is not in the module"
        self._fullmap[attr] = (tid, slicers, val_chunks)

    def get_full_map(self):
        return self._fullmap

    def set_batch_size(self, bs: Optional[int]):
        assert (bs is None) or (isinstance(bs, int) and bs > 0)
        self._batch_size = bs

    def get_batch_size(self) -> Optional[int]:
        return self._batch_size

    def load_attr_content(self, filename: str):
        with torch.no_grad():
            full = torch.load(filename)
            for attr in self._fullmap.keys():
                tensor: torch.Tensor = getattr(self, attr)
                tid, slicers, nchunks = self._fullmap[attr]
                content = full[tid][slicers] / nchunks
                tensor.copy_(content)
                # print(f'attr {attr}:\n{getattr(self, attr)}')

    def init_group(self, ranks: List[int]):
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected ranks to be List[int]")
        DeviceGroup().get_group(ranks)

    def get_checkpoint(self, optimizer: torch.optim.Optimizer = None):
        state_dict = super().state_dict()
        assert os.path.isfile('dist_param_map.pt'), 'Cannot open distributed parameter mapping file: dist_param_map.pt'
        dist_param_map = torch.load('dist_param_map.pt')
        param_area_map = self._fullmap
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        return state_dict, dist_param_map, param_area_map, optimizer_state_dict

    def save_checkpoint(self, optimizer: torch.optim.Optimizer = None, filename_prefix: str = None):
        filename_prefix = 'dist_checkpoint' if filename_prefix is None else filename_prefix
        filename = f"{filename_prefix}-{DeviceGroup().rank}.ckpt"
        state_dict, dist_param_map, param_area_map, optimizer_state_dict = self.get_checkpoint(optimizer)
        print(f'> Saving distributed checkpoint to {filename}')
        torch.save({
            'state_dict': state_dict,
            'dist_param_map': dist_param_map,
            'param_area_map': param_area_map,
            'optim_state_dict': optimizer_state_dict,
        }, filename)

    @staticmethod
    def merge_partial_states(state_dicts):
        """
        :param state_dicts: list of state_dict from different ranks
        state_dict(model_state_dict, optimizer_state_dict, dist_param_map, param_area_map)
        :return: merged state_dict(model_state_dict, optimizer_state_dict,)
        """
        assert len(state_dicts) > 0
        if len(state_dicts) == 1:
            return state_dicts[0][0], state_dicts[0][1]

        # find tensor full shape
        param_max_dimsize = {}
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            for param_area in param_area_map.items():
                local_name = param_area[0][0:param_area[0].rfind('_')]
                assert len(local_name) > 0
                raw_name = dist_param_map[local_name]
                slices = param_area[1][1]
                if param_area[1][2] != 1:
                    print(f'TODO: value-split on {raw_name}')
                if raw_name in param_max_dimsize:
                    param_max_dimsize[raw_name] = max(param_max_dimsize[raw_name], slices)
                else:
                    param_max_dimsize[raw_name] = slices

        # create full tensors
        param_full_tensors = {}
        sample_step = -1
        optim_full_tensors: Dict[int, Dict[any, any]] = {}  # param_id, (state_name, state_val)
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            if len(optimizer_state_dict['state'].items()) > 0:
                optimizer_state_names = list(optimizer_state_dict['state'][0].keys())
                print(f'optimizer_state_names = {optimizer_state_names}')
                if 'step' in optimizer_state_names:
                    sample_step = optimizer_state_dict['state'][0]['step']
                    optimizer_state_names.remove('step')
                print(f'optimizer_state_names (without step) = {optimizer_state_names}')
            else:
                optimizer_state_names = []

            other_optim_keys = [key for key in optimizer_state_dict.keys() if key != 'state']
            optimizer_other_state_dict = {}
            for key in other_optim_keys:
                optimizer_other_state_dict[key] = optimizer_state_dict[key]

            # for raw_name in param_max_dimsize.keys():
            model_state_dict_keys = list(model_state_dict.keys())
            for param_area in param_area_map.items():
                local_name_with_id = param_area[0]
                local_name = local_name_with_id[0:local_name_with_id.rfind('_')]
                raw_name = dist_param_map[local_name]

                tensor_size_slice = param_max_dimsize[raw_name]
                tensor_size = []
                for dim_slice in tensor_size_slice:
                    tensor_size.append(dim_slice.stop)
                param_full_tensors[raw_name] = torch.zeros(tuple(tensor_size))

                index = model_state_dict_keys.index(local_name_with_id)
                if index in optimizer_state_dict['state']:
                    for state_name in optimizer_state_names:  # 'step'
                        if index not in optim_full_tensors:
                            optim_full_tensors[index] = {}
                        optim_full_tensors[index][state_name] = torch.zeros(tuple(tensor_size))
                else:
                    print(f'INFO: merge_checkpoint skips {local_name_with_id}\'s optimizer state')
            # print(f'param_full_tensors = {param_full_tensors}')
            # print(f'optim_full_tensors = {optim_full_tensors}')
            break  # only create once

        # assign value
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            model_state_dict_keys = list(model_state_dict.keys())
            for param_area in param_area_map.items():
                local_name_with_id = param_area[0]
                local_name = local_name_with_id[0:local_name_with_id.rfind('_')]
                raw_name = dist_param_map[local_name]
                slices = param_area[1][1]
                partial_tensor = model_state_dict[local_name_with_id]
                param_full_tensors[raw_name][slices] = partial_tensor

                index = model_state_dict_keys.index(local_name_with_id)
                if index in optimizer_state_dict['state']:
                    states = optimizer_state_dict['state'][index]
                    for name in optimizer_state_names:
                        val = states[name]
                        optim_full_tensors[index][name][slices] = val
                        if sample_step > 0:
                            optim_full_tensors[index]['step'] = sample_step

        # print(f'param_full_tensors (assigned) = {param_full_tensors}')
        # print(f'optim_full_tensors (assigned) = {optim_full_tensors}')

        optimizer_other_state_dict.update({'state': optim_full_tensors})
        # dump to ckpt
        return param_full_tensors, optimizer_other_state_dict

    @staticmethod
    def merge_checkpoints(filename_prefix='dist_checkpoint'):
        ckpts = {}
        for rank in range(DeviceGroup().world_size):
            filename = f"{filename_prefix}-{rank}.ckpt"
            ckpts[rank] = torch.load(filename)
        print(f'checkpoints = {ckpts}')

        state_dicts = []
        for ckpt in ckpts.values():
            model_state_dict = ckpt['state_dict']
            dist_param_map = ckpt['dist_param_map']
            param_area_map = ckpt['param_area_map']
            optimizer_state_dict = ckpt['optim_state_dict']
            state_dicts.push(model_state_dict, optimizer_state_dict, dist_param_map, param_area_map, )

        merged_model_state_dict, merged_optimizer_state_dict = CubeModule.merge_partial_states(state_dicts)

        # dump to ckpt
        torch.save({'state_dict': merged_model_state_dict,
                    'optim_state_dict': merged_optimizer_state_dict
                    }, filename_prefix + '.full.ckpt')