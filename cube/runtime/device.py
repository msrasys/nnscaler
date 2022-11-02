"""
Communication group settings among devices
"""
from typing import List
import numpy as np
import torch
import os

from cube.flags import CompileFlag


class DeviceGroup:

    class __DeviceGroup:

        def __init__(self):
            if CompileFlag.dev_mode:
                print(f"DeviceGroup init using single device mode...")
                self.rank = 0
                self.world_size = 1
                self.local_world_size = 1
                self.local_rank = 0
                self.node_rank = 0
                self.groups = dict()
                torch.cuda.set_device(0)
            else:
                torch.distributed.init_process_group(
                    backend='nccl',
                )
                self.rank = torch.distributed.get_rank()
                self.world_size = torch.distributed.get_world_size()
                # assume each node has the same device number
                self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))
                self.local_rank = int(os.environ.get('LOCAL_RANK'))
                self.node_rank = int(os.environ.get('GROUP_RANK'))
                self.groups = dict()
                torch.cuda.set_device(self.local_rank)

    instance = None

    def __init__(self):
        if not DeviceGroup.instance:
            DeviceGroup.instance = DeviceGroup.__DeviceGroup()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    # def __setattr__(self, name):
    #     return setattr(self.instance, name)

    def __len__(self, name):
        return DeviceGroup.instance.world_size

    def get_group(self, ranks):
        """
        Create and return rank groups on-demand

        None will be returned if length of ranks are equal to world size
        """
        if len(ranks) == self.instance.world_size:
            return None
        rank_bits = DeviceGroup.bitmap(ranks)
        if rank_bits not in self.instance.groups:
            self.groups[rank_bits] = torch.distributed.new_group(list(ranks))
        return self.groups[rank_bits]

    def create_hybrid(self, group_num: List[int]) -> List[List[int]]:
        """
        Create hybrid (nested) groups given the each group number.

        The product of group_num should be same with total devices.
        """
        group_num = np.array(group_num)
        cnt = np.prod(group_num)
        if cnt != self.world_size:
            raise RuntimeError("product of group_num should be same with total device number")
        grid = np.arange(cnt).reshape(tuple(group_num))
        dims = list(range(len(group_num)))
        outputs = []
        for dim, num in enumerate(group_num):
            remain = np.prod(np.delete(group_num, dim))
            order = tuple(dims[:dim] + dims[dim+1:] + [dim])
            grid_dim = np.transpose(grid, order).reshape((remain,num))
            grid_dim = grid_dim.tolist()
            for ranks in grid_dim:
                # initialize group
                _ = self.get_group(ranks)
                if self.rank in ranks:
                    outputs.append(ranks)
        assert len(outputs) == len(group_num)
        return outputs


    @staticmethod
    def bitmap(ranks):
        """
        map the rank list to the bit map string
        """
        bits = '0' * DeviceGroup.instance.world_size
        for rank in ranks:
            if rank >= len(bits):
                raise ValueError("rank {} out of range ({})".format(rank, len(bits)))
            bits = bits[0:rank] + '1' + bits[rank+1:]
        return bits

    def __repr__(self):
        msg = 'node id: [{}] rank: [{}] local rank: [{}]\n'.format(self.node_id, self.rank, self.local_rank)
        msg += 'communication groups (ranks):\n'
        for bitmap, group in self.groups.items():
            ranks = [rank for rank, bit in enumerate(bitmap) if bit == '1']
            if self.instance.rank in ranks:
                msg += '\t group {}: my group rank: [{}]\n'.format(ranks, torch.distributed.get_rank(group))
        return msg
