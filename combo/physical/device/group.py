"""
Communication group settings among devices
"""

import torch
import os


class DeviceGroup:

    class __DeviceGroup:

        def __init__(self):
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                # world_size=device_num,
                # init_method='tcp://' + '{master_ip}:{port}'.format(master_ip=master_ip, port=port)
            )
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            # assume each node has the same device number
            self.local_rank = int(os.environ.get('LOCAL_RANK'))
            self.node_id = self.rank // torch.cuda.device_count()
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
        """
        rank_bits = DeviceGroup.bitmap(ranks)
        if rank_bits not in self.instance.groups:
            self.groups[rank_bits] = torch.distributed.new_group(list(ranks))
        return self.groups[rank_bits]

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
