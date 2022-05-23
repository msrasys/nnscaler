r"""
Runtime information
"""

import torch
import os


class EnvResource:

    class __EnvResource:

        def __init__(self):
            # number of gpus
            single_device_mode = os.environ.get('SINGLE_DEV_MODE')
            if single_device_mode:
                self.ngpus = 1
            else:
                self.ngpus = torch.distributed.get_world_size()
            # device topology
            self.topo = None

    instance = None

    def __init__(self):
        if not EnvResource.instance:
            EnvResource.instance = EnvResource.__EnvResource()

    def __getattr__(self, name):
        return getattr(self.instance, name)


    def __setattr__(self, name, val) -> None:
        setattr(EnvResource.instance, name, val)
