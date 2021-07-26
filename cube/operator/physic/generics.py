"""
This should be the interface with C level kernel launch
"""

import torch


class GenericPhysicOp:

    def __init__(self, func, placement=None):
        
        if not callable(func):
            raise TypeError("Expect callable function")
        self.func = [func]
        self.placement = placement
    
    def set_placement(self, placement):
        if not isinstance(placement, torch.device):
            raise TypeError("Expected torch device")
        self.placement = placement

    def __call__(self, *args, **kwargs):

        # tensor movement
        for arg in args:
            if torch.is_tensor(arg):
                if arg.device != self.placement:
                    arg.data = arg.detach().to(self.placement)
        for key in kwargs:
            if torch.is_tensor(kwargs[key]):
                if kwargs[key].device != self.placement:
                    kwargs[key].data = kwargs[key].detach().to(self.placement)

        outputs = self.func[0](*args, **kwargs)
        return outputs
