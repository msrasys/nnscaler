"""
This should be the interface with C level kernel launch
"""

from cube.device.physic.group import DeviceGroup
import torch

class OpResult:
    """
    The empty result is used for re-constructing community
    """
    def __init__(self, result, ranks):
        self.res = result
        self.placement = ranks
    
    def get_result(self):
        return self.res

    def __repr__(self):
        return "OpResult(res={}, placement={})".format(self.res, self.placement)


class GenericPhysicOp:
    """
    The generic physical op takes at least one physical tensor,
    and generates at least one physical tensor.

    If there is no tensor as input, will return an empty result
    which indicates which rank will generate the correct one.
    """

    def __init__(self, func, placement=None):
        
        if not callable(func):
            raise TypeError("Expect callable function")
        if not (isinstance(placement, list) or placement is None):
            raise TypeError("Expected placement init with None or list[int]")
        self.func = (func,)
        self._placement = None
        self.execute_flag = False
        self.policy_fn = None
        if isinstance(placement, list):
            self.placement = placement
    
    @property
    def placement(self):
        """
        Ranks for the op to execute
        """
        return self._placement

    @placement.setter
    def placement(self, ranks):
        if not isinstance(ranks, list):
            raise TypeError("Expected list of int ranks")
        self._placement = ranks
        if DeviceGroup().rank not in self.placement:
            self.execute_flag = False
        else:
            self.execute_flag = True

    def register_policy(self, policy_fn):
        if not callable(policy_fn):
            raise TypeError("Expected callable policy function")
        self.policy_fn = [policy_fn]

    def __call__(self, *args, **kwargs):
        #TODO: fix for model-partition with send/recv
        if self.placement is None:
            if self.policy_fn is None:
                #TODO: fix: this will break between-device consistency view
                self.placement = [torch.cuda.current_device()]
            else:
                self.placement = self.policy_fn(*args, **kwargs)
        if not self.execute_flag:
            return OpResult(None, self.placement)

        # tensor movement
        for arg in args:
            if torch.is_tensor(arg):
                #TODO: rank -> device mapping, send/recv
                if arg.device.index not in self.placement:
                    #TODO: rank -> device mapping, send/recv
                    arg.data = arg.detach().cuda()
        for key in kwargs:
            if torch.is_tensor(kwargs[key]):
                #TODO: rank -> device mapping, send/recv
                if kwargs[key].device.index not in self.placement:
                    # TODO: rank -> device mapping, send/recv
                    kwargs[key].data = kwargs[key].detach().cuda()

        outputs = self.func[0](*args, **kwargs)
        return OpResult(outputs, self.placement)
