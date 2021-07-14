import torch


__all__ = ['ReductionOpPool', 'Community']


class _Reduction(type):

    Sum = torch.distributed.all_reduce

    # identity for replica
    Replica = lambda physical_tensor, group : physical_tensor

    def register(name, udf):
        """
        Reduction functions should be in function format:

        Arguments:
            PhysicalTensor
            Communication Group

        Return:
            PhysicalTensor
        """
        if hasattr(cls, name):
            raise KeyError("{} is registered".format(name))
        setattr(cls, name, udf)


class ReductionOpPool(metaclass=_Reduction):
    pass


class Community:

    def __init__(self, logical_tensor, reduction=None):
        """Create Community based on the logical tensor

        Attribute:
            parent (LogicalTensor):
                Logical Tensor the Community belongs to
            reduction (Callable or None):
                Reduction function for retrieve back physical tensors

        """
        self.parent = logical_tensor
        self.reduction = reduction

    def spread(self, device_list):
        """Spread physical tensors to devices
    
        Create physical tensors for this community and spread out
        based on the given device list.

        This offers policy module an interface to decide which devices
        to spread.

        Argument:
            device_list (list[int]): device id list
        
        Return:
            PhysicalTensor(s) or None:

                For SPMD programming model:
                    if current device is in the `device_list`,
                        than return the corresponding physical tensor,
                    else None

                For Global View programming model:
                    return list[PhysicalTensor] with the same
                    order of `device_list`.
        """
        pass

    def sync(self):
        """Synchrnoize the spread physical tensors by reduction operation"""
        pass
