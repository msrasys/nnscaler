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

    def __init__(self, logical_tensor, segment, reduction=None):
        """Create Community based on the logical tensor

        Community manages one:

        1). Logical Tensor data mapping to Physical Tensor data storage
        2). Materialized Physical Tensors

        Attribute:
            parent (LogicalTensor):
                Logical Tensor the Community belongs to
            segment (DataSegment):
                indices of logical_tensor for this community
            reduction (Callable or None):
                Reduction function for retrieve back physical tensors

        """
        # connection to logical tensor
        self.parent = logical_tensor

        # DataSegment to indicate both element set and data format mapping
        self.segment = segment

        # connection to physical tensor (the PyTorch Tensor)
        self.phsyical_tensor = None
        self.materialized = False

        self.reduction = reduction

    def spread(self, device_list):
        """Spread physical tensors to devices
    
        Materialize physical tensors for this community and spread out
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

    def get_physical_tensor(self):
        """Get physical tensor if materialized

        Returns:
            PhysicalTensor (if materialized)
        """"
        if self.materialized:
            return self.physical_tensor
        else:
            raise RuntimeError("The Community has not been materialized to physical tensors")
