import torch
from cube.device.physic.group import DeviceGroup


__all__ = ['Community']


class Community:

    def __init__(self, segment):
        """Create Community based on the logical tensor

        Community manages one:

        1). Logical Tensor data mapping to Physical Tensor data storage
        2). Materialized Physical Tensors

        Attribute:
            segment (DataSegment):
                indices of logical_tensor for this community
        """
        # connection to logical tensor
        # DataSegment to indicate both element set and data format mapping
        self.segment = segment

        # connection to physical tensor (the PyTorch Tensor)
        self.physical_tensor = None
        self.group = list()
        self.materialized = False

    def deploy(self, ranks, logic_tensor, value_map_fn=None):
        """deploy (materialize) to physical tensors
    
        Materialize physical tensors for this community and spread out
        based on the given device list.

        This offers policy module an interface to decide which devices
        to spread.

        Argument:
            ranks (list[int]): device id list
            value_map_fn (callable):
                takes the tensor, rank, world_size,
                return a new tensor
        """
        
        rank = DeviceGroup().rank
        self.group = DeviceGroup().get_group(ranks)
        if rank not in ranks:
            self.physical_tensor = None
        else:
            if logic_tensor.data is None:
                # TODO: check overlap
                self.physical_tensor = torch.randn(tuple(self.segment.shape), device='cuda')
            else:
                # select from cpu view
                self.physical_tensor = torch.empty(tuple(self.segment.shape), device='cuda')
                self.physical_tensor.copy_(logic_tensor.data[self.segment.get_indices()])
            if value_map_fn is not None:
                self.physical_tensor.data = value_map_fn(self.physical_tensor)
        self.materialized = True

    def sync(self):
        """
        Synchrnoize the spread physical tensors by reduction operation

        This should be a out-placement device for differentiable communication ops.
        
        Each device should call this, including no-physical-tensor devices
        """
        if self.materialized:
            if self.physical_tensor is not None:
                # self.segment.reduction.__func__(self.physical_tensor, group=self.group)
                torch.distributed.all_reduce(self.physical_tensor, group=self.group)
        else:
            raise RuntimeError("The Community has not been materialized to physical tensors")

    def get_physical_tensor(self):
        """Get physical tensor if materialized

        Returns:
            PhysicalTensor (if materialized)
        """
        if self.materialized:
            return self.physical_tensor
        else:
            raise RuntimeError("The Community has not been materialized to physical tensors")
