from cube.device.physic.group import DeviceGroup
from cube.tensor.indices import BaseIndices

import torch


class Segment:

    def __init__(self, logical_tensor, indices, val_map_op, shape):
        """Create Segment based on the logical tensor

        Segment manages:

        1). LogicalTensor indices mapping to Physical Tensor data storage
        2). Materialized Physical Tensor

        Attribute:
            indices (tuple(slice,) or list[list[int]]):
                indices of logical_tensor for this segment
            val_map_op (None or callable):
                deploy op to take logical value and map
            merge_op (None or callable):
                merge op to take physical tensor
        """
        if not isinstance(indices, BaseIndices):
            raise TypeError("Expected indices to be BaseIndices")

        # logical tensor
        self.logical_tensor = logical_tensor
        
        # segment info
        self.indices = indices
        self.val_map_ops = list()
        self.add_val_map_op(val_map_op)
        self.shape = tuple(shape)

        # physical tensor (the PyTorch Tensor)
        self.physical_tensor = None

        # deploy information
        self.placement = list()
        self.group = None
        self.materialized = False

        # recover op
        self.merge_op = None

    def deploy(self, ranks):
        """deploy (materialize) to physical tensors
    
        Materialize physical tensors for this community and spread out
        based on the given device list.

        This offers policy module an interface to decide which devices
        to spread.

        Argument:
            ranks (list[int]): device id list
            value_map_op (callable):
                takes the tensor, rank, world_size,
                return a new tensor
        """
        if not isinstance(ranks, list):
            raise TypeError("Expected ranks in list[int]")

        rank = DeviceGroup().rank
        self.placement = ranks
        self.group = DeviceGroup().get_group(ranks)
        if rank in ranks:
            if self.logical_tensor.data is None:
                raise RuntimeError("Try deploying a segment from a logical tensor without data")
            # select from logical data
            self.physical_tensor = torch.empty(tuple(self.shape), device='cuda')
            self.physical_tensor.copy_(
                self.logical_tensor.data[self.indices.get()].reshape(self.shape)
            )
            for val_map_op in self.val_map_ops:
                self.physical_tensor.data = val_map_op(self.physical_tensor, rank, len(ranks))
        self.materialized = True

    def recover(self, reduction_op):
        """
        Recover the deployed physical tensors by reduction operation

        Each rank can call this even there is no physical tensor on it.

        Args:
            reduction_op (callable):
                inplacement update on physical tensor
        
        Returns:
            None. The physical tensor will be updated to match logical data
        """
        if self.materialized:
            if self.physical_tensor is not None:
                reduction_op(self.physical_tensor, group=self.group)
        else:
            raise RuntimeError("The Segment has not been materialized")

    def add_val_map_op(self, val_map_op):
        """
        Append val_map_op to the end 
        """
        if val_map_op is not None:
            if not callable(val_map_op):
                raise TypeError("Expected val_map_op to be callable or None")
            self.val_map_ops.append(val_map_op)

    def get_physical_tensor(self):
        """Get physical tensor if materialized

        Returns:
            PhysicalTensor (if materialized)
        """
        if self.materialized:
            return self.physical_tensor
        else:
            raise RuntimeError("The Segment has not been materialized")

    def set_physical_tensor(self, physical_tensor, ranks):
        if self.materialized:
            raise RuntimeError("Setting physical tensors to a materialized community")
        if not isinstance(ranks, list):
            raise TypeError("ranks: Expected a list[int]")
        if physical_tensor is not None:
            if list(physical_tensor.size()) != list(self.shape):
                raise RuntimeError(
                    "Trying to set a community where physical tensor shape "
                    "doesn't match with segment shape")
        self.physical_tensor = physical_tensor
        self.group = DeviceGroup().get_group(ranks)
        self.materialized = True
