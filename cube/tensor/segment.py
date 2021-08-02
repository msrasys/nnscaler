from cube.device.physic.group import DeviceGroup
from cube.tensor.indices import BaseIndices

import torch


class Segment:

    def __init__(self, logical_tensor, indices, val_op, shape):
        """Create Segment based on the logical tensor

        Segment manages:

        1). LogicalTensor indices mapping to Physical Tensor data storage
        2). Materialized Physical Tensor

        Attribute:
            indices (tuple(slice,) or list[list[int]]):
                indices of logical_tensor for this segment
            val_op (ValueMapReduceOp):
                deploy op to take logical value and group in for value mapping
                merge op to take mapped value and group in for value reduction
        """
        if not isinstance(indices, BaseIndices):
            raise TypeError("Expected indices to be BaseIndices")

        # logical tensor
        self.logical_tensor = logical_tensor
        
        # segment info
        self.indices = indices
        self.shape = tuple(shape)

        # val ops
        self.val_ops = list()
        self.val_op_segs = list()
        self.add_val_op(val_op)

        # physical tensor (the PyTorch Tensor)
        self.physical_tensor = None

        # deploy information
        self.placement = list()
        self.group = None
        self.materialized = False

    def deploy(self, ranks=None):
        """deploy (materialize) to physical tensors
    
        Materialize physical tensors for this community and spread out
        based on the given device list.

        This offers policy module an interface to decide which devices
        to spread.

        Argument:
            ranks (list[int] or None): 
                if rank id list: deploy based on this list
                if None: deploy based on setted self.placement
            value_map_op (callable):
                takes the tensor, rank, world_size,
                return a new tensor
        """
        if isinstance(ranks, list):
            self.placement = ranks
        elif ranks is None and self.placement is None:
            raise TypeError("Expected self.placement when ranks is None")

        #TODO: remove this constraints
        if len(self.val_ops) > 0 and len(self.placement) > 1:
            raise RuntimeError("Currently segment with val_ops only allows to deploy on one rank")

        rank = DeviceGroup().rank
        self.group = DeviceGroup().get_group(self.placement)

        # set physical tensors
        if rank in self.placement:
            if self.logical_tensor.data is None:
                raise RuntimeError("Try deploying a segment from a logical tensor without data")
            # select from logical data
            self.physical_tensor = torch.empty(tuple(self.shape), device='cuda')
            self.physical_tensor.copy_(
                self.logical_tensor.data[self.indices.get()].reshape(self.shape)
            )

        # go through val_op
        for val_op, segs in zip(self.val_ops, self.val_op_segs):
            if len(segs) == 0:
                raise RuntimeError("Missing segments for val op")
            op_ranks = [seg.placement[0] for seg in segs]
            group = DeviceGroup().get_group(op_ranks)
            if rank in self.placement:
                self.physical_tensor.data = val_op.map(self.physical_tensor, group)

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

    def add_val_op(self, val_op):
        """
        Append val_op to the end 
        """
        if val_op is not None:
            if not (callable(val_op.map) and callable(val_op.reduce)):
                raise TypeError("Expected val_op to be ValMapReudceOp")
            self.val_ops.append(val_op)

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

    def __repr__(self):
        return 'Segment(Indices: {} | Materialized: {})'.format(
            self.indices, self.materialized
        )
    