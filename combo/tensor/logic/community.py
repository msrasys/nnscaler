

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
        """Create physical tensors and spread to devices

        Argument:
            device_list (list[int]): device id list
        
        Return:

        """
        pass

    def fuse(self):
        """Fuse the spread physical tensors into the one
        Perform reduction function to get the results on each physical tensor
        """
        pass
