"""
This is the runtime primitive sets to setup community for a logical tensor.
"""

import torch


# TODO: reduction op should be in torch autograd function
class _Reduction(type):

    Sum = torch.distributed.all_reduce

    # identity for replica
    Replica = lambda physical_tensor, group : physical_tensor

    def register(cls, name, udf):
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


class ReductionOp(metaclass=_Reduction):
    pass


## Basic structure for holding a segment -> cover all the cases ##
class DataSegment:
    """
    The basic primitive to gather data in the logical tensor.

    The order of indices indicate the physical storage (1-D array) order
    """

    def __init__(self, indices_list=None, shape=None, reduction=None):
        """
        Args:
            indices_list (list[ list[int], ]):
                List of index
            reduction (ReductionOp):
                How to reduction to the logical value
            shape:
                shape on the indices list
        """

        self.indices = indices_list
        if shape is None:
            if indices_list is None:
                raise RuntimeError("Provide shape if indices_list is empty")
            self.shape = (len(indices_list[0]),)
        else:
            # TODO: check shape
            self.shape = shape
        self.reduction = staticmethod(reduction)

    def get_indices(self):
        """
        Convert to index list
        """
        return tuple(self.indices)

    def reorder(self, new_orders):
        """
        Reorder the indices.

        Note this can be only called before materialize physical tensors,
        or called from underlying operation that will change physical storage format

        Args:
            new_orders (iteratable): order of each index
        """
        #TODO: check if materialized
        for dim in range(len(self.indices)):
            self.indices[dim] = [self.indices[dim][idx] for idx in new_orders]


## Higher structure to cover the most cases ##
class TileSegment(DataSegment):
    """
    A tile is a contigonous block on the logical tensor shape,
    which can be represented as the start position + offset (shape)
    """

    def __init__(self, anchor, shape, reduction=None):
        """
        Args:
            anchor (list[int]): start position of the tile
            offset (list[int]): offset (shape) of the tile
        """
        if len(anchor) != len(shape):
            raise ValueError("Require anchor length to be equal with offset length")
        super().__init__(shape=shape, reduction=reduction)
        self.anchor = anchor
    
    def get_indices(self):
        """
        Convert anchor and offset to index list
        """
        indices = list()
        for start, ofst in zip(self.anchor, self.shape):
            indices.append(slice(start, start + ofst))
        return tuple(indices)

    def reorder(self):
        pass


## Primitive sets for translation ##

def create_from_indices(indices, shape, reduction):
    """
    Create a data segment from indices, and format in shape.
    The indices list will determine how data will be organized in
    storage.

    Args:
        indices (list[list[int]]): 
            Represent indices from logical tensor shape
            len(indices) is the dimension,
            e.g., index [3,4,5] and [2,7,9] is represented as
                  [[3,2], [4,7], [5,9]]
        shape (tuple or list):
            the segment shape
        reduction (ReductionOp):
            How to generate correct logical results from reduction op.

    Returns:
        DataSegment instance
    """
    return DataSegment(indices, shape, reduction)


def create_from_tiles(anchor, shape, reduction):
    # segments = list()
    # dims = len(offset)
    # for dim_id in range(dims):
    #     indices = None # -> TODO: generate indices along the dim_id
    #     segment = create_from_indices(indices)
    #     segments.append(segment)
    # segment = merge_segments(segments)
    # return segment
    return TileSegment(anchor, shape, reduction)