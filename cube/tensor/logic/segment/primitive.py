"""
This is the runtime primitive sets to setup community for a logical tensor.
"""


## Basic structure for holding a segment -> cover all the cases ##
class DataSegment:
    """
    The basic primitive to gather data in the logical tensor.

    The order of indices indicate the physical storage (1-D array) order
    """

    def __init__(self, indices_list=None):
        """
        Args:
            indices_list (list[ list[int], ]):
                List of index
        """

        self.indices = indices_list

    def convert_to_indices(self):
        """
        Convert to index list
        """
        pass

    def reorder(self, new_orders):
        """
        Reorder the indices.

        Note this can be only called before materialize physical tensors,
        or called from underlying operation that will change physical storage format
        """
        #TODO: validation check
        self.indices = new_orders


## Higher structure to cover the most cases ##
class TileSegment(DataSegment):
    """
    A tile is a contigonous block on the logical tensor shape,
    which can be represented as the start position + offset (shape)
    """

    def __init__(self, anchor, offset):
        """
        Args:
            anchor (list[int]): start position of the tile
            offset (list[int]): offset (shape) of the tile
        """
        if len(anchor) != len(offset):
            raise ValueError("Require anchor length to be equal with offset length")
        super().__init__()
        self.anchor = anchor
        self.offset = offset
    
    def convert_to_indices(self):
        """
        Convert anchor and offset to index list
        """
        pass

    def reorder(self):
        pass


## Primitive sets for translation ##

def create_from_indices(indices):
    return DataSegment(indices)


def create_from_tiles(anchor, offset):
    # segments = list()
    # dims = len(offset)
    # for dim_id in range(dims):
    #     indices = None # -> TODO: generate indices along the dim_id
    #     segment = create_from_indices(indices)
    #     segments.append(segment)
    # segment = merge_segments(segments)
    # return segment
    return TileSegment(anchor, offset)