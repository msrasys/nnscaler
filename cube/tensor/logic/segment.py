"""
This is the interface for describing which set of data is needed for
gathering a community.
"""


## Basic interface to cover all the cases
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


## Higher level interface to cover the most cases ##
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


# primitives to describe segmentation pattern

class SplitAxis:

    def __init__(self, axis, chunk_num=None, chunk_size=None, overlap=0):
        """
        Segmentation Pattern Requirement (parameters):

        axis (int): the axis to split

        chunk_num (None, int, tuple(int, int)):
            valid chunk numbers to split.
            If None, then any chunk number is valid;
            If an integer, only the specified chunk number is valid;
            If a tuple(min, max), the chunk number wihtin the scope [min,max] is valid

        chunk_size (None, int, tuple(int, int)):
            valid chunk size.
            If None, any size is valid;
            If an integer, each chunk size is valid;
            if a tuple(min, max), the chunk size wihtin the scope [min,max] is valid
        
        overlap (0, int, tuple(int, int)):
            valid size for overlaping on the boundary of each splitted chunks.
            If None, any overlapping is valid
            If an integer, each overlap size is valid;
            if a tuple(min, max), the overlap size wihtin the scope [min,max] is valid

        """
        self.axis = axis
        self.chunk_num = chunk_num
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def __call__(self, shape):
        """
        Runtime community generation given the logical tensor shape
        """
        pass
