"""
This is the interface for describing which set of data is needed for
gathering a community.
"""

## Basic interface to cover all the cases


class DataSegment:
    """
    The basic primitive to gather data in the logical tensor.

    """

    def __init__(self, indices_list=None):
        """
        Args:
            indices_list (list[ list[int] ]):
                List of index
        """

        self.indices = indices_list

    def convert_to_indices(self):
        """
        Convert to index list
        """
        pass


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
    

