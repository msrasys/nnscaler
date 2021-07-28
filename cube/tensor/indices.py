"""
Basic structure for holding indices -> cover all the cases
"""


class BaseIndices:
    """
    The basic primitive to gather data in the logical tensor.

    The order of indices indicate the physical storage (1-D array) order
    """

    def __init__(self, indices_list):
        """
        Args:
            indices_list (list[list[int],], tuple(slice(int, int),)):
                indices list
        """
        self.indices = tuple(indices_list)

    def get(self):
        """
        Get indexable indices
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
        for dim in range(len(self.indices)):
            self.indices[dim] = [self.indices[dim][idx] for idx in new_orders]

    def __repr__(self):
        msg = 'BaseIndices(indices_len={})'.format(
            len(self.indices), self.reduction
        )


class TileIndices(BaseIndices):
    """
    A tile is a contigonous block on the logical tensor shape,
    which can be represented as the start position + offset (shape)
    """

    def __init__(self, anchor, shape):
        """
        Args:
            anchor (list[int]): start position of the tile
            offset (list[int]): offset (shape) of the tile
        """
        indices = list()
        for start, ofst in zip(self.anchor, self.shape):
            indices.append(slice(start, start + ofst))
        super().__init__(tuple(indices))
        self.anchor = anchor
        self.shape = shape

    def reorder(self):
        raise NotImplementedError

    def __repr__(self):
        msg = 'TileIndices(anchor={}, shape={})'.format(
            self.anchor, self.shape
        )
        return msg
