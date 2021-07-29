from cube.tensor.segment import Segment
from cube.tensor.indices import BaseIndices


class LogicalTensor:
    """
    The logical tensor
    """

    def __init__(self, shape, init_data=True):
        """
        Create an empty logical tensor with no segmentations

        Args:
            shape (tuple[int] or list[int]):
                shape of the tensor
            init_data (Boolean):
                if True, init a CPU data. Otherwise no data initialized.
        """
        self.shape = tuple(shape)
        self.segments = list()
        self.data = None
        if init_data:
            import torch
            self.data = torch.randn(shape).detach()

    def fill(self, physical_tensors, ranks):
        """
        Construct the logical tensor with physical tensors.

        Args:
            physical_tensors (list[PhysicalTensor, None]):
                the list length should be equal to len(self.segments)
            ranks (list[list[int],]):
                each segment will pair with a list of ranks
        """
        if self.data is not None:
            raise RuntimeError("Only allowed fill physical tensors when data is not None")
        for segment, physical_tensor, ranks in zip(self.segments, physical_tensors, ranks):
            segment.set_physical_tensor(physical_tensor, ranks)
    
    def select(self, indices, shape):
        """
        Create a Segment given the indices for this logical tensor,
        and the Segment will use shape. 
        """
        segment = Segment(self, indices, shape)
        return segment

    def transform(self, segments, ranks=None, val_map_ops=None):
        """
        Transform the LogicalTensor with community list.
        TODO: check if this should create a new logical tensor
        """
        if not (isinstance(ranks, list) and len(ranks) == len(segments)):
            raise ValueError("Expected ranks to be a list with equal length of segments")
        if not (isinstance(ranks, list) and len(val_map_ops) == len(segments)):
            raise ValueError("Expected ranks to be a list with equal length of segments")
        
        if len(self.segments) == 0:
            for sid in range(len(segments)):
                segment = segments[sid]
                self.add_segment(segment)
                if not segment.materialized:
                    deploy_ranks = ranks[sid]
                    if not isinstance(deploy_ranks, list):
                        raise TypeError('Expected ranks to be list[list[int],]')
                    deploy_ops = val_map_ops[sid]
                    segment.deploy(deploy_ranks, deploy_ops)
        #TODO: segment transformation on existing segments
        else:
            raise NotImplementedError
        
    def get_physical_tensor(self, index):
        """
        Get physical tensor from the segment.

        Args:
            idx: index for segment
        
        Returns:
            torch.Tensor or None
        """
        return self.get_segment(index).get_physical_tensor()
    
    def __len__(self):
        """
        Return community number
        """
        return len(self.segments)

    def __getitem__(self, key):
        """

        """
        # TODO: create new logical tensor / change layout
        return self.data[key]

    def get_segment(self, idx):
        """
        Get a segment using index

        Args:
            idx (int): index to segment list

        Returns:
            Segment
        """
        return self.segments[idx]

    def add_segment(self, segment):
        """
        Add a segment.

        Note adding a segment will change the segment parent logical tensor
        to this tensor
        """
        if not isinstance(segment, Segment):
            raise TypeError("Expected a segment")
        segment.logical_tensor = self
        if segment in self.segments:
            raise RuntimeError("Segment is already added")
        self.segments.append(segment)

    def remove_segment(self, segment_or_index):
        """
        Remove a community by given the segment
        """
        #TODO: check whether a sync-back is needed
        if isinstance(segment_or_index, Segment):
            if segment not in self.segments:
                raise KeyError("The segment doesn't exist")
            self.segments.remove(segment)
        elif isinstance(segment_or_index, int):
            del self.segments[segment_or_index]
        else:
            raise ValueError("Expected Segment instance or index int")

