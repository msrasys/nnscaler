from cube.tensor.community import Community
from cube.tensor.logic.segment.segment import DataSegment


class LogicalTensor:
    """
    The logical tensor
    """

    def __init__(self, shape, init_data=True):
        
        self.shape = shape
        # segment -> community
        self.communities = dict()
        self.segments = list()
        self.data = None
        if init_data:
            import torch
            self.data = torch.randn(shape).detach()

    def get_physical_tensor(self, segment):
        """
        Get physical tensor from the community.

        Args:
            idx: index for community
        
        Returns:
            torch.Tensor or None
        """
        community = self.communities[idx]
        return community.get_physical_tensor()

    def get_community(self, segment):
        """
        Get Community based on the segment
        """
        if not isinstance(segment, DataSegment):
            raise ValueError("Expected (derived) DataSegment to chooese Community")
        if segment not in self.communities:
            raise KeyError("The segment doesn't found in current tensor")
        return self.communities[segment]

    def __getitem__(self, key):
        """
        key:
            if key is DataSegment, return community
            ##TODO: DOUBLE CHECK
            if key is slice, return new logical tensor
        """
        if isinstance(key, DataSegment):
            return self.get_community(key)
        else:
            ## TODO: should return logical tensor / views
            return self.data[key]

    def create_community(self, segment):
        """
        Create a community by given the segment
        """
        if segment in self.communities:
            raise KeyError("The segment already exists")
        self.communities[segment] = Community(segment)
        self.segments.append(segment)
    
    def set_community(self, community):
        """
        Set a community

        Warning: if there is a segment in this tensor that matches
        with the given community's segment, the original community
        will be overrided
        """
        if not isinstance(community):
            raise TypeError("Expected a community")
        segment = community.segment
        if segment not in self.communities:
            self.segments.append(segment)
        self.communities[segment] = community

    def remove_community(self, segment):
        """
        Remove a community by given the segment
        """
        #TODO: check whether a sync-back is needed
        if segment not in self.communities:
            raise KeyError("The segment doesn't exist")
        del self.communities[segment]
        self.segments.remove(segment)
