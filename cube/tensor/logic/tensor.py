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

    def match(self, communities, ranks=None, val_map_fns=None):
        """
        Match the LogicalTensor with community list.
        """
        # type check
        ranks = [None] * len(communities) if ranks is None else ranks
        val_map_fns = [None] * len(communities) if val_map_fns is None else val_map_fns
        if not isinstance(ranks, list):
            raise TypeError("Expected ranks to be a list or None")
        if not isinstance(ranks, list):
            raise TypeError("Expected ranks to be a list or None")
        
        #TODO: community matching and transformation
        if len(self.communities) == 0:
            for cid in range(len(communities)):
                community = communities[cid]
                self.set_community(community)
                if not community.materialized:
                    rank_list = ranks[cid]
                    val_map_fn = val_map_fns[cid]
                    community.deploy(rank_list, self, val_map_fn)
        else:
            raise NotImplementedError

    @staticmethod
    def construct(shape, communities):
        tensor = LogicalTensor(shape=shape, init_data=False)
        for community in communities:
            tensor.set_community(community)
        return tensor

    def get_physical_tensor(self, segment):
        """
        Get physical tensor from the community.

        Args:
            idx: index for community
        
        Returns:
            torch.Tensor or None
        """
        community = self.communities[segment]
        return community.get_physical_tensor()

    def get_community(self, segment_or_index):
        """
        Get Community based on the segment

        Args:
            segment_or_index (DataSegment or int):
        
        Returns:
            Community
        """
        if isinstance(segment_or_index, DataSegment):
            return self.communities[segment_or_index]
        elif isinstance(segment_or_index, int):
            return self.communities[self.segments[segment_or_index]]
        else:    
            raise ValueError("Expected (derived) DataSegment to chooese Community")
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
        if not isinstance(community, Community):
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
