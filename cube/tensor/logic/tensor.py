from cube.tensor.community import Community, ReductionOpPool


class LogicalTensor:

    def __init__(self, ):
        
        self.communities = list()

    def create_community(self, segment):
        """Create a community by given the segment"""
        self.communities.append(segment)

    def fuse(self, communities=None, reduction=ReductionOpPool.Replica):
        """Fuse multiple communities into one

        Synchronization will done for each community to retrieve the right
        result.
        
        Args:
            communities (list[Community]):
                The particular comunities to merge.
                If not specified (None), fuse all the communities.
            reduction:
                Reduction operator for the new fused community.
        """"
        pass
