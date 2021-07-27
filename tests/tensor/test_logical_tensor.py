from cube.tensor.logic.tensor import LogicalTensor
from cube.tensor.community import Community
import cube.tensor.logic.segment as segment


def test_logical_tensor_init():

    #TODO
    pass


def test_logical_tensor_construct():

    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4),
        reduction=segment.ReductionOp.Replica)
    community = Community(seg)

    logical_tensor = LogicalTensor.construct((10,10,10), [community])

    assert isinstance(logical_tensor, LogicalTensor)
    assert len(logical_tensor.communities) == 1
    assert logical_tensor.get_community(0) is community
    assert logical_tensor.shape == (10,10,10)


if __name__ == '__main__':

    test_logical_tensor_init()
    test_logical_tensor_construct()