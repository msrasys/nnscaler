import cube.tensor.logic.segment as segment
import torch


def test_reduction_op_register():

    def reduce_fn(physical_tensor, group):
        return physical_tensor
    segment.ReductionOp.register("ReduceSum", reduce_fn)

    # segment.ReductionOp.register("Replica", reduce_fn)

    tensor = torch.randn((3,4))
    out = segment.ReductionOp.ReduceSum(tensor, None)
    assert out is tensor


## TODO: test all the provided reduction op
def test_reduction_op_replica():
    #TODO: check correctness
    assert callable(segment.ReductionOp.Replica)


def test_data_segment_init():

    tensor = torch.randn((10,10,10))
    indices = [[5,3,2,4],
               [1,2,7,4],
               [3,4,5,4]]
    seg = segment.DataSegment(
        indices, shape=(4,1), reduction=segment.ReductionOp.Replica)
    assert seg.indices == indices
    assert seg.shape == (4,1)
    assert seg.reduction == segment.ReductionOp.Replica


def test_data_segment_get_indices():

    tensor = torch.randn((10,10,10))
    indices = [[5,3,2,4],
               [1,2,7,4],
               [3,4,5,4]]
    seg = segment.DataSegment(
        indices, shape=(4,1), reduction=segment.ReductionOp.Replica)
    sub_tensor = tensor[seg.get_indices()]
    assert sub_tensor.size() == torch.Size([4])


def test_data_segment_reorder():

    tensor = torch.randn((10,10,10))
    indices = [[5,3,2,4],
               [1,2,7,4],
               [3,4,5,4]]
    seg = segment.DataSegment(
        indices, shape=(4,1), reduction=segment.ReductionOp.Replica)
    sub_tensor = tensor[seg.get_indices()]

    seg.reorder([2,3,1,0])
    ref_tensor = sub_tensor[([2,3,1,0])]
    check_tensor = tensor[seg.get_indices()]
    assert torch.all(torch.eq(ref_tensor, check_tensor))


def test_tile_segment_init():

    tensor = torch.randn((10,10,10))
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4), reduction=segment.ReductionOp.Replica)
    assert seg.shape == (4,4,4)
    assert seg.anchor == (2,3,1)
    assert seg.reduction == segment.ReductionOp.Replica


def test_tile_segment_get_indices():

    tensor = torch.randn((10,10,10))
    seg = segment.TileSegment(
        anchor=(2,3,1), shape=(4,4,4), reduction=segment.ReductionOp.Replica)
    ref_tensor = tensor[(slice(2,2+4), slice(3,3+4), slice(1,1+4))]
    sub_tensor = tensor[seg.get_indices()]
    assert sub_tensor.size() == torch.Size([4,4,4])
    assert torch.all(torch.eq(ref_tensor, sub_tensor))


if __name__ == '__main__':

    test_reduction_op_register()
    test_reduction_op_replica()
    test_data_segment_init()
    test_data_segment_get_indices()
    test_data_segment_reorder()
    test_tile_segment_init()
    test_tile_segment_get_indices()