import cube.tensor.logic.segment.outline as outline
import cube.tensor.logic.segment as segment

import torch


def test_full():

    shape = (10,10,10)
    tensor = torch.randn(shape)
    full_dsp = outline.Full(reduction=segment.ReductionOp.Replica)
    assert full_dsp.reduction == segment.ReductionOp.Replica
    
    segments = full_dsp(tensor.shape)
    assert len(segments) == 1
    tile_seg = segments[0]
    assert type(tile_seg) == segment.TileSegment
    
    sub_tensor = tensor[tile_seg.get_indices()]
    assert torch.all(torch.eq(sub_tensor, tensor))


def test_split_axis():

    axis = 1
    num = 8

    shape = (4,16,4)
    tensor = torch.randn(shape)
    split_dsp = outline.SplitAxis(
        axis=axis, chunk_num=None, overlap=0, 
        reduction=segment.ReductionOp.Replica, uniform=True)
    assert split_dsp.axis == 1
    assert split_dsp.chunk_num is None
    assert split_dsp.uniform is True
    assert split_dsp.overlap == 0
    assert split_dsp.reduction == segment.ReductionOp.Replica

    ## Policy here to decide how to split
    if split_dsp.chunk_num is None:
        split_dsp.chunk_num = num
        split_dsp.reduction = [segment.ReductionOp.Replica] * num
    ###


    segs = split_dsp(tensor.shape)
    assert len(segs) == num
    assert torch.all(torch.Tensor([type(seg) == segment.TileSegment for seg in segs]))
    
    ofst = 0
    expected_shape = list(shape)
    expected_shape[axis] = shape[axis] // num
    for cid in range(num):
        seg = segs[cid]
        sub_tensor = tensor[seg.get_indices()]
        ref_tensor = tensor[:,ofst:ofst+expected_shape[axis],:]
        # print('sub tensor {}: {}'.format(sub_tensor.size(), sub_tensor))
        # print('ref tensor {}: {}'.format(ref_tensor.size(), ref_tensor))
        assert sub_tensor.size() == torch.Size(expected_shape)
        assert torch.all(torch.eq(sub_tensor, ref_tensor))
        ofst += expected_shape[axis]


if __name__ == '__main__':

    test_full()
    test_split_axis()