import cube.tensor.logic.segment.outline as outline
import cube.tensor.logic.segment as segment

import torch


def test_base():

    dsp1 = outline.BaseOutline(reduction=segment.ReductionOp.Sum)
    assert isinstance(dsp1.reduction, outline.ConstantContainer)
    assert dsp1.reduction.get() == segment.ReductionOp.Sum

    choice = {segment.ReductionOp.Sum, segment.ReductionOp.Replica}
    dsp2 = outline.BaseOutline(reduction=choice)
    assert isinstance(dsp2.reduction, outline.MutableContainer)
    assert dsp2.reduction.get() is None
    assert dsp2.reduction.get(scope=True) == choice


def test_full():

    shape = (10,10,10)
    tensor = torch.randn(shape)
    full_dsp = outline.Full(reduction=segment.ReductionOp.Replica)
    assert full_dsp.reduction.get() == segment.ReductionOp.Replica
    
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
    assert split_dsp.axis.get() == 1
    assert split_dsp.chunk_num.get() is None
    assert split_dsp.uniform.get() is True
    assert split_dsp.overlap.get() == 0
    assert split_dsp.reduction.get() == segment.ReductionOp.Replica

    ## Policy here to decide how to split
    if split_dsp.chunk_num.get() is None:
        split_dsp.chunk_num = num
    ###


    segs = split_dsp(tensor.shape)
    assert len(segs) == num
    assert torch.all(
        torch.Tensor(
            [type(seg) == segment.TileSegment for seg in segs])).item() is True
    
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


def test_align():

    dsp1 = outline.SplitAxis(
        axis=1, chunk_num=None, overlap=0, 
        reduction=segment.ReductionOp.Replica, uniform=True)
    
    dsp2 = outline.SplitAxis(
        axis=2, chunk_num=dsp1.chunk_num, overlap=0, 
        reduction=segment.ReductionOp.Replica, uniform=True)
    
    dsp1.chunk_num = 3
    assert dsp2.chunk_num.get() == 3
    assert dsp2.axis.get() == 2


if __name__ == '__main__':

    test_base()
    test_full()
    test_split_axis()
    test_align()