
from cube.schedule.adapter.transform import IRTransformType
from cube.schedule.adapter.transform import IRTensorTransform

from cube.graph.tensor import IRFullTensor, IndexMap


def test_tensor_reshape_init():

    tensor1 = IRFullTensor(shape=[1024,1024], name='test1').tosub()
    
    tensor2 = tensor1.select(
        indices = (slice(0, 512), slice(0, 1024)),
        val_map  = None,
        shape = [512, 1024]
    )

    tensor3 = tensor1.select(
        indices = (slice(512, 1024), slice(0, 1024)),
        val_map  = None,
        shape = [512, 1024]
    )

    reshape = IRTensorTransform(
        src_tensors=[tensor1],
        dst_tensors=[tensor2, tensor3]
    )

    assert len(reshape.inputs()) == 1
    assert len(reshape.outputs()) == 2
    assert reshape.ttype == IRTransformType.Select
    assert reshape.select_indices == [
        IndexMap((slice(0, 512, 1), slice(0, 1024, 1))),
        IndexMap((slice(512, 1024, 1), slice(0, 1024, 1))),
    ]
    assert reshape.merge_axis is None

    reshape = IRTensorTransform(
        dst_tensors=[tensor1],
        src_tensors=[tensor2, tensor3]
    )

    assert len(reshape.inputs()) == 2
    assert len(reshape.outputs()) == 1
    assert reshape.ttype == IRTransformType.Merge
    assert reshape.merge_axis == 0
    assert len(reshape.select_indices) == 0


def test_adapter_select_is_identity():

    tensor1 = IRFullTensor(shape=[1024,1024], name='test1').tosub()
    
    tensor2 = tensor1.select(
        indices = (slice(512, 1024), slice(0, 1024)),
        val_map  = None,
        shape = [512, 1024]
    )

    tensor3 = tensor2.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map  = None,
        shape = [256, 1024]
    )

    tensor4 = tensor1.select(
        indices = (slice(512, 768), slice(0, 1024)),
        val_map = None,
        shape = [256, 1024]
    )

    tensor5 = tensor1.select(
        indices = (slice(512, 768), slice(0, 1024)),
        val_map = None,
        shape = [256, 1024]
    )

    reshape = IRTensorTransform(
        src_tensors=[tensor2],
        dst_tensors=[tensor4, tensor5]
    )
    assert not reshape.is_identity()

    reshape = IRTensorTransform(
        src_tensors=[tensor3],
        dst_tensors=[tensor4, tensor5]
    )
    assert reshape.is_identity()
