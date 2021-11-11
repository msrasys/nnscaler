from cube.schedule.adapter.transform import IRTransformType
from cube.schedule.adapter.transform import IRTensorTransform

from cube.graph.tensor import IRFullTensor


def test_tensor_transform_select():

    tensor1 = IRFullTensor(shape=[1024,1024], name='test1').tosub()
    
    tensor2 = tensor1.select(
        indices = (slice(0, 512), slice(0, 1024)),
        val_map  = (0, 1),
        shape = [512, 1024]
    )

    tensor3 = tensor1.select(
        indices = (slice(512, 1024), slice(0, 1024)),
        val_map  = (0, 2),
        shape = [512, 1024]
    )

    tensor4 = tensor3.select(
        indices = (slice(0, 256), slice(0, 512)),
        val_map = (0, 1),
        shape = [256, 512]
    )

    tensor5 = tensor3.select(
        indices = (slice(256, 512), slice(0, 512)),
        val_map = (0, 1),
        shape = [256, 512]
    )

    select1 = IRTensorTransform(
        src_tensors=[tensor1],
        dst_tensors=[tensor2, tensor3]
    )
    assert len(select1.inputs()) == 1
    assert len(select1.outputs()) == 2
    assert select1.ttype == IRTransformType.Select

    print('> select1:', select1)
    for prim in select1.trace():
        print(prim)

    select2 = IRTensorTransform(
        src_tensors=[tensor3],
        dst_tensors=[tensor4, tensor5]
    )
    print('> select2:', select2)
    for prim in select2.trace():
        print(prim)
    assert False


def test_tensor_transform_merge():
    tensor0 = IRFullTensor(shape=[1024,1024], name='test1').tosub()
    
    tensor1 = tensor0.select(
        indices = (slice(0, 512), slice(0, 512)),
        val_map  = None,
        shape = [256, 1024]
    )

    tensor2 = tensor0.select(
        indices = (slice(0, 512), slice(512, 1024)),
        val_map  = None,
        shape = [256, 1024]
    )

    tensor3 = tensor0.select(
        indices = (slice(512, 1024), slice(0, 512)),
        val_map = None,
        shape = [256, 512]
    )

    tensor4 = tensor0.select(
        indices = (slice(512, 1024), slice(512, 1024)),
        val_map = None,
        shape = [256, 512]
    )

    tensor5 = tensor0.select(
        indices = (slice(512, 1024), slice(0, 1024)),
        val_map = None,
        shape = [256, 512]
    )

    merge1 = IRTensorTransform(
        src_tensors=[tensor1, tensor2, tensor3, tensor4],
        dst_tensors=[tensor0]
    )
    assert len(merge1.inputs()) == 4
    assert len(merge1.outputs()) == 1
    assert merge1.ttype == IRTransformType.Merge

    print('> merge1:')
    for prim in merge1.trace():
        print(prim)
    assert merge1.trace()[-1].output == tensor0
    assert merge1.trace()[-1].output._id == tensor0._id

    merge2 = IRTensorTransform(
        src_tensors=[tensor3, tensor4],
        dst_tensors=[tensor5]
    )
    print('> merge2:')
    for prim in merge2.trace():
        print(prim)
    assert merge2.trace()[-1].output == tensor5
    assert merge2.trace()[-1].output._id == tensor5._id
    # assert False

    tensor6 = tensor0.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map = (0, 4),
        shape = [256, 1024]
    )
    tensor7 = tensor0.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map = (1, 4),
        shape = [256, 1024]
    )
    tensor8 = tensor0.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map = (2, 4),
        shape = [256, 1024]
    )
    tensor9 = tensor0.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map = (3, 4),
        shape = [256, 1024]
    )

    tensor10 = tensor0.select(
        indices = (slice(0, 256), slice(0, 1024)),
        val_map = (0, 1)
    )

    merge3 = IRTensorTransform(
        src_tensors=[tensor6, tensor7, tensor8, tensor9],
        dst_tensors=[tensor10]
    )
    print('> merge3:')
    for prim in merge3.trace():
        print(prim)
    assert merge3.trace()[-1].output._id == tensor10._id
    # assert False


def test_transform_identity():

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

    select1 = IRTensorTransform(
        src_tensors=[tensor2],
        dst_tensors=[tensor4, tensor5]
    )
    assert not select1.is_identity()

    select2 = IRTensorTransform(
        src_tensors=[tensor3],
        dst_tensors=[tensor4, tensor5]
    )
    assert select2.is_identity()

    merge1 = IRTensorTransform(
        src_tensors=[tensor4],
        dst_tensors=[tensor5]
    )
    assert merge1.is_identity()
