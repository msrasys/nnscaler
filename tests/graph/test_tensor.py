import copy

from cube.graph.tensor import IRFullTensor, IRSubTensor


def test_full_tensor_init():

    tensor = IRFullTensor(shape=[1024,1024], name='full_tensor')
    assert tensor.shape == [1024, 1024]
    assert tensor.name == 'full_tensor'

def test_full_tensor_constrcut():

    tensor = IRFullTensor(shape=[1024,1024], name='full_tensor')
    ctensor = copy.copy(tensor)
    assert isinstance(ctensor, IRFullTensor)

def test_full_tensor_select():

    tensor = IRFullTensor(shape=[1024,1024], name='tensor')
    assert len(tensor.segments()) == 0
    assert len(tensor.indices()) == 0
    assert len(tensor.val_maps()) == 0

    sub_tensor1 = tensor.select(
        indices = (slice(0, 1024), slice(0, 512)),
        val_map = None,
        shape = (1024, 512)
    )

    sub_tensor2 = tensor.select(
        indices = (slice(0, 1024), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )

    assert sub_tensor1.shape == (1024, 512)
    assert sub_tensor1.name == 'tensor'

    assert sub_tensor2.shape == (1024, 512)
    assert sub_tensor2.name == 'tensor'

    assert len(tensor.segments()) == 2
    assert len(tensor.indices()) == 2
    assert len(tensor.val_maps()) == 2


def test_full_tensor_overlap():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indices = (slice(0, 1024), slice(256, 1024)),
        val_map = None,
        shape = (1024, 768)
    )

    sub_tensor2 = tensor1.select(
        indices = (slice(0, 1024, 2), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )
    sub_tensor3 = tensor1.select(
        indices = (slice(1, 1024, 2), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )

    tensor2 = IRFullTensor(shape=[1024,1024], name='tensor')

    assert tensor1.overlap(sub_tensor1)
    assert tensor1.overlap(tensor1)
    assert not tensor1.overlap(tensor2)
    assert not tensor2.overlap(sub_tensor1)

    assert not sub_tensor2.overlap(sub_tensor3)


def test_sub_tensor_select():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indices = (slice(0, 1024), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )
    sub_tensor2 = sub_tensor1.select(
        indices = (slice(512, 1024), slice(0, 256)),
        val_map = None,
        shape = (512, 256)
    )
    sub_tensor3 = sub_tensor1.select(
        indices = (slice(512, 1024), slice(256, 512)),
        val_map = None,
        shape = (512, 256)
    )
    
    indices = sub_tensor2.indices.get()
    assert indices == (slice(512, 1024, 1), slice(512, 768, 1))
    indices = sub_tensor3.indices.get()
    assert indices == (slice(512, 1024, 1), slice(768, 1024, 1))

    assert len(tensor1.segments()) == 3
    assert sub_tensor1 in tensor1.segments()
    assert sub_tensor2 in tensor1.segments()
    assert sub_tensor3 in tensor1.segments()


def test_sub_tensor_overlap():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indices = (slice(0, 1024), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )
    sub_tensor2 = sub_tensor1.select(
        indices = (slice(512, 1024), slice(0, 256)),
        val_map = None,
        shape = (512, 256)
    )
    sub_tensor3 = sub_tensor1.select(
        indices = (slice(512, 1024), slice(256, 512)),
        val_map = None,
        shape = (512, 256)
    )

    assert sub_tensor1.overlap(sub_tensor2)
    assert sub_tensor1.overlap(sub_tensor3)
    assert not sub_tensor2.overlap(sub_tensor3)


def test_sub_tensor_common():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor_col1 = tensor1.select(
        indices = (slice(0, 1024), slice(0, 512)),
        val_map = None,
        shape = (1024, 512)
    )
    sub_tensor_col2 = tensor1.select(
        indices = (slice(0, 1024), slice(512, 1024)),
        val_map = None,
        shape = (1024, 512)
    )
    sub_tensor_row1 = tensor1.select(
        indices = (slice(0, 512), slice(0, 1024)),
        val_map = None,
        shape = (512, 1024)
    )
    sub_tensor_row2 = tensor1.select(
        indices = (slice(512, 1024), slice(0, 1024)),
        val_map = None,
        shape = (512, 1024)
    )

    lt = sub_tensor_col1.common(sub_tensor_row1)
    rt = sub_tensor_col2.common(sub_tensor_row1)
    lb = sub_tensor_row2.common(sub_tensor_col1)
    rb = sub_tensor_row2.common(sub_tensor_col2)

    assert lt.indices.get() == (slice(0, 512, 1), slice(0, 512, 1))
    assert rt.indices.get() == (slice(0, 512, 1), slice(512, 1024, 1))
    assert lb.indices.get() == (slice(512, 1024, 1), slice(0, 512, 1))
    assert rb.indices.get() == (slice(512, 1024, 1), slice(512, 1024, 1))
