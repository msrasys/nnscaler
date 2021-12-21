import copy

from cube.graph.tensor import IRFullTensor, IRSubTensor, ValueMap


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
    assert len(tensor.indmap()) == 0
    assert len(tensor.val_maps()) == 0

    sub_tensor1 = tensor.select(
        indmap = (slice(0, 1024), slice(0, 512)),
        valmap = None,
        shape = (1024, 512)
    )

    sub_tensor2 = tensor.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )

    assert sub_tensor1.shape == (1024, 512)
    assert sub_tensor1.name == 'tensor'

    assert sub_tensor2.shape == (1024, 512)
    assert sub_tensor2.name == 'tensor'

    assert len(tensor.segments()) == 2
    assert len(tensor.indmap()) == 2
    assert len(tensor.val_maps()) == 2


def test_full_tensor_overlap():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indmap = (slice(0, 1024), slice(256, 1024)),
        valmap = None,
        shape = (1024, 768)
    )

    sub_tensor2 = tensor1.select(
        indmap = (slice(0, 1024, 2), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor3 = tensor1.select(
        indmap = (slice(1, 1024, 2), slice(512, 1024)),
        valmap = None,
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
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor2 = sub_tensor1.select(
        indmap = (slice(512, 1024), slice(0, 256)),
        valmap = None,
        shape = (512, 256)
    )
    sub_tensor3 = sub_tensor1.select(
        indmap = (slice(512, 1024), slice(256, 512)),
        valmap = None,
        shape = (512, 256)
    )
    
    indmap = sub_tensor2.indmap.get()
    assert indmap == (slice(512, 1024, 1), slice(512, 768, 1))
    indmap = sub_tensor3.indmap.get()
    assert indmap == (slice(512, 1024, 1), slice(768, 1024, 1))

    assert len(tensor1.segments()) == 3
    assert sub_tensor1 in tensor1.segments()
    assert sub_tensor2 in tensor1.segments()
    assert sub_tensor3 in tensor1.segments()


def test_sub_tensor_ind_overlap():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor2 = sub_tensor1.select(
        indmap = (slice(512, 1024), slice(0, 256)),
        valmap = None,
        shape = (512, 256)
    )
    sub_tensor3 = sub_tensor1.select(
        indmap = (slice(512, 1024), slice(256, 512)),
        valmap = None,
        shape = (512, 256)
    )

    assert sub_tensor1.overlap(sub_tensor2)
    assert sub_tensor1.overlap(sub_tensor3)
    assert not sub_tensor2.overlap(sub_tensor3)


def test_sub_tensor_val_overlap():
    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor2 = tensor1.select(
        indmap = (slice(0, 1024), slice(0, 512)),
        valmap = (0, 4),
        shape = (1024, 512)
    )
    sub_tensor3 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = (0, 4),
        shape = (1024, 512)
    )
    sub_tensor4 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = (1, 4),
        shape = (1024, 512)
    )

    assert not sub_tensor1.overlap(sub_tensor2)
    assert not sub_tensor2.overlap(sub_tensor3)
    assert sub_tensor1.overlap(sub_tensor3)
    assert sub_tensor1.overlap(sub_tensor4)
    assert sub_tensor4.overlap(sub_tensor1)
    assert not sub_tensor3.overlap(sub_tensor4)

def test_sub_tensor_common():

    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor_col1 = tensor1.select(
        indmap = (slice(0, 1024), slice(0, 512)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor_col2 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor_row1 = tensor1.select(
        indmap = (slice(0, 512), slice(0, 1024)),
        valmap = None,
        shape = (512, 1024)
    )
    sub_tensor_row2 = tensor1.select(
        indmap = (slice(512, 1024), slice(0, 1024)),
        valmap = None,
        shape = (512, 1024)
    )

    lt = sub_tensor_col1.common(sub_tensor_row1)
    rt = sub_tensor_col2.common(sub_tensor_row1)
    lb = sub_tensor_row2.common(sub_tensor_col1)
    rb = sub_tensor_row2.common(sub_tensor_col2)

    assert lt.indmap.get() == (slice(0, 512, 1), slice(0, 512, 1))
    assert rt.indmap.get() == (slice(0, 512, 1), slice(512, 1024, 1))
    assert lb.indmap.get() == (slice(512, 1024, 1), slice(0, 512, 1))
    assert rb.indmap.get() == (slice(512, 1024, 1), slice(512, 1024, 1))


def test_sub_tensor_as_grad():
    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )

    sub_tensor1.as_grad()
    assert sub_tensor1.is_grad()

    sub_tensor2 = tensor1.select(
        indmap = (slice(0, 1024), slice(0, 512)),
        valmap = (0, 4),
        shape = (1024, 512)
    )
    assert sub_tensor2.is_grad()


def test_sub_tensor_copy():
    tensor1 = IRFullTensor(shape=[1024,1024], name='tensor')
    sub_tensor1 = tensor1.select(
        indmap = (slice(0, 1024), slice(512, 1024)),
        valmap = None,
        shape = (1024, 512)
    )
    sub_tensor2 = tensor1.select(
        indmap = (slice(0, 1024), slice(0, 512)),
        valmap = (0, 4),
        shape = (1024, 512)
    )
    sub_tensor1.grads = [sub_tensor2]
    cpy_tensor = copy.copy(sub_tensor1)
    assert cpy_tensor.grads[0] == sub_tensor2
    
