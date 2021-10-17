import copy

from cube.ir.cten import IRTensor, IRCell


def test_tensor_init():

    tensor1 = IRTensor()
    tensor2 = IRTensor(shape=[1,2,3])
    tensor3 = IRTensor(shape=[1024], name='tensor')

    assert tensor1._id != tensor2._id
    assert tensor2._id != tensor3._id

    assert tensor1.shape is None
    assert tensor2.shape == [1,2,3]
    assert tensor3.shape == [1024,]

    assert tensor1.name is None
    assert tensor2.name is None
    assert tensor3.name == 'tensor'

    assert len(tensor1.device) == 0
    assert len(tensor2.device) == 0

    assert tensor1.requires_grad
    assert tensor2.requires_grad
    assert tensor3.requires_grad


def test_tensor_attach():

    tensor1 = IRTensor()
    tensor2 = IRTensor()
    cell = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )

    tensor1.attach_cell(cell)
    assert cell in tensor1._cell
    assert len(tensor1._cell) == 1
    assert len(tensor2._cell) == 0

    tensor1.detach_cell(cell)
    assert cell not in tensor1._cell
    assert len(tensor1._cell) == 0

    cell.set_input(0, tensor1)
    cell.set_output(0, tensor1)
    assert len(tensor1._cell) == 0
    assert len(cell.inputs(0)._cell) == 1


def test_tensor_renew():

    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell.set_input(0, tensor1)
    tensor1 = cell.inputs(0)

    tensor2 = tensor1.renew()
    assert tensor2.shape == tensor1.shape
    assert tensor2.name == tensor1.name
    assert tensor2 not in cell.inputs()
    assert len(tensor2._cell) == 0
    assert tensor2.requires_grad == tensor1.requires_grad


def test_tensor_copy():

    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    tensor1 = cell.set_input(0, tensor1)

    tensor2 = copy.copy(tensor1)
    assert tensor2 == tensor1
    assert len(tensor2._cell) == 0


def test_tensor_device():

    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell1 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell2 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    tensor1 = cell1.set_input(0, tensor1)
    tensor2 = cell2.set_input(0, tensor1)

    assert tensor1 == tensor2

    assert len(tensor1.device) == 0
    assert len(tensor2.device) == 0
    
    cell1.device = 2
    assert tensor1.device == [2]
    assert len(tensor2.device) == 0

    cell2.device = 3
    assert tensor1.device == [2]
    assert tensor2.device == [3]


def test_tensor_dst():
    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell1 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell2 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )

    cell1.set_input(0, tensor1)
    cells = tensor1.dst([cell1, cell2])
    assert set(cells) == set([cell1])

    cell2.set_input(0, tensor1)
    cells = tensor1.dst([cell1, cell2])
    assert set(cells) == set([cell1, cell2])


def test_tensor_src():
    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell1 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell2 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    
    cell1.set_output(0, tensor1)
    cells = tensor1.src([cell1, cell2])
    assert set(cells) == set([cell1])

    cell2.set_output(0, tensor1)
    cells = tensor1.src([cell1, cell2])
    assert set(cells) == set([cell1, cell2])


def test_tensor_is_leaf():
    tensor1 = IRTensor(shape=[1024], name='renew_tensor')
    cell1 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell2 = IRCell(
        name='cell',
        signature='any',
        input_length=3,
        output_length=1
    )
    cell1.set_input(0, tensor1)
    assert tensor1.is_leaf([cell1])
    
    cell2.set_input(0, cell1.outputs(0))
    assert cell2.outputs(0).is_leaf([cell1])
    assert not cell2.outputs(0).is_leaf([cell1, cell2])
