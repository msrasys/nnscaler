from cube.ir.cten import IRCell, IRTensor


def test_cell_init():

    cell = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    cell2 = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    assert cell2._id != cell._id

    assert len(cell.device) == 0
    assert cell.name == 'cell_test'
    assert cell.signature == 'torch.nn.functional.linear'
    assert len(cell.inputs()) == 3
    assert len(cell.outputs()) == 1
    assert len(cell.device) == 0


def test_cell_device():

    cell = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    assert len(cell.device) == 0
    cell.device = 2
    assert len(cell.device) == 1
    assert cell.device[0] == 2
    assert cell.on_device(2)
    assert not cell.on_device(3)

    cell.device = [2,3]
    assert len(cell.device) == 2
    assert set(cell.device) == set([2, 3])
    assert cell.on_device(2)
    assert cell.on_device(3)
    assert not cell.on_device(4)


def test_cell_inputs():

    cell = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    assert len(cell.inputs()) == 3
    for input in cell.inputs():
        assert input is None

    # the copy behavior
    inputs = cell.inputs()
    inputs[2] = 0
    assert cell.inputs(2) is None

    for idx in range(len(cell.inputs())):
        assert cell.inputs(idx) is None
        tensor = IRTensor(shape=[1024,], name='input')
        cell.set_input(idx, tensor)
        assert cell.inputs(idx) == tensor


def test_cell_outputs():

    cell = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    assert len(cell.outputs()) == 1
    for output in cell.outputs():
        assert isinstance(output, IRTensor)
    
    # the copy behavior
    outputs = cell.outputs()
    outputs[0] = 4
    assert cell.outputs(0) != 4

    for idx in range(len(cell.outputs())):
        output = cell.outputs(idx)
        tensor = IRTensor(shape=[1024,], name='output')
        cell.set_output(0, tensor)
        assert cell.outputs(0) == tensor
        assert cell.outputs(0) != output


def test_cell_predecessor():

    cell_prev = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    cell_post = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    assert len(cell_post.predecessors()) == 0
    assert len(cell_prev.predecessors()) == 0

    cell_post.add_predecessor(1, cell_prev)
    assert cell_prev in cell_post.predecessors()
    assert len(cell_post.predecessors()) == 1
    assert cell_prev in cell_post.predecessors(1)

    assert len(cell_post.successors()) == 0


def test_cell_successor():

    cell_prev = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    cell_post = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    assert len(cell_prev.successors()) == 0
    assert len(cell_post.successors()) == 0

    cell_prev.add_successor(0, cell_post)
    assert cell_post in cell_prev.successors()
    assert len(cell_prev.successors()) == 1
    assert cell_post in cell_prev.successors()
    
    assert len(cell_post.predecessors()) == 0


def test_cell_get_inputs_and_outputs():

    cell1 = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    input1 = IRTensor(shape=[1024, 1024])
    weight1 = IRTensor(shape=[1024, 1024])
    bias1 = IRTensor(shape=[1024,])

    cell1.set_input(0, input1)
    cell1.set_input(1, weight1)
    cell1.set_input(2, bias1)


    cell2 = IRCell(
        name='cell_test',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )

    input2 = IRTensor(shape=[1024, 1024])
    weight2 = IRTensor(shape=[1024, 1024])
    bias2 = IRTensor(shape=[1024,])

    cell2.set_input(0, input2)
    cell2.set_input(1, weight2)
    cell2.set_input(2, bias2)

    inputs = IRCell.get_inputs([cell1, cell2])
    assert len(inputs) == 6
    assert input1 in inputs
    assert weight1 in inputs
    assert bias1 in inputs
    assert input2 in inputs
    assert weight2 in inputs
    assert bias2 in inputs

    outputs = IRCell.get_outputs([cell1, cell2])
    assert len(outputs) == 2
    for output in cell1.outputs() + cell2.outputs():
        assert output in outputs

    # overlapped
    cell2.set_input(1, weight1)
    cell2.set_input(0, cell1.outputs(0))

    inputs = IRCell.get_inputs([cell1, cell2])
    assert len(inputs) == 5
    assert input1 in inputs
    assert weight1 in inputs
    assert bias1 in inputs
    assert bias2 in inputs

    outputs = IRCell.get_outputs([cell1, cell2])
    assert len(outputs) == 1
    assert cell2.outputs(0) in outputs
    assert cell1.outputs(0) not in outputs
