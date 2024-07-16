from nnscaler.ir.tensor import IRSubTensor, IRFullTensor


def test_tensor_grad():

    ftensor = IRFullTensor((128, 512), requires_grad=True)
    subtensor = ftensor.tosub()

    assert isinstance(ftensor.grad, IRFullTensor)
    subtensor.grad = ftensor.grad.tosub()

    assert isinstance(subtensor.grad, IRSubTensor)

    ftensor.requires_grad = False
    assert ftensor.grad is None
    assert subtensor.grad is None
    assert subtensor.requires_grad is False
