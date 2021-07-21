import torch
from cube.tensor.physic.tensor import PhysicTensor

def test_type():
    tensor1 = PhysicTensor([1,2,3,4])
    tensor2 = PhysicTensor([2,3,4,5])
    tensor_out = tensor1 + tensor2
    assert isinstance(tensor_out, PhysicTensor)


def test_data_host_device():
    tensor = PhysicTensor([1,2,3,4])
    assert tensor.data_host_device == torch.device('cpu')
    tensor.data_host_device = torch.device('cuda:0')
    assert tensor.device == torch.device('cuda:0')
    tensor.move_(torch.device('cpu'))
    assert tensor.device == torch.device('cpu')


if __name__ == '__main__':
    
    test_type()
    test_data_host_device()

    print('test passed')