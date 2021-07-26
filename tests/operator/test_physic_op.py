from cube.operator.physic.generics import GenericPhysicOp
import torch


def test_physic_generic_op():

    op = GenericPhysicOp(torch._C._nn.linear)
    assert op.placement is None

    op.set_placement(torch.device('cuda:0'))
    
    matA = torch.randn((1024,1024))
    matB = torch.randn((1024,1024))
    matC = op(matA, matB, bias=None)
    assert matC.device == torch.device('cuda:0')

    matC_ref = torch._C._nn.linear(matA, matB, bias=None)
    assert torch.allclose(matC, matC_ref) is True


if __name__ == '__main__':
    test_physic_generic_op()
