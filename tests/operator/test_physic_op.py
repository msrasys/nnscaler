"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/operator/test_physic_op.py
"""

from cube.device.physic.group import DeviceGroup
from cube.operator.physic.generics import GenericPhysicOp, OpResult
import torch


def test_physic_generic_op():

    myrank = DeviceGroup().rank
    ranks = [0, 2]

    op = GenericPhysicOp(torch._C._nn.linear)
    assert op.placement is None
    
    op.placement = ranks
    assert op.func[0] is torch._C._nn.linear
    assert op.placement == [0, 2]
    assert op.execute_flag == (myrank in ranks)
    
    matA = torch.randn((1024,1024))
    matB = torch.randn((1024,1024))
    matC = op(matA, matB, bias=None)
    
    assert set(matC.placement) == set(ranks)
    if myrank in ranks:
        assert torch.is_tensor(matC.get_result())
    else:
        assert matC.get_result() is None

    matC_ref = torch._C._nn.linear(matA, matB, bias=None)
    if myrank in ranks:
        assert torch.allclose(matC.get_result(), matC_ref) is True


if __name__ == '__main__':
    test_physic_generic_op()
