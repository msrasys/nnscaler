import nnscaler
import torch


@nnscaler.register_op('* -> *, *', name='multi2ref')
def multi2ref(x: torch.Tensor):
    return (x, x)