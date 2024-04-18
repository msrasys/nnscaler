import nnscaler
import torch


@nnscaler.graph.parser.register('* -> *, *', name='multi2ref')
def multi2ref(x: torch.Tensor):
    return (x, x)