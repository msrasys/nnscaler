import cube
import torch
from examples.openfold.blocks.utils import multi2ref


# @cube.graph.parser.register('N S R Z^, Z^ E, Z^ E -> N S R E')
# def tmu_projection(pair_repr: torch.Tensor, proj1: torch.Tensor, proj2: torch.Tensor):
#     x = torch.matmul(pair_repr, proj1)
#     x = torch.sigmoid(x)
#     x = x * torch.matmul(pair_repr, proj2)
# 
# 
# @cube.graph.parser.register('N S R Z+, Z+ E-> N S R E')
# def tmu_gate(pair_repr: torch.Tensor, proj: torch.Tensor):
#     return torch.sigmoid(torch.matmul(pair_repr, proj))


@cube.graph.parser.register('N S R Z^, Z^ E^, Z^ E^, Z^ E, Z^ E^, Z^ Z^ -> N S R E, N S R E^, N S R Z^', name='tmu_projection')
def tmu_projection(pair_repr: torch.Tensor, 
                   left1: torch.Tensor, left2: torch.Tensor,
                   right1: torch.Tensor, right2: torch.Tensor,
                   gate: torch.Tensor):
    # left
    left = torch.matmul(pair_repr, left1)
    left = torch.sigmoid(left)
    left = left * torch.matmul(pair_repr, left2)
    # right
    right = torch.matmul(pair_repr, right1)
    right = torch.sigmoid(right)
    right = right * torch.matmul(pair_repr, right2)
    # gate
    gate = torch.sigmoid(torch.matmul(pair_repr, gate))

    return left, right, gate


@cube.graph.parser.register('N S R^ E, N T^ R^ E^, N S^ T^ Z^, E^, E^, E^ Z^ -> N S T^ Z^', name='tmo')
def tmo(left: torch.Tensor, right: torch.Tensor, gate: torch.Tensor,
        norm_w: torch.Tensor, norm_b: torch.Tensor, out: torch.Tensor):
    a = left.permute(0, 3, 1, 2)
    b = right.permute(0, 3, 2, 1)
    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), norm_w, norm_b)
    p = torch.matmul(p, out)
    p = p * gate
    return p


@cube.graph.parser.register('N R^ S E, N R^ T^ E^, N T^ S^ Z^, E^, E^, E^ Z^ -> N T^ S Z^', name='tmi')
def tmi(left: torch.Tensor, right: torch.Tensor, gate: torch.Tensor,
        norm_w: torch.Tensor, norm_b: torch.Tensor, out: torch.Tensor):
    a = left.permute(0, 3, 2, 1)
    b = right.permute(0, 3, 1, 2)
    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), norm_w, norm_b)
    p = torch.matmul(p, out)
    p = p.permute(0, 2, 1, 3) * gate
    return p


class TriangleMultiplicativeUpdate(torch.nn.Module):

    def __init__(self, cz: int, mult: int, outgoing: bool) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm((cz,))

        self.left1 = torch.nn.Parameter(torch.empty(cz, mult))
        self.left2 = torch.nn.Parameter(torch.empty(cz, mult))
        self.right1 = torch.nn.Parameter(torch.empty(cz, mult))
        self.right2 = torch.nn.Parameter(torch.empty(cz, mult))
        
        # self.norm = torch.nn.LayerNorm(mult)
        self.normw = torch.nn.Parameter(torch.empty(mult))
        self.normb = torch.nn.Parameter(torch.empty(mult))

        self.out = torch.nn.Parameter(torch.empty(mult, cz))
        self.gate = torch.nn.Parameter(torch.empty(cz, cz))
        self.outgoing = outgoing
        
    def forward(self, pair_repr: torch.Tensor):
        """
        pair_repr: [N S R Z]
        """
        residual = pair_repr
        pair_repr = self.layer_norm(pair_repr)

        left, right, gate = tmu_projection(pair_repr,
            self.left1, self.left2, 
            self.right1, self.right2, self.gate
        )

        if self.outgoing:
            pair_repr = tmo(left, right, gate, self.normw, self.normb, self.out)
        else:
            pair_repr = tmi(left, right, gate, self.normw, self.normb, self.out)

        pair_repr = residual + pair_repr
        return pair_repr
