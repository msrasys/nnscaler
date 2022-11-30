import cube
import torch
import torch.utils.checkpoint as ckpt


@cube.graph.parser.register('N S R Z, Z E, Z E, Z E, Z E, Z Z, E, E, E Z -> N S R Z')
@torch.jit.ignore
def tmu(pair_repr: torch.Tensor, 
        left1: torch.Tensor, left2: torch.Tensor,
        right1: torch.Tensor, right2: torch.Tensor,
        gate: torch.Tensor, 
        norm_weight: torch.Tensor, norm_bias: torch.Tensor,
        out: torch.Tensor, outgoing: bool) -> torch.Tensor:
    # cube.profiler.CudaTimer().start('tmu')
    # Note S == R
    # left projection: N S R Z^, Z^ E, Z^ E -> N S R E
    left = torch.matmul(pair_repr, left1)
    left = torch.sigmoid(left)
    left = left * torch.matmul(pair_repr, left2)
    # right projection: N S R Z^, Z^ E, Z^ E -> N S R E
    right = torch.matmul(pair_repr, right1)
    right = torch.sigmoid(right)
    right = right * torch.matmul(pair_repr, right2)
    if outgoing:
        # N S R E -> N E S R
        left = left.permute(0, 3, 1, 2)
        # N S R E -> N E R S
        right = right.permute(0, 3, 2, 1)
    else:
        # N S R E -> N E R S
        left = left.permute(0, 3, 2, 1)
        # N S R E -> N E S R
        right = right.permute(0, 3, 1, 2)
    # N E S R+, N E R+ S -> N E S S -> N S S E (for out)
    # N E R S, N E S R -> N E R R -> N R R E (for in)
    p = torch.matmul(left, right).permute(0, 2, 3, 1)
    e = p.size(3)
    # N S S E^ -> N S S E^
    p = torch.nn.functional.layer_norm(p, (e,), norm_weight, norm_bias)
    # N S S E+, E+ Z -> N S S Z
    p = torch.matmul(p, out)
    if not outgoing:
        p = p.permute(0, 2, 1, 3)
    # gate: N S R Z+, Z+ Z -> N S R Z
    g = torch.matmul(pair_repr, gate)
    g = torch.sigmoid(g)
    # N S S Z, N S R Z -> N S R Z (broadcast R == S == 0)
    p = p * g
    # cube.profiler.CudaTimer().stop('tmu')
    return p


class TriangleMultiplicativeUpdate(torch.nn.Module):

    def __init__(self, cz: int, mult: int, outgoing: bool) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm((cz,))

        self.left1 = torch.nn.Parameter(torch.empty(cz, mult))
        self.left2 = torch.nn.Parameter(torch.empty(cz, mult))
        self.right1 = torch.nn.Parameter(torch.empty(cz, mult))
        self.right2 = torch.nn.Parameter(torch.empty(cz, mult))
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
        pair_repr = tmu(pair_repr, 
            self.left1, self.left2, self.right1, self.right2,
            self.gate, self.normw, self.normb, self.out, self.outgoing)
        pair_repr = residual + pair_repr
        return pair_repr
