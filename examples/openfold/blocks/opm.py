"""
Outer Product Mean module for Evoformer
"""

import cube
import torch
import torch.utils.checkpoint as ckpt


@cube.graph.parser.register('N S+ R^ M^, M^ c^, M^ c^, (c^ c^) cz^ -> N R^ R^ cz^', name='outer_prod_mean')
@torch.jit.ignore
def outer_prod_mean(msa_repr: torch.Tensor, left_proj: torch.Tensor, right_proj: torch.Tensor,
                    out_proj: torch.Tensor, chunk_size: int, training: bool):
    # cube.profiler.CudaTimer().start('opm')
    # N S R M, M c -> N S R c
    opm_left = torch.matmul(msa_repr, left_proj)
    # N S T M, M c -> N S T c
    opm_right = torch.matmul(msa_repr, right_proj)
    bs, s, r, c = opm_left.size()
    t = opm_right.size(2)

    # N S R M -> N R S M
    a = opm_left.transpose(-2, -3)
    # N S T M -> N T S M
    b = opm_right.transpose(-2, -3)

    if chunk_size == -1:
        # N R S M, N T S M -> N R T M M -> N R T (M M)
        outer = torch.einsum('...bac,...dae->...bdce', a,
                             b).reshape(bs, r, t, c * c)
        # N R T (M M), (M M) Z -> N R T Z
        outer = torch.matmul(outer, out_proj)
    else:
        out_chunks = []

        def opm(lhs: torch.Tensor, rhs: torch.Tensor, start: int):
            lhs_slice = lhs[:, start:start + chunk_size, :, :]
            out = torch.einsum('...bac,...dae->...bdce', lhs_slice,
                               rhs).reshape(bs, chunk_size, t, c * c)
            out = torch.matmul(out, out_proj)
            return out

        for start in range(0, r, chunk_size):
            ret = ckpt.checkpoint(opm, a, b, start)
            ret = opm(a, b, start)
            out_chunks.append(ret)
        outer = torch.cat(out_chunks, dim=1)
    # cube.profiler.CudaTimer().stop('opm')
    return outer


class OuterProducterMean(torch.nn.Module):

    def __init__(self, cm: int, c: int, cz: int, chunk_size: int) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.empty(cm, c))
        self.right = torch.nn.Parameter(torch.empty(cm, c))
        self.out = torch.nn.Parameter(torch.empty(c * c, cz))
        self.chunk_size = chunk_size

    def forward(self, msa_repr: torch.Tensor):
        """
        msa_repr: [N S R M]
        """
        return outer_prod_mean(
            msa_repr, self.left, self.right, self.out,
            self.chunk_size, self.training
        )
