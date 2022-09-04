import torch
import math
"""
[bs, s, r, cm] -> [bs, s, r, cm]

used as column-wise gated self-attention
"""


def MSAAttention(x: torch.Tensor, gate_proj: torch.Tensor,
                 qkv_proj: torch.Tensor, out_proj: torch.Tensor, bias,
                 head: int, c: int, scale: float):
    bs, s, r, cm = x.size()

    gate = torch.sigmoid(torch.matmul(x, gate_proj))
    q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)

    gate = gate.reshape(bs, s, r, head,
                        c).transpose(2, 3).reshape(bs * s * head, r, c)
    q = q.reshape(bs, s, r, head, c).transpose(2,
                                               3).reshape(bs * s * head, r, c)
    k = k.reshape(bs, s, r, head, c).transpose(2,
                                               3).reshape(bs * s * head, r,
                                                          c).transpose(1, 2)
    v = v.reshape(bs, s, r, head, c).transpose(2,
                                               3).reshape(bs * s * head, r, c)

    sim = torch.bmm(q, k) * scale
    sim = torch.nn.functional.softmax(sim, dim=-1)

    if isinstance(bias, torch.Tensor):
        sim = sim.reshape(bs, s, head, r, r) + bias
        sim = sim.reshape(bs * s * head, r, r)

    attend = torch.bmm(sim, v) * gate

    out = attend.reshape(bs, s, head, r, c).transpose(2,
                                                      3).reshape(bs, s, r, cm)
    out = torch.matmul(out, out_proj)
    return out


"""
([bs, s, r, cm], [bs, r, r, cz]) -> [bs, s, r, cm]
"""


def MSARowAttentionWithPairBias(msa_repr: torch.Tensor,
                                pair_repr: torch.Tensor,
                                gate_proj: torch.Tensor,
                                qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                                bias_proj: torch.Tensor, head: int, c: int,
                                scale: float):
    bs, s, r, cm = msa_repr.size()

    bias = torch.matmul(pair_repr,
                        bias_proj).permute(0, 3, 1,
                                           2).reshape(bs, 1, head, r, r)

    return MSAAttention(msa_repr, gate_proj, qkv_proj, out_proj, bias, head, c,
                        scale)


"""
[bs, s, r, cm] -> [bs, s, r, cm]
"""


def MSATransition(msa_repr: torch.Tensor, proj1: torch.Tensor,
                  proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(msa_repr, proj1)), proj2)


"""
[bs, s, r, cm] -> [r, r, cz]
"""


def OuterProductMean(msa_repr: torch.Tensor, proj1: torch.Tensor,
                     proj2: torch.Tensor, out_proj: torch.Tensor):
    bs, s, r, cm = msa_repr.size()
    c = proj1.size(-1)

    a = torch.matmul(msa_repr, proj1).transpose(-2, -3)
    b = torch.matmul(msa_repr, proj2).transpose(-2, -3)

    outer = torch.einsum('...bac,...dae->...bdce', a,
                         b).reshape(bs, r, r, c * c)
    outer = torch.matmul(outer, out_proj)
    return outer


def TriangleMultiplication(
        pair_repr: torch.Tensor, tri_mul_norm1_weight: torch.Tensor,
        tri_mul_norm1_bias: torch.Tensor, tri_mul_norm2_weight: torch.Tensor,
        tri_mul_norm2_bias: torch.Tensor, tri_mul_proj1: torch.Tensor,
        tri_mul_proj2: torch.Tensor, tri_mul_proj3: torch.Tensor,
        tri_mul_proj4: torch.Tensor, tri_mul_proj5: torch.Tensor,
        tri_mul_proj6: torch.Tensor, cz: int, out_going: bool):
    pair_repr = torch.nn.functional.layer_norm(pair_repr, (cz, ),
                                               tri_mul_norm1_weight,
                                               tri_mul_norm1_bias)
    a = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj1))
    a = a * torch.matmul(pair_repr, tri_mul_proj2)
    b = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj3))
    b = b * torch.matmul(pair_repr, tri_mul_proj4)

    if out_going:
        a = a.permute(0, 3, 1, 2)
        b = b.permute(0, 3, 2, 1)
    else:
        a = a.permute(0, 3, 2, 1)
        b = b.permute(0, 3, 1, 2)

    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), tri_mul_norm2_weight,
                                       tri_mul_norm2_bias)
    p = torch.matmul(p, tri_mul_proj5)
    g = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj6))
    return p * g


def TriangleAttentionNode(pair_repr: torch.Tensor, gate_proj: torch.Tensor,
                          qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                          bias_proj: torch.Tensor, head: int, c: int,
                          scale: float):
    bias = torch.matmul(pair_repr, bias_proj).permute(0, 3, 1, 2)

    return MSAAttention(pair_repr, gate_proj, qkv_proj, out_proj, bias, head,
                        c, scale)


def PairTransition(pair_repr: torch.Tensor, proj1: torch.Tensor,
                   proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(pair_repr, proj1)), proj2)


"""
a simplified version for evoformer in alphafold2
  - dropout layers are omitted
  - masks are omitted
"""


class Evoformer(torch.nn.Module):

    def __init__(self,
                 s: int,
                 cm: int,
                 cz: int,
                 c=32,
                 msa_head=8,
                 pair_head=4,
                 c_tri_mult=128,
                 ff_mult=4):
        super().__init__()

        self.s, self.cm, self.cz, self.c, self.msa_head, self.pair_head, self.c_tri_mult, self.ff_mult = s, cm, cz, c, msa_head, pair_head, c_tri_mult, ff_mult
        self.scale = 1.0 / math.sqrt(c)

        # MSA row-wise gated self-attention with pair bias
        self.row_norm_m = torch.nn.LayerNorm(cm)
        self.row_norm_z = torch.nn.LayerNorm(cz)
        self.row_gate_proj = torch.nn.Parameter(torch.randn(cm, msa_head * c))
        self.row_qkv_proj = torch.nn.Parameter(
            torch.randn(cm, 3 * msa_head * c))
        self.row_out_proj = torch.nn.Parameter(torch.randn(msa_head * c, cm))
        self.row_bias_proj = torch.nn.Parameter(torch.randn(cz, msa_head))

        # MSA column-wise gated self-attention
        self.col_norm = torch.nn.LayerNorm(cm)
        self.col_gate_proj = torch.nn.Parameter(torch.randn(cm, msa_head * c))
        self.col_qkv_proj = torch.nn.Parameter(
            torch.randn(cm, 3 * msa_head * c))
        self.col_out_proj = torch.nn.Parameter(torch.randn(msa_head * c, cm))

        # MSA transition
        self.msa_transition_norm = torch.nn.LayerNorm(cm)
        self.msa_transition_proj1 = torch.nn.Parameter(
            torch.randn(cm, ff_mult * cm))
        self.msa_transition_proj2 = torch.nn.Parameter(
            torch.randn(ff_mult * cm, cm))

        # Outer product mean
        self.outer_norm = torch.nn.LayerNorm(cm)
        self.outer_proj1 = torch.nn.Parameter(torch.randn(cm, c))
        self.outer_proj2 = torch.nn.Parameter(torch.randn(cm, c))
        self.outer_out_proj = torch.nn.Parameter(torch.randn(c * c, cz))

        # Triangular multiplicative update using outgoing edges
        self.tri_mul_out_norm1_weight = torch.nn.Parameter(torch.empty(cz))
        self.tri_mul_out_norm1_bias = torch.nn.Parameter(torch.empty(cz))
        self.tri_mul_out_norm2_weight = torch.nn.Parameter(
            torch.empty(c_tri_mult))
        self.tri_mul_out_norm2_bias = torch.nn.Parameter(
            torch.empty(c_tri_mult))
        self.tri_mul_out_proj1 = torch.nn.Parameter(torch.randn(
            cz, c_tri_mult))
        self.tri_mul_out_proj2 = torch.nn.Parameter(torch.randn(
            cz, c_tri_mult))
        self.tri_mul_out_proj3 = torch.nn.Parameter(torch.randn(
            cz, c_tri_mult))
        self.tri_mul_out_proj4 = torch.nn.Parameter(torch.randn(
            cz, c_tri_mult))
        self.tri_mul_out_proj5 = torch.nn.Parameter(torch.randn(
            c_tri_mult, cz))
        self.tri_mul_out_proj6 = torch.nn.Parameter(torch.randn(cz, cz))

        # Triangular multiplicative update using incoming edges
        self.tri_mul_in_norm1_weight = torch.nn.Parameter(torch.empty(cz))
        self.tri_mul_in_norm1_bias = torch.nn.Parameter(torch.empty(cz))
        self.tri_mul_in_norm2_weight = torch.nn.Parameter(
            torch.empty(c_tri_mult))
        self.tri_mul_in_norm2_bias = torch.nn.Parameter(
            torch.empty(c_tri_mult))
        self.tri_mul_in_proj1 = torch.nn.Parameter(torch.randn(cz, c_tri_mult))
        self.tri_mul_in_proj2 = torch.nn.Parameter(torch.randn(cz, c_tri_mult))
        self.tri_mul_in_proj3 = torch.nn.Parameter(torch.randn(cz, c_tri_mult))
        self.tri_mul_in_proj4 = torch.nn.Parameter(torch.randn(cz, c_tri_mult))
        self.tri_mul_in_proj5 = torch.nn.Parameter(torch.randn(c_tri_mult, cz))
        self.tri_mul_in_proj6 = torch.nn.Parameter(torch.randn(cz, cz))

        # Triangular gated self-attention around starting node
        self.tri_att_start_norm = torch.nn.LayerNorm(cz)
        self.tri_att_start_gate_proj = torch.nn.Parameter(
            torch.randn(cz, pair_head * c))
        self.tri_att_start_qkv_proj = torch.nn.Parameter(
            torch.randn(cz, 3 * pair_head * c))
        self.tri_att_start_out_proj = torch.nn.Parameter(
            torch.randn(pair_head * c, cz))
        self.tri_att_start_bias_proj = torch.nn.Parameter(
            torch.randn(cz, pair_head))

        # Triangular gated self-attention around ending node
        self.tri_att_end_norm = torch.nn.LayerNorm(cz)
        self.tri_att_end_gate_proj = torch.nn.Parameter(
            torch.randn(cz, pair_head * c))
        self.tri_att_end_qkv_proj = torch.nn.Parameter(
            torch.randn(cz, 3 * pair_head * c))
        self.tri_att_end_out_proj = torch.nn.Parameter(
            torch.randn(pair_head * c, cz))
        self.tri_att_end_bias_proj = torch.nn.Parameter(
            torch.randn(cz, pair_head))

        # Transition in the pair stack
        self.pair_transition_norm = torch.nn.LayerNorm(cz)
        self.pair_transition_proj1 = torch.nn.Parameter(
            torch.randn(cz, ff_mult * cz))
        self.pair_transition_proj2 = torch.nn.Parameter(
            torch.randn(ff_mult * cz, cz))

    def forward(self, msa_repr: torch.Tensor, pair_repr: torch.Tensor):

        msa_repr = msa_repr + MSARowAttentionWithPairBias(
            self.row_norm_m(msa_repr), pair_repr, self.row_gate_proj,
            self.row_qkv_proj, self.row_out_proj, self.row_bias_proj,
            self.msa_head, self.c, self.scale)

        msa_repr = msa_repr.transpose(-3, -2)
        msa_repr = msa_repr + MSAAttention(
            self.col_norm(msa_repr), self.col_gate_proj, self.col_qkv_proj,
            self.col_out_proj, None, self.msa_head, self.c, self.scale)
        msa_repr = msa_repr.transpose(-3, -2)

        msa_repr = msa_repr + MSATransition(self.msa_transition_norm(msa_repr),
                                            self.msa_transition_proj1,
                                            self.msa_transition_proj2)

        pair_repr = pair_repr + OuterProductMean(
            self.outer_norm(msa_repr), self.outer_proj1, self.outer_proj2,
            self.outer_out_proj)

        pair_repr = pair_repr + TriangleMultiplication(
            pair_repr, self.tri_mul_out_norm1_weight,
            self.tri_mul_out_norm1_bias, self.tri_mul_out_norm2_weight,
            self.tri_mul_out_norm2_bias, self.tri_mul_out_proj1,
            self.tri_mul_out_proj2, self.tri_mul_out_proj3,
            self.tri_mul_out_proj4, self.tri_mul_out_proj5,
            self.tri_mul_out_proj6, self.cz, True)

        pair_repr = pair_repr + TriangleMultiplication(
            pair_repr, self.tri_mul_in_norm1_weight,
            self.tri_mul_in_norm1_bias, self.tri_mul_in_norm2_weight,
            self.tri_mul_in_norm2_bias, self.tri_mul_in_proj1,
            self.tri_mul_in_proj2, self.tri_mul_in_proj3,
            self.tri_mul_in_proj4, self.tri_mul_in_proj5,
            self.tri_mul_in_proj6, self.cz, False)

        pair_repr = pair_repr + TriangleAttentionNode(
            self.tri_att_start_norm(pair_repr), self.tri_att_start_gate_proj,
            self.tri_att_start_qkv_proj, self.tri_att_start_out_proj,
            self.tri_att_start_bias_proj, self.pair_head, self.c, self.scale)

        pair_repr = pair_repr.transpose(-3, -2)
        pair_repr = pair_repr + TriangleAttentionNode(
            self.tri_att_end_norm(pair_repr), self.tri_att_end_gate_proj,
            self.tri_att_end_qkv_proj, self.tri_att_end_out_proj,
            self.tri_att_end_bias_proj, self.pair_head, self.c, self.scale)
        pair_repr = pair_repr.transpose(-3, -2)

        pair_repr = pair_repr + PairTransition(
            self.pair_transition_norm(pair_repr), self.pair_transition_proj1,
            self.pair_transition_proj2)

        return msa_repr, pair_repr


def test():
    bs, s, r, cm, cz = 1, 128, 256, 256, 128
    model = Evoformer(s, cm, cz)

    msa = torch.randn(bs, s, r, cm)
    pair = torch.randn(bs, r, r, cz)

    new_msa, new_pair = model(msa, pair)


test()
