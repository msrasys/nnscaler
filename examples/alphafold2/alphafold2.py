from audioop import mul
import torch
import math
import cube

from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from torch import nn
from torch.utils.checkpoint import checkpoint

import examples.alphafold2.policy.spmd as spmd

cube.init()
"""
[bs, s, r, cm] -> [bs, s, r, cm]

used as column-wise gated self-attention
"""


@cube.graph.parser.register('N S R^ M^, M^ E^, M^ F^, E^ M^ -> N S R M',
                            name='MSAAttention')
def MSAAttention(x: torch.Tensor, gate_proj: torch.Tensor,
                 qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
                 c: int, scale: float):
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

    attend = torch.bmm(sim, v) * gate

    out = attend.reshape(bs, s, head, r, c).transpose(2,
                                                      3).reshape(bs, s, r, cm)
    out = torch.matmul(out, out_proj)
    return out


@cube.graph.parser.register('N S R M, M E, M F, E M, N 1 8 R R -> N S R M',
                            name='MSAAttentionWithBias')
def MSAAttentionWithBias(x: torch.Tensor, gate_proj: torch.Tensor,
                         qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                         bias: torch.Tensor, head: int, c: int, scale: float):
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


# note: code not reused constrained by cube's interface
@cube.graph.parser.register('N S R M, N R R Z, M E, M F, E M, Z H -> N S R M',
                            name='MSARowAttentionWithPairBias')
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

    return MSAAttentionWithBias(msa_repr, gate_proj, qkv_proj, out_proj, bias,
                                head, c, scale)


@cube.graph.parser.register('N S R M, M E, M F, E M -> N S R M',
                            name='MSAColAttention')
def MSAColAttention(msa_repr: torch.Tensor, gate_proj: torch.Tensor,
                    qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
                    c: int, scale: float):
    return MSAAttention(msa_repr.permute(0, 2, 1, 3), gate_proj, qkv_proj,
                        out_proj, head, c, scale).permute(0, 2, 1, 3)


"""
[bs, s, r, cm] -> [bs, s, r, cm]
"""


@cube.graph.parser.register('N S R M, M E, E M -> N S R M',
                            name='MSATransition')
def MSATransition(msa_repr: torch.Tensor, proj1: torch.Tensor,
                  proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(msa_repr, proj1)), proj2)


"""
[bs, s, r, cm] -> [bs, r, r, cz]
"""


@cube.graph.parser.register('N S R M, N S T M, M C, M C, F Z -> N R T Z',
                            name='OuterProductMean')
def OuterProductMean(msa_repr: torch.Tensor, dummy_msa_repr: torch.Tensor,
                     proj1: torch.Tensor, proj2: torch.Tensor,
                     out_proj: torch.Tensor):
    bs, s, r, cm = msa_repr.size()
    t = dummy_msa_repr.size(2)
    c = proj1.size(-1)

    a = torch.matmul(msa_repr, proj1).transpose(-2, -3)
    b = torch.matmul(dummy_msa_repr, proj2).transpose(-2, -3)

    outer = torch.einsum('...bac,...dae->...bdce', a,
                         b).reshape(bs, r, t, c * c)
    outer = torch.matmul(outer, out_proj)
    return outer


@cube.graph.parser.register(
    'N S R Z, N T R Z, Z, Z, E, E, Z E, Z E, Z E, Z E, E Z, Z Z -> N S T Z',
    name='TriangleMultiplicationOut')
def TriangleMultiplicationOut(
        pair_repr: torch.Tensor, dummy_pair_repr: torch.Tensor,
        tri_mul_norm1_weight: torch.Tensor, tri_mul_norm1_bias: torch.Tensor,
        tri_mul_norm2_weight: torch.Tensor, tri_mul_norm2_bias: torch.Tensor,
        tri_mul_proj1: torch.Tensor, tri_mul_proj2: torch.Tensor,
        tri_mul_proj3: torch.Tensor, tri_mul_proj4: torch.Tensor,
        tri_mul_proj5: torch.Tensor, tri_mul_proj6: torch.Tensor, cz: int):
    pair_repr = torch.nn.functional.layer_norm(pair_repr, (cz, ),
                                               tri_mul_norm1_weight,
                                               tri_mul_norm1_bias)
    dummy_pair_repr = torch.nn.functional.layer_norm(dummy_pair_repr, (cz, ),
                                                     tri_mul_norm1_weight,
                                                     tri_mul_norm1_bias)
    a = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj1))
    a = a * torch.matmul(pair_repr, tri_mul_proj2)
    b = torch.sigmoid(torch.matmul(dummy_pair_repr, tri_mul_proj3))
    b = b * torch.matmul(dummy_pair_repr, tri_mul_proj4)

    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 2, 1)

    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), tri_mul_norm2_weight,
                                       tri_mul_norm2_bias)
    p = torch.matmul(p, tri_mul_proj5)
    g = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj6))
    return p * g


@cube.graph.parser.register(
    'N R S Z, N R T Z, Z, Z, E, E, Z E, Z E, Z E, Z E, E Z, Z Z -> N S T Z',
    name='TriangleMultiplicationIn')
def TriangleMultiplicationIn(
        pair_repr: torch.Tensor, dummy_pair_repr: torch.Tensor,
        tri_mul_norm1_weight: torch.Tensor, tri_mul_norm1_bias: torch.Tensor,
        tri_mul_norm2_weight: torch.Tensor, tri_mul_norm2_bias: torch.Tensor,
        tri_mul_proj1: torch.Tensor, tri_mul_proj2: torch.Tensor,
        tri_mul_proj3: torch.Tensor, tri_mul_proj4: torch.Tensor,
        tri_mul_proj5: torch.Tensor, tri_mul_proj6: torch.Tensor, cz: int):
    pair_repr = torch.nn.functional.layer_norm(pair_repr, (cz, ),
                                               tri_mul_norm1_weight,
                                               tri_mul_norm1_bias)
    dummy_pair_repr = torch.nn.functional.layer_norm(dummy_pair_repr, (cz, ),
                                                     tri_mul_norm1_weight,
                                                     tri_mul_norm1_bias)
    a = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj1))
    a = a * torch.matmul(pair_repr, tri_mul_proj2)
    b = torch.sigmoid(torch.matmul(dummy_pair_repr, tri_mul_proj3))
    b = b * torch.matmul(dummy_pair_repr, tri_mul_proj4)

    a = a.permute(0, 3, 2, 1)
    b = b.permute(0, 3, 1, 2)

    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), tri_mul_norm2_weight,
                                       tri_mul_norm2_bias)
    p = torch.matmul(p, tri_mul_proj5)
    g = torch.sigmoid(torch.matmul(pair_repr, tri_mul_proj6)).transpose(1, 2)
    return p * g


@cube.graph.parser.register('N S R Z, Z E, Z F, E Z, Z G -> N S R Z',
                            name='TriangleAttentionNodeStart')
def TriangleAttentionNodeStart(pair_repr: torch.Tensor,
                               gate_proj: torch.Tensor, qkv_proj: torch.Tensor,
                               out_proj: torch.Tensor, bias_proj: torch.Tensor,
                               head: int, c: int, scale: float):
    bias = torch.matmul(pair_repr, bias_proj).permute(0, 1, 3, 2).unsqueeze(3)

    return MSAAttentionWithBias(pair_repr, gate_proj, qkv_proj, out_proj, bias,
                                head, c, scale)


@cube.graph.parser.register('N S R Z, Z E, Z F, E Z, Z G -> N S R Z',
                            name='TriangleAttentionNodeEnd')
def TriangleAttentionNodeEnd(pair_repr: torch.Tensor, gate_proj: torch.Tensor,
                             qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                             bias_proj: torch.Tensor, head: int, c: int,
                             scale: float):
    pair_repr = pair_repr.permute(0, 2, 1, 3)
    out = TriangleAttentionNodeStart(pair_repr, gate_proj, qkv_proj, out_proj,
                                     bias_proj, head, c, scale)
    return out.permute(0, 2, 1, 3)


@cube.graph.parser.register('N R T Z, Z E, E Z -> N R T Z',
                            name='PairTransition')
def PairTransition(pair_repr: torch.Tensor, proj1: torch.Tensor,
                   proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(pair_repr, proj1)), proj2)


@cube.graph.parser.register('* -> *, *', name='multi2ref')
def multi2ref(x: torch.Tensor):
    return (x, x)


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

    def forward(self, msa_repr, pair_repr):

        msa_repr = msa_repr + MSARowAttentionWithPairBias(
            self.row_norm_m(msa_repr), pair_repr, self.row_gate_proj,
            self.row_qkv_proj, self.row_out_proj, self.row_bias_proj,
            self.msa_head, self.c, self.scale)

        msa_repr = msa_repr + MSAColAttention(
            self.col_norm(msa_repr), self.col_gate_proj, self.col_qkv_proj,
            self.col_out_proj, self.msa_head, self.c, self.scale)

        msa_repr = msa_repr + MSATransition(self.msa_transition_norm(msa_repr),
                                            self.msa_transition_proj1,
                                            self.msa_transition_proj2)
        succ_msa_repr, msa_repr = multi2ref(msa_repr)

        pair_msa_repr, dummy_pair_msa_repr = multi2ref(
            self.outer_norm(msa_repr))
        pair_repr = pair_repr + OuterProductMean(
            pair_msa_repr, dummy_pair_msa_repr, self.outer_proj1,
            self.outer_proj2, self.outer_out_proj)

        out_pair_repr, out_dummy_pair_repr = multi2ref(pair_repr)
        pair_repr = out_pair_repr + TriangleMultiplicationOut(
            out_pair_repr, out_dummy_pair_repr, self.tri_mul_out_norm1_weight,
            self.tri_mul_out_norm1_bias, self.tri_mul_out_norm2_weight,
            self.tri_mul_out_norm2_bias, self.tri_mul_out_proj1,
            self.tri_mul_out_proj2, self.tri_mul_out_proj3,
            self.tri_mul_out_proj4, self.tri_mul_out_proj5,
            self.tri_mul_out_proj6, self.cz)

        in_pair_repr, in_dummy_pair_repr = multi2ref(pair_repr)
        pair_repr = in_pair_repr + TriangleMultiplicationIn(
            in_pair_repr, in_dummy_pair_repr, self.tri_mul_in_norm1_weight,
            self.tri_mul_in_norm1_bias, self.tri_mul_in_norm2_weight,
            self.tri_mul_in_norm2_bias, self.tri_mul_in_proj1,
            self.tri_mul_in_proj2, self.tri_mul_in_proj3,
            self.tri_mul_in_proj4, self.tri_mul_in_proj5,
            self.tri_mul_in_proj6, self.cz)

        pair_repr = pair_repr + TriangleAttentionNodeStart(
            self.tri_att_start_norm(pair_repr), self.tri_att_start_gate_proj,
            self.tri_att_start_qkv_proj, self.tri_att_start_out_proj,
            self.tri_att_start_bias_proj, self.pair_head, self.c, self.scale)

        pair_repr = pair_repr + TriangleAttentionNodeEnd(
            self.tri_att_end_norm(pair_repr), self.tri_att_end_gate_proj,
            self.tri_att_end_qkv_proj, self.tri_att_end_out_proj,
            self.tri_att_end_bias_proj, self.pair_head, self.c, self.scale)

        pair_repr = pair_repr + PairTransition(
            self.pair_transition_norm(pair_repr), self.pair_transition_proj1,
            self.pair_transition_proj2)

        return (succ_msa_repr, pair_repr)


class AlphaFold2(nn.Module):

    def __init__(self, s: int, cm: int, cz: int, evo_num: int):
        super().__init__()
        self.evo_num = evo_num
        self.evoformer = Evoformer(s, cm, cz)

    def forward(self, msa, pair):
        new_msa, new_pair = self.evoformer(msa, pair)
        loss = torch.sum(new_msa) * torch.sum(new_pair)
        return loss


def test():
    # bs, s, r, cm, cz = 1, 128, 256, 256, 128
    bs, s, r, cm, cz = 1, 512, 384, 256, 128

    model = AlphaFold2(s, cm, cz, 1)

    # msa_repr, pair_repr = torch.randn(bs, s, r, cm), torch.randn(bs, r, r, cz)
    # x = model(msa_repr, pair_repr)
    # return

    model = cube.SemanticModel(
        model,
        input_shapes=(
            [bs, s, r, cm],
            [bs, r, r, cz],
        ),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(shapes=(
        [bs, s, r, cm],
        [bs, r, r, cz],
    ),
                                                    dtypes=(
                                                        torch.float32,
                                                        torch.float32,
                                                    ),
                                                    batch_dims=(
                                                        0,
                                                        0,
                                                    ))

    # @cube.compile(model, dataloader, PAS=spmd.PASSingle)
    # @cube.compile(model, dataloader, PAS=spmd.PASData)
    @cube.compile(model, dataloader, PAS=spmd.PASMegatron, override=True)
    def train_iter(model, dataloader):
        msa_repr, pair_repr = next(dataloader)
        loss = model(msa_repr, pair_repr)
        loss.backward()

    model = model.get_gen_module()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    warm_up = 20
    iter_num = 64
    CudaTimer(enable=False).warmup()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    for i in range(iter_num):
        if i >= warm_up:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if i >= warm_up:
            CudaTimer().stop('e2e')
        if i > 0 and (i + 1) % 20 == 0:
            print_each_rank(f'iter [{i + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num - warm_up, field_name='e2e')))
    CudaTimer().print_all(times=iter_num - warm_up)
    print_each_rank('memory consumption: {} MB'.format(
        int(torch.cuda.max_memory_allocated() / 1024 / 1024)))


test()
