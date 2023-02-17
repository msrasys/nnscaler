import cube
import torch
import math
from torch import nn

from examples.alphafold2.module import *
from dataclasses import dataclass
"""
a simplified version for evoformer in alphafold2
  - dropout layers are omitted
  - masks are omitted
"""
@dataclass
class Config:
    bs: int = 1
    s: int = 128
    r: int = 256
    cm: int = 256
    cz: int = 128
    dtype = torch.float16
    evo_num: int = 4
    use_chunk: bool = False
    is_train : bool = True
    is_extra : bool = False

class Evoformer(torch.nn.Module):

    def __init__(self,
                 s: int,
                 cm: int,
                 cz: int,
                 use_chunk=False,
                 is_extra=False,
                 is_train=True,
                 c=32,
                 msa_head=8,
                 pair_head=4,
                 c_tri_mult=128,
                 ff_mult=4):
        super().__init__()

        self.s, self.cm, self.cz, self.c = s, cm, cz, c
        self.msa_head, self.pair_head = msa_head, pair_head
        self.c_tri_mult, self.ff_mult = c_tri_mult, ff_mult
        self.scale = 1.0 / math.sqrt(c)

        self.is_extra = is_extra
        self.is_train = is_train

        if use_chunk:
            if is_extra:
                self.msa_row_chunk, self.msa_col_chunk = 4, -1
                self.opm_chunk, self.tans_chunk, self.tane_chunk = -1, -1, -1
            else:
                self.msa_row_chunk, self.msa_col_chunk = 4, 256
                self.opm_chunk, self.tans_chunk, self.tane_chunk = 4, 4, 4
        else:
            self.msa_row_chunk, self.msa_col_chunk = -1, -1
            self.opm_chunk, self.tans_chunk, self.tane_chunk = -1, -1, -1

        # MSA row-wise gated self-attention with pair bias
        self.row_norm_m = torch.nn.LayerNorm(cm)
        self.row_norm_z = torch.nn.LayerNorm(cz)
        self.row_gate_proj = torch.nn.Parameter(torch.randn(cm, cm))
        self.row_qkv_proj = torch.nn.Parameter(torch.randn(cm, 3 * cm))
        self.row_out_proj = torch.nn.Parameter(torch.randn(cm, cm))
        self.row_bias_proj = torch.nn.Parameter(torch.randn(cz, msa_head))

        # MSA column-wise gated self-attention
        self.col_norm = torch.nn.LayerNorm(cm)
        self.col_gate_proj = torch.nn.Parameter(torch.randn(cm, cm))
        # TODO: fix me
        self.col_q_proj = torch.nn.Parameter(torch.randn(cm, cm))
        self.col_k_proj = torch.nn.Parameter(torch.randn(cm, 8))
        self.col_v_proj = torch.nn.Parameter(torch.randn(cm, 8))
        self.col_qkv_proj = torch.nn.Parameter(
            torch.randn(cm, 3 * msa_head * c))
        self.col_out_proj = torch.nn.Parameter(torch.randn(cm, cm))

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
        self.tri_mul_out_norm1 = torch.nn.LayerNorm(cz)
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
        self.tri_mul_in_norm1 = torch.nn.LayerNorm(cz)
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
        cube.runtime.function.anchor('MSARow')

        pair_repr, dummy_pair_repr = multi2ref(pair_repr)
        msa_repr = msa_repr + MSARowAttentionWithPairBias(
            self.row_norm_m(msa_repr), dummy_pair_repr, self.row_gate_proj,
            self.row_qkv_proj, self.row_out_proj, self.row_bias_proj,
            self.msa_head, self.c, self.scale, self.msa_row_chunk,
            self.is_train)

        cube.runtime.function.anchor('MSACol')
        if self.is_extra:
            msa_repr = msa_repr + MSAColGlobalAttention(
                self.col_norm(msa_repr), self.col_q_proj, self.col_k_proj,
                self.col_v_proj, self.col_gate_proj, self.col_out_proj,
                self.msa_head, self.c, self.scale)
        else:
            msa_repr = msa_repr + MSAColAttention(
                self.col_norm(msa_repr), self.col_gate_proj, self.col_qkv_proj,
                self.col_out_proj, self.msa_head, self.c, self.scale,
                self.msa_col_chunk, self.is_train)

        cube.runtime.function.anchor('MSATrans')
        msa_repr = msa_repr + MSATransition(self.msa_transition_norm(msa_repr),
                                            self.msa_transition_proj1,
                                            self.msa_transition_proj2)
        succ_msa_repr, msa_repr = multi2ref(msa_repr)

        cube.runtime.function.anchor('OPM')
        msa_repr = self.outer_norm(msa_repr)
        opm_left, opm_right = OPMLeftProj(msa_repr,
                                          self.outer_proj1), OPMRightProj(
                                              msa_repr, self.outer_proj2)
        pair_repr = pair_repr + OuterProductMean(
            opm_left, opm_right, self.outer_out_proj, self.opm_chunk,
            self.is_train)

        cube.runtime.function.anchor('TMO')
        pair_repr = self.tri_mul_out_norm1(pair_repr)
        tmo_left, tmo_right = TMOLeftProj(
            pair_repr, self.tri_mul_out_proj1,
            self.tri_mul_out_proj2), TMORightProj(pair_repr,
                                                  self.tri_mul_out_proj3,
                                                  self.tri_mul_out_proj4)
        tmo_g = TMOGate(pair_repr, self.tri_mul_out_proj6)
        pair_repr = pair_repr + TriangleMultiplicationOut(
            tmo_left, tmo_right, tmo_g, self.tri_mul_out_norm2_weight,
            self.tri_mul_out_norm2_bias, self.tri_mul_out_proj5, self.cz)

        cube.runtime.function.anchor('TMI')
        pair_repr = self.tri_mul_in_norm1(pair_repr)
        tmi_left = TMILeftProj(pair_repr, self.tri_mul_in_proj1,
                               self.tri_mul_in_proj2)
        tmi_right = TMIRightProj(pair_repr, self.tri_mul_in_proj3,
                                 self.tri_mul_in_proj4)
        tmi_gate = TMIGate(pair_repr, self.tri_mul_in_proj6)
        pair_repr = pair_repr + TriangleMultiplicationIn(
            tmi_left, tmi_right, tmi_gate, self.tri_mul_in_norm2_weight,
            self.tri_mul_in_norm2_bias, self.tri_mul_in_proj5, self.cz)

        cube.runtime.function.anchor('TANS')
        pair_repr = self.tri_att_start_norm(pair_repr)
        bias = TANSBias(pair_repr, self.tri_att_start_bias_proj)
        pair_repr = pair_repr + TriangleAttentionNodeStart(
            pair_repr, self.tri_att_start_gate_proj,
            self.tri_att_start_qkv_proj, self.tri_att_start_out_proj, bias,
            self.pair_head, self.c, self.scale, self.tans_chunk, self.is_train)

        cube.runtime.function.anchor('TANE')
        pair_repr = self.tri_att_end_norm(pair_repr)
        bias = TANEBias(pair_repr, self.tri_att_end_bias_proj)
        pair_repr = pair_repr + TriangleAttentionNodeEnd(
            pair_repr, self.tri_att_end_gate_proj, self.tri_att_end_qkv_proj,
            self.tri_att_end_out_proj, bias, self.pair_head, self.c,
            self.scale, self.tane_chunk, self.is_train)

        cube.runtime.function.anchor('PairTrans')
        pair_repr = pair_repr + PairTransition(
            self.pair_transition_norm(pair_repr), self.pair_transition_proj1,
            self.pair_transition_proj2)

        return succ_msa_repr, pair_repr


class AlphaFold2(nn.Module):

    def __init__(self,
                 s: int,
                 cm: int,
                 cz: int,
                 evo_num: int,
                 use_chunk=False,
                 is_extra=False,
                 is_train=True):
        super().__init__()
        self.evo_num = evo_num
        # add norm to work with PyTorch's recompute mechanism
        self.msa_norm = torch.nn.LayerNorm(cm)
        self.pair_norm = torch.nn.LayerNorm(cz)
        self.evoformers = torch.nn.ModuleList([
            Evoformer(s,
                      cm,
                      cz,
                      use_chunk=use_chunk,
                      is_extra=is_extra,
                      is_train=is_train) for _ in range(evo_num)
        ])

    def forward(self, msa, pair):
        msa = self.msa_norm(msa)
        pair = self.pair_norm(pair)

        cube.runtime.function.anchor('Evoformer Stack Start')
        for evoformer in self.evoformers:
            cube.runtime.function.anchor('One Layer Evoformer Start')
            msa, pair = evoformer(msa, pair)
            cube.runtime.function.anchor('One Layer Evoformer End')
        cube.runtime.function.anchor('Evoformer Stack End')
        loss = torch.sum(msa) * torch.sum(pair)
        return loss

class AlphaFoldlrw(nn.Module):

    def __init__(self, cfg=Config()):
        super().__init__()
        self.evo_num = cfg.evo_num
        # add norm to work with PyTorch's recompute mechanism
        self.msa_norm = torch.nn.LayerNorm(cfg.cm)
        self.pair_norm = torch.nn.LayerNorm(cfg.cz)
        self.evoformers = torch.nn.ModuleList([
            Evoformer(cfg.s,
                      cfg.cm,
                      cfg.cz,
                      use_chunk=cfg.use_chunk,
                      is_extra=cfg.is_extra,
                      is_train=cfg.is_train) for _ in range(cfg.evo_num)
        ])

    def forward(self, msa, pair):
        msa = self.msa_norm(msa)
        pair = self.pair_norm(pair)

        cube.runtime.function.anchor('Evoformer Stack Start')
        for evoformer in self.evoformers:
            cube.runtime.function.anchor('One Layer Evoformer Start')
            msa, pair = evoformer(msa, pair)
            cube.runtime.function.anchor('One Layer Evoformer End')
        cube.runtime.function.anchor('Evoformer Stack End')
        loss = torch.sum(msa) * torch.sum(pair)
        return loss