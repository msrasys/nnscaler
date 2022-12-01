import torch
from examples.openfold.blocks.attention import MSARowAttention, MSAColAttention, Transition, TriangleAttentionNodeStart, TriangleAttentionNodeEnd
from examples.openfold.blocks.tmu import TriangleMultiplicativeUpdate
from examples.openfold.blocks.opm import OuterProducterMean
from examples.openfold.blocks.utils import multi2ref

import math
import cube



class Evoformer(torch.nn.Module):
    """
    Simulate execution of evoformer in alphafold.

    The mask and dropout is ommited for simplicity.
    """

    def __init__(self, s: int, cm: int, cz: int,
                 use_chunk=False, is_train=True,
                 c=32, msa_head=8, pair_head=4,
                 c_tri_mult=128, ff_mult=4):
        super().__init__()

        self.s, self.cm, self.cz, self.c = s, cm, cz, c
        self.msa_head, self.pair_head = msa_head, pair_head
        self.c_tri_mult, self.ff_mult = c_tri_mult, ff_mult
        self.scale = 1.0 / math.sqrt(c)

        self.is_train = is_train

        self.msa_row_chunk = 4 if use_chunk else -1
        self.msa_col_chunk = -1
        self.opm_chunk = self.tans_chunk = self.tane_chunk = -1

        # MSA row-wise gated self-attention with pair bias
        self.row_norm_m = torch.nn.LayerNorm(cm)
        self.row_norm_z = torch.nn.LayerNorm(cz)
        self.row_attn = MSARowAttention(cm, msa_head, cz, self.scale, self.msa_row_chunk)

        # MSA column-wise gated self-attention
        self.col_norm = torch.nn.LayerNorm(cm)
        self.col_attn = MSAColAttention(cm, msa_head, self.scale, self.msa_col_chunk)
        
        # MSA transition
        self.msa_transition_norm = torch.nn.LayerNorm(cm)
        self.msa_transition = Transition(cm, ff_mult)
        
        # Outer product mean
        self.outer_norm = torch.nn.LayerNorm(cm)
        self.outer_prod_mean = OuterProducterMean(cm, c, cz, self.opm_chunk)

        # Triangular multiplicative update using outgoing edges
        self.tmo = TriangleMultiplicativeUpdate(cz, c_tri_mult, outgoing=True)

        # Triangular multiplicative update using incoming edges
        self.tmi = TriangleMultiplicativeUpdate(cz, c_tri_mult, outgoing=False)

        # Triangular gated self-attention around starting node
        self.tri_attn_node_start = TriangleAttentionNodeStart(cz, pair_head, c, self.scale, self.tans_chunk)

        # Triangular gated self-attention around ending node
        self.tri_attn_node_end = TriangleAttentionNodeEnd(cz, pair_head, c, self.scale, self.tane_chunk)
        
        # Transition in the pair stack
        self.pair_transition_norm = torch.nn.LayerNorm(cz)
        self.pair_transition = Transition(cz, ff_mult)

    def forward(self, msa_repr, pair_repr):

        cube.runtime.function.anchor('MSARow')
        pair_repr, dummy_pair_repr = multi2ref(pair_repr)
        residual = msa_repr
        msa_repr = self.row_norm_m(msa_repr)
        dummy_pair_repr = self.row_norm_z(dummy_pair_repr)
        msa_repr = residual + self.row_attn(msa_repr, dummy_pair_repr)

        cube.runtime.function.anchor('MSACol')
        residual = msa_repr
        msa_repr = self.col_norm(msa_repr)
        msa_repr = residual + self.col_attn(msa_repr)

        # cube.runtime.function.anchor('MSATrans')
        residual = msa_repr
        msa_repr = self.msa_transition_norm(msa_repr)
        msa_repr = self.msa_transition(msa_repr)
        msa_repr = residual + msa_repr
        succ_msa_repr, msa_repr = multi2ref(msa_repr)

        cube.runtime.function.anchor('OPM')
        msa_repr = self.outer_norm(msa_repr)
        pair_repr = pair_repr + self.outer_prod_mean(msa_repr)

        cube.runtime.function.anchor('TMO')
        pair_repr = self.tmo(pair_repr)

        cube.runtime.function.anchor('TMI')
        pair_repr = self.tmi(pair_repr)

        cube.runtime.function.anchor('TANS')
        residual = pair_repr
        pair_repr = self.tri_attn_node_start(pair_repr)
        pair_repr = residual + pair_repr

        cube.runtime.function.anchor('TANE')
        residual = pair_repr
        pair_repr = self.tri_attn_node_end(pair_repr)
        pair_repr = residual + pair_repr

        cube.runtime.function.anchor('PairTrans')
        residual = pair_repr
        pair_repr = self.pair_transition_norm(pair_repr)
        pair_repr = self.pair_transition(pair_repr)
        pair_repr = residual + pair_repr

        return succ_msa_repr, pair_repr
