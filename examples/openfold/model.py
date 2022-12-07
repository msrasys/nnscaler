"""
Alphafold 2, using implementation similar with OpenFold.
"""
import torch
import torch.nn as nn

# from examples.openfold.blocks.embedder import InputEmbedder, RecyclingEmbedder, TemplateAngleEmbedder
from examples.openfold.blocks.evoformer import Evoformer
# from examples.openfold.blocks.evoformer import input_packing, input_unpacking

from dataclasses import dataclass

import cube


@dataclass
class Config:
    
    # input_embedder
    # input_embedder_cm = 256
    # input_embedder_cz = 128
    # input_embedder_msa_dim = 49
    # input_embedder_relpos_k = 32
    # input_embedder_tf_dim = 22

    # recycling embedder
    # recycling_embedder_cm = 256
    # recycling_embedder_cz = 128
    # recycling_embedder_inf = 1000000000.0
    # recycling_embedder_maxbin = 20.75
    # recycling_embedder_minbin = 3.25
    # recycling_embedder_nobins = 15

    # templates
    # template_angle_embedder_cin = 57
    # template_angle_embedder_cout = 256
    # template_pair_embedder_cin = 88
    # template_pair_embedder_cout = 64
    # template_pair_stack_hidden_tri_att = 16
    # template_pair_stack_hidden_tri_mul = 64
    # template_pair_stack_ct = 64
    # template_pair_stack_dp = 0.25
    # template_pair_stack_inf = 100000000.0
    # template_pair_stack_noblocks = 2
    # template_pair_stack_noheads = 4
    # template_pair_stack_pair_transition_n = 2
    # template_pointwise_attention_hidden = 16
    # template_pointwise_attention_ct = 64
    # template_pointwise_attention_cz = 128
    # template_pointwise_inf = 1000000000.0
    # template_pointwise_noheads = 4

    # extra msa
    # extra_msa_embedder_cin = 25
    # extra_msa_embedder_cout = 64
    # extra_msa_stack_hidden_att = 8
    # extra_msa_stack_hidden_mul = 128
    # extra_msa_stack_opm = 32
    # extra_msa_stack_pair_att = 32
    # extra_msa_stack_cm = 64
    # extra_msa_stack_cz = 128
    # extra_msa_stack_eps = 1e-8
    # extra_msa_stack_inf = 1000000000.0
    # extra_msa_dp = 0.15
    # extra_msa_stack_noblocks = 4
    # extra_msa_stack_no_heads_msa = 8
    # extra_msa_stack_no_heads_pair = 4
    # extra_msa_stack_pair_dp = 0.25
    # extra_msa_stack_transition_n = 4

    # evoformer
    evoformer_s: int = 128
    evoformer_r: int = 256
    evoformer_cm: int = 256
    evoformer_cz: int = 128
    evoformer_c: int = 32
    evoformer_use_chunk: bool = False
    evoformer_is_extra: bool = False
    evoformer_nlayers: int = 4

    # batch size
    bs: int = 1


class AlphaFold(nn.Module):


    def __init__(self, cfg: Config = Config()) -> None:
        super().__init__()
        self.cfg = cfg

        # self.input_embedder = InputEmbedder(
        #     cfg.input_embedder_tf_dim, cfg.input_embedder_msa_dim,
        #     cfg.input_embedder_cz, cfg.input_embedder_cm,
        #     cfg.input_embedder_relpos_k
        # )
        # self.recycling_embedder = RecyclingEmbedder(
        #     cfg.recycling_embedder_cm, cfg.recycling_embedder_cz,
        #     cfg.recycling_embedder_minbin, cfg.recycling_embedder_maxbin,
        #     cfg.recycling_embedder_nobins, cfg.recycling_embedder_inf
        # )
        
        # template config
        # self.template_angle_embedder = TemplateAngleEmbedder(
        #     cfg.template_angle_embedder_cin,
        #     cfg.template_angle_embedder_cout
        # )
        # self.template_pair_embedder = nn.Linear(
        #     cfg.template_pair_embedder_cin,
        #     cfg.template_pair_embedder_cout,
        # )
        self.template_pair_stack = None # TemplatePairStack()
        self.template_pointwise_att = None # TemplatePointwiseAttention()

        # extra msa
        # self.extra_msa_embedder = nn.Linear(
        #     cfg.extra_msa_embedder_cin, cfg.extra_msa_embedder_cout
        # )

        self.extra_msa_stack = None # ExtraMSAStack()

        # evoformer
        self.s, self.r, self.cm, self.cz = cfg.evoformer_s, cfg.evoformer_r, cfg.evoformer_cm, cfg.evoformer_cz
        self.c = self.cfg.evoformer_c
        assert self.cm % self.c == 0 and self.cz % self.c == 0

        self.msa_norm = torch.nn.LayerNorm(cfg.evoformer_cm)
        self.pair_norm = torch.nn.LayerNorm(cfg.evoformer_cz)
        self.evoformers = nn.ModuleList(
            [Evoformer(
                self.s, self.r, self.cm, self.cz,
                c=self.c, msa_head=self.cm // self.c, pair_head=self.cz // self.c,
            ) for _ in range(cfg.evoformer_nlayers)]
        )

        self.structure_module = None # StructureModule()
        self.aux_heads = None # AuxiliaryHeads()

    def forward(self, msa, pair):
        """
        msa: [N S R cm]
        pair: [N R R cz]
        """
        msa = self.msa_norm(msa)
        pair = self.pair_norm(pair)
        # cube.runtime.function.anchor('PackingRegion')
        # x = input_packing(msa, pair, self.fout)
        for evoformer in self.evoformers:
            cube.runtime.function.anchor('Evoformer Start')
            msa, pair = evoformer(msa, pair)
            # x = evoformer(x)
        # msa, pair = input_unpacking(x, self.s, self.r, self.cm, self.cz)
        loss = torch.sum(msa) * torch.sum(pair)
        return loss

    
    def tflops(self) -> float:
        """
        TFLOPs for one sample
        """
        tflops = 0.
        for layer in self.evoformers:
            tflops += layer.tflops(self.s, self.r)
        return tflops
