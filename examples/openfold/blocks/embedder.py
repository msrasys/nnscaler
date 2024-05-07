import torch
import torch.nn as nn

from typing import Tuple, Optional

import nnscaler



@nnscaler.register_op('N res, cz nobins, cz -> N res res cz', name='relpos')
def input_embedder_pair_emb(ri: torch.Tensor,
                            tf_emb_i: torch.Tensor, tf_emb_j: torch.Tensor,
                            w_relpos: torch.Tensor, b_relpos: torch.Tensor,
                            relpos_k) -> torch.Tensor:

    ri = ri.type(tf_emb_i.dtype)
    d = ri[..., None] - ri[..., None, :]
    boundaries = torch.arange(
        start=-relpos_k, end=relpos_k + 1, device=torch.cuda.current_device()
    )
    reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
    d = d[..., None] - reshaped_bins
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)
    d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
    d = d.to(ri.dtype)
    pair_emb = torch.nn.functional.linear(d, w_relpos, b_relpos)

    pair_emb = pair_emb + tf_emb_i[..., None, :]
    pair_emb = pair_emb + tf_emb_j[..., None, :, :]

    return pair_emb


@nnscaler.register_op('N res tfdim^, cm tfdim^, cm -> N nclust^, res, cm')
def input_embedder_tf_m(tf: torch.Tensor, w_tf_m: torch.Tensor, b_tf_m: torch.Tensor, nclust: int) -> torch.Tensor:
    tf_m = torch.nn.linear(tf, w_tf_m, b_tf_m)
    tf_m = tf_m.unsqueeze(-3).expand(((-1,) * len(tf.shape[:-2]) + (nclust, -1, -1)))
    return tf_m


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(self, tf_dim: int, msa_dim: int, c_z: int, c_m: int, relpos_k: int):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        # self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.w_tf_m = torch.nn.Parameter(torch.empty((c_m, tf_dim)))
        self.b_tf_m = torch.nn.Parameter(torch.empty((c_m)))
        self.linear_msa_m = nn.Linear(msa_dim, c_m)
        self.w_tf_m

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.w_linear_relpos = torch.nn.Parameter(torch.empty((c_z, self.no_bins)))
        self.b_linear_relpos = torch.nn.Parameter(torch.empty((c_z,)))

    def forward(self, tf: torch.Tensor, ri: torch.Tensor, msa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = input_embedder_pair_emb(
            ri, tf_emb_i, tf_emb_j,
            self.w_linear_relpos, self.b_linear_relpos
        )
        # pair_emb = relpos(ri.type(tf_emb_i.dtype))
        # pair_emb = pair_emb + tf_emb_i[..., None, :]
        # pair_emb = pair_emb + tf_emb_j[..., None, :, :]

        # [*, N_clust, N_res, c_m]
        tf_m = input_embedder_tf_m(tf, self.w_tf_m, self.b_tf_m)
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb



@nnscaler.register_op()
def sum_d(x: torch.Tensor, bins: torch.Tensor, inf: float) -> torch.Tensor:
    squared_bins = bins ** 2
    upper = torch.cat(
        [squared_bins[1:], squared_bins.new_tensor([inf])], dim=-1
    )
    d = torch.sum(
        (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
    )
    d = ((d > squared_bins) * (d < upper)).type(x.dtype)
    return d


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(self, c_m: int, c_z: int,
                min_bin: float, max_bin: float, no_bins: int,
                inf: float = 1e8):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = nn.Linear(self.no_bins, self.c_z)
        self.layer_norm_m = nn.LayerNorm(self.c_m)
        self.layer_norm_z = nn.LayerNorm(self.c_z)

        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins, requires_grad=False)
        self.register_buffer('bins', bins)

    def forward(self, m: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        m = self.layer_norm_m(m)
        z = self.layer_norm_z(z)
        d = sum_d(x, self.bins, self.inf)
        d = self.linear(d)
        z = z + d
        return m, z


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """
    def __init__(self, c_in: int, c_out: int):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = nn.Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

