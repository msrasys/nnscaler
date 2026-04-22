# Qwen3.5 modifier for nnscaler
# Registers custom ops and monkey-patches model for clean tracing.

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnscaler.graph.parser.register import register_op

# Auto-detect optimized kernels (fla + causal-conv1d)
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
    _use_fla = True
    # Disable FusedRMSNormGated: fla's fused kernel breaks nnscaler tracer
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen3_5_mod
    _qwen3_5_mod.FusedRMSNormGated = None
except ImportError:
    _fla_chunk_gated_delta_rule = None
    _use_fla = False

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
    _use_causal_conv1d = True
except ImportError:
    _causal_conv1d_fn = None
    _use_causal_conv1d = False

# 1. Register flash attention (for Gated Attention layers)
# Import the shared flash_attn_anno module which registers _flash_attention_forward
# and patches ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transformers_utils'))
import flash_attn_anno  # noqa: F401 — side-effect import

# Pre-load flash attention so runtime processes have _loaded_implementation set.
# Without this, _flash_attention_forward(attn_implementation=None) fails in gencode.
from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
lazy_import_flash_attention('flash_attention_2')

# 2. Register chunked cross-entropy loss (avoids materializing full logits)
# For 128K seq × 248K vocab, full logits = 60+ GB. Chunk to avoid OOM.

CHUNK_SIZE = 512  # tokens per chunk for CE computation (small to avoid OOM at 128K)


def nnscaler_chunked_cross_entropy(hidden_states, weight, labels):
    """Compute cross-entropy loss without materializing full logits.

    Args:
        hidden_states: (batch, seq, hidden_dim) — already shifted
        weight: (vocab, hidden_dim) — lm_head weight
        labels: (batch, seq) — already shifted targets
    Returns:
        loss: scalar tensor — mean cross-entropy loss
    """
    B, S, D = hidden_states.shape
    V = weight.shape[0]
    total_loss = torch.zeros(1, device=hidden_states.device, dtype=torch.float32)
    total_tokens = 0
    for i in range(0, S, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, S)
        chunk_h = hidden_states[:, i:end, :].contiguous()
        chunk_logits = F.linear(chunk_h, weight)  # (B, chunk, V)
        chunk_labels = labels[:, i:end].contiguous()
        chunk_loss = F.cross_entropy(
            chunk_logits.reshape(-1, V),
            chunk_labels.reshape(-1),
            reduction='sum',
        )
        total_loss = total_loss + chunk_loss
        total_tokens += chunk_labels.numel()
    return (total_loss / total_tokens).squeeze()


register_op('b l^ d^, v^ d^, b l^ -> 1')(nnscaler_chunked_cross_entropy)


# 3. Register torch_chunk_gated_delta_rule as atomic op
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule as _torch_chunk_gated_delta_rule,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5Attention,
    Qwen3_5RMSNormGated,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
)


def nnscaler_chunk_gated_delta_rule(query, key, value, g, beta):
    """Wrapper with fixed kwargs for training (no cache, no final state).
    Uses fla optimized kernel if available, otherwise PyTorch fallback.
    Inputs (after repeat_interleave in DeltaNet forward):
      query, key, value: (batch, seq_len, num_v_heads, head_dim)
      g, beta: (batch, seq_len, num_v_heads)
    Output:
      core_attn_out: (batch, seq_len, num_v_heads, head_dim)
    """
    if _use_fla:
        output, _ = _fla_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        output, _ = _torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
    return output


# Annotation: batch can split, heads can split, seq and head_dim frozen
register_op('b l^ h d^, b l^ h d^, b l^ h d^, b l^ h, b l^ h -> b l^ h d^')(
    nnscaler_chunk_gated_delta_rule
)


# 4. Register causal conv1d wrapper (auto-selects optimized kernel or fallback)

def nnscaler_causal_conv1d(x, weight):
    """Causal conv1d + SiLU. Uses causal_conv1d_fn if available, else nn.Conv1d fallback.
    Args:
        x: (batch, dim, seqlen)
        weight: (dim, 1, kernel_size) — Conv1d weight (no bias)
    Returns:
        (batch, dim, seqlen)
    """
    if _use_causal_conv1d:
        return _causal_conv1d_fn(x=x, weight=weight.squeeze(1), bias=None, activation="silu")
    else:
        seq_len = x.shape[2]
        return F.silu(F.conv1d(x, weight, None, groups=x.shape[1],
                               padding=weight.shape[2] - 1)[:, :, :seq_len])


register_op('b d^ l^, d^ 1 k^ -> b d^ l^')(nnscaler_causal_conv1d)


# 5. Monkey-patch Qwen3_5GatedDeltaNet.forward for clean tracing
# Remove all cache/precomputed_state branches and causal_conv1d_fn branches.
# Training path only: no cache, use nn.Conv1d fallback, call registered delta rule.
def _patched_deltanet_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    # Projections
    mixed_qkv = self.in_proj_qkv(hidden_states)  # (b, s, conv_dim)
    mixed_qkv = mixed_qkv.transpose(1, 2)        # (b, conv_dim, s)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Causal conv1d (registered atomic op, auto-selects optimized kernel)
    mixed_qkv = nnscaler_causal_conv1d(mixed_qkv, self.conv1d.weight)

    mixed_qkv = mixed_qkv.transpose(1, 2)  # (b, s, conv_dim)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # Use registered atomic op
    core_attn_out = nnscaler_chunk_gated_delta_rule(query, key, value, g, beta)

    # Norm and output
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


Qwen3_5GatedDeltaNet.forward = _patched_deltanet_forward


# 6. Monkey-patch Qwen3_5TextRotaryEmbedding.forward
# Remove maybe_autocast (its type annotation `torch.dtype | None` breaks
# nnscaler tracer's exec-based patching).  The original call is
# maybe_autocast(enabled=False) which is effectively a no-op during training.

def _patched_rope_forward(self, x, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


Qwen3_5TextRotaryEmbedding.forward = _patched_rope_forward


# 7. Monkey-patch Qwen3_5Attention.forward
# The original forward uses ALL_ATTENTION_FUNCTIONS.get_interface() which
# calls dict.get() — the tracer can't inspect the returned callable.
# Directly call flash_attention_forward from flash_attn_anno.
from flash_attn_anno import flash_attention_forward as _flash_attn_fwd


def _patched_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=3
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Directly call flash attention (bypass ALL_ATTENTION_FUNCTIONS.get_interface)
    attn_output, attn_weights = _flash_attn_fwd(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


Qwen3_5Attention.forward = _patched_attention_forward
