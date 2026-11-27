# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for Kimi-K2.5 Vision Encoder & Multimodal Scatter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from transformers.models.kimi_k25.modeling_kimi_k25 import (
    get_vision_attention_seqlens,
    get_vision_frame_index,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_temporal_merge_index,
)

from nnscaler.graph.parser.register import CustomizedOps, register_op


def kimi_k25_vision_varlen_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Variable-length non-causal vision SDPA without FX `.tolist()` graph breaks.

    Inputs have shape `[1, H, S_total, D]`; returns `[1, S_total, H, D]`.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_splits = torch.split(query_states, lengths, dim=2)
    k_splits = torch.split(key_states, lengths, dim=2)
    v_splits = torch.split(value_states, lengths, dim=2)
    outs = [
        F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scaling).transpose(1, 2)
        for q, k, v in zip(q_splits, k_splits, v_splits)
    ]
    return torch.cat(outs, dim=1)


def kimi_k25_vision_varlen_sdpa_fake(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    b, h, s, d = query_states.shape
    return query_states.new_empty(
        (b, s, h, d),
        requires_grad=torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query_states, key_states, value_states)
        ),
    )


def kimi_k25_vision_varlen_sdpa_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 4:
        raise ValueError(f"kimi_k25_vision_varlen_sdpa expects 4 tensor inputs, got {len(args)}")
    return "1^ h s^ d^, 1^ h s^ d^, 1^ h s^ d^, n^ -> 1^ s^ h d^"


def kimi_k25_embed_scatter(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Scatter multimodal vision tokens into `inputs_embeds` without dynamic `.tolist()` checks."""
    image_mask = (input_ids == image_token_id).unsqueeze(-1)
    return inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))


def kimi_k25_embed_scatter_fake(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    return torch.empty_like(
        inputs_embeds,
        requires_grad=torch.is_grad_enabled() and (inputs_embeds.requires_grad or image_embeds.requires_grad),
    )


def kimi_k25_embed_scatter_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 3:
        raise ValueError(f"kimi_k25_embed_scatter expects 3 tensor inputs, got {len(args)}")
    return "b s^ d^, b s^, nv^ d^ -> b s^ d^"


def kimi_k25_vision_pos_emb(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    position_embeddings: torch.Tensor,
    time_position_embeddings: torch.Tensor,
    num_grid_per_side: int = 64,
    interpolation_mode: str = "bicubic",
    interpolation_align_corners: bool = False,
) -> torch.Tensor:
    table = position_embeddings.flatten(0, 1)
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side,
        mode=interpolation_mode,
        align_corners=interpolation_align_corners,
        kwargs={},
    )
    pos = F.embedding_bag(interp_indices, table, per_sample_weights=interp_weights.to(table.dtype), mode="sum")
    pos = pos + time_position_embeddings[get_vision_frame_index(grid_thw, kwargs={})]
    return hidden_states + pos.to(hidden_states.dtype)


def kimi_k25_vision_pos_emb_fake(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    position_embeddings: torch.Tensor,
    time_position_embeddings: torch.Tensor,
    num_grid_per_side: int = 64,
    interpolation_mode: str = "bicubic",
    interpolation_align_corners: bool = False,
) -> torch.Tensor:
    return torch.empty_like(
        hidden_states,
        requires_grad=torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (hidden_states, position_embeddings, time_position_embeddings)
        ),
    )


def kimi_k25_vision_pos_emb_anno(*args: Any, **kwargs: Any) -> str:
    return "s^ d^, g^ 3^, h^ w^ d^, f^ d^ -> s^ d^"


def kimi_k25_vision_meta_indices(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_h: int = 2,
    merge_w: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs={})
    cfg = SimpleNamespace(merge_kernel_size=(merge_h, merge_w), _attn_implementation="sdpa")
    cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, cfg, merge_temporal=True, kwargs={})
    merge_index = get_vision_temporal_merge_index(grid_thw, merge_h, merge_w, kwargs={})
    return position_ids, cu_seqlens, merge_index


def kimi_k25_vision_meta_indices_fake(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_h: int = 2,
    merge_w: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = hidden_states.shape[0]
    g = grid_thw.shape[0]
    dev = hidden_states.device
    nm = s // (merge_h * merge_w)
    # These indices feed real indexing operations during CPU tracing.
    return (
        torch.zeros((s, 2), dtype=torch.int64, device=dev),
        torch.zeros((g + 1,), dtype=torch.int32, device=dev),
        torch.zeros((nm, 1, merge_h * merge_w), dtype=torch.int64, device=dev),
    )


def kimi_k25_vision_meta_indices_anno(*args: Any, **kwargs: Any) -> str:
    s = args[0].shape[0]
    g = args[1].shape[0]
    mh = int(kwargs.get("merge_h", 2))
    mw = int(kwargs.get("merge_w", 2))
    nm = s // (mh * mw)
    return f"s^ d^, g^ 3^ -> s^ 2^, {g + 1}^, {nm}^ 1^ {mh * mw}^"


def register_vision_ops() -> dict[str, str]:
    """Idempotently register Kimi-K2.5 vision operators."""
    status: dict[str, str] = {}
    specs = (
        (kimi_k25_vision_varlen_sdpa, kimi_k25_vision_varlen_sdpa_anno, kimi_k25_vision_varlen_sdpa_fake),
        (kimi_k25_embed_scatter, kimi_k25_embed_scatter_anno, kimi_k25_embed_scatter_fake),
        (kimi_k25_vision_pos_emb, kimi_k25_vision_pos_emb_anno, kimi_k25_vision_pos_emb_fake),
        (kimi_k25_vision_meta_indices, kimi_k25_vision_meta_indices_anno, kimi_k25_vision_meta_indices_fake),
    )
    for fn, anno, fake_fn in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        register_op(anno, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
