# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

import nnscaler


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., : key.shape[-2]]
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = torch.dropout(probabilities, dropout, module.training)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return output, probabilities


@nnscaler.register_op(
    "b t h d^, b t h d^, b t h v^, b t h v^, b t h, h, (h v^) "
    "-> b t h v^, b h v^ d^"
)
def nnscaler_chunk_kda(q, k, v, g, beta, A_log, dt_bias):
    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
        cu_seqlens=None,
    )


def precomputed_kimi_causal_mask(
    config,
    input_embeds,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    **kwargs,
):
    sequence_length = input_embeds.shape[1]
    return torch.full(
        (input_embeds.shape[0], 1, sequence_length, sequence_length),
        torch.finfo(input_embeds.dtype).min,
        dtype=input_embeds.dtype,
        device=input_embeds.device,
    ).triu(diagonal=1)


def batched_sparse_moe_forward(self, hidden_states):
    identity = hidden_states
    original_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    if self.use_latent_moe:
        hidden_states = self.routed_expert_down_proj(hidden_states)

    num_tokens = hidden_states.shape[0]
    top_k = topk_indices.shape[-1]
    token_indices = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    expert_indices = topk_indices.reshape(-1)
    selected_states = hidden_states[token_indices]

    w1 = torch.stack([expert.w1.weight for expert in self.experts])[expert_indices]
    w2 = torch.stack([expert.w2.weight for expert in self.experts])[expert_indices]
    w3 = torch.stack([expert.w3.weight for expert in self.experts])[expert_indices]
    gate = torch.bmm(w1, selected_states.unsqueeze(-1)).squeeze(-1)
    up = torch.bmm(w3, selected_states.unsqueeze(-1)).squeeze(-1)
    activated = self.experts[0].act_fn(torch.cat((gate, up), dim=-1))
    routed = torch.bmm(w2, activated.unsqueeze(-1)).squeeze(-1)
    routed = routed * topk_weights.reshape(-1, 1)
    output = routed.view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)

    if self.use_latent_moe:
        if self.latent_moe_use_norm:
            output = self.routed_expert_norm(output)
        output = self.routed_expert_up_proj(output)

    output = output.view(*original_shape)
    if self.config.num_shared_experts is not None:
        output = output + self.shared_experts(identity)
    return output
