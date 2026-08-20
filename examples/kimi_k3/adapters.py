# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

import nnscaler

from examples.moe_utils import tensorized_situ_moe


@nnscaler.register_op("b^ t^ d^, d^ 1 w -> b^ t^ d^, ?")
def nnscaler_short_convolution(
    x,
    weight,
    *,
    activation,
    backend,
):
    from fla.modules.conv.causal_conv1d import causal_conv1d

    return causal_conv1d(
        x=x,
        weight=weight.squeeze(1),
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        backend=backend,
        cu_seqlens=None,
        chunk_indices=None,
    )


def short_convolution_forward(
    self,
    x,
    residual=None,
    mask=None,
    cache=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    **kwargs,
):
    if any(value is not None for value in (residual, mask, cache, cu_seqlens, chunk_indices)):
        raise ValueError("the reduced Kimi K3 probe only supports dense prefill convolution")
    if output_final_state:
        raise ValueError("the reduced Kimi K3 probe does not produce convolution cache state")
    return nnscaler_short_convolution(
        x,
        self.weight,
        activation=self.activation,
        backend=self.backend,
    )


@nnscaler.register_op("* d^, * d^, d^ -> * d^")
def nnscaler_rms_norm_gated(x, gate, weight, *, activation, eps):
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(
        x,
        gate,
        weight,
        None,
        activation,
        residual=None,
        eps=eps,
        prenorm=False,
        residual_in_fp32=False,
    )


def rms_norm_gated_forward(
    self,
    x,
    gate,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
):
    if residual is not None or prenorm or residual_in_fp32:
        raise ValueError("the reduced Kimi K3 probe only supports post-norm without residual fusion")
    return nnscaler_rms_norm_gated(
        x,
        gate,
        self.weight,
        activation=self.activation,
        eps=self.eps,
    )


def merge_kimi_expert_parameters(model, sparse_moe_type) -> None:
    for module in model.modules():
        if not isinstance(module, sparse_moe_type) or hasattr(module, "expert_w1"):
            continue
        activation = module.experts[0].act_fn
        module.register_parameter(
            "expert_w1",
            torch.nn.Parameter(torch.stack([expert.w1.weight.detach() for expert in module.experts])),
        )
        module.register_parameter(
            "expert_w2",
            torch.nn.Parameter(torch.stack([expert.w2.weight.detach() for expert in module.experts])),
        )
        module.register_parameter(
            "expert_w3",
            torch.nn.Parameter(torch.stack([expert.w3.weight.detach() for expert in module.experts])),
        )
        module.expert_activation_beta = activation.beta
        module.expert_activation_linear_beta = activation.linear_beta
        del module.experts


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
def nnscaler_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    *,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=False,
    lower_bound=None,
    transpose_state_layout=True,
    cu_seqlens=None,
):
    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        transpose_state_layout=transpose_state_layout,
        cu_seqlens=cu_seqlens,
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

    output = tensorized_situ_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        self.expert_w1,
        self.expert_w2,
        self.expert_w3,
        num_experts=self.num_experts,
        local_expert_start=0,
        local_expert_end=self.num_experts,
        activation_beta=self.expert_activation_beta,
        activation_linear_beta=self.expert_activation_linear_beta,
    )

    if self.use_latent_moe:
        if self.latent_moe_use_norm:
            output = self.routed_expert_norm(output)
        output = self.routed_expert_up_proj(output)

    output = output.view(*original_shape)
    if self.config.num_shared_experts is not None:
        output = output + self.shared_experts(identity)
    return output
