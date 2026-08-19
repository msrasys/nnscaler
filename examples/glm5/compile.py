# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa

import nnscaler

from examples.moe_utils import select_experts
from examples.model_runtime import (
    assert_finite_tensors,
    default_output_dir,
    finish_distributed,
    init_distributed,
    print_rank0,
    require_generated_output,
)


def _precomputed_causal_mask(
    config,
    inputs_embeds,
    attention_mask,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    **kwargs,
):
    if attention_mask is None or attention_mask.ndim not in (3, 4):
        raise ValueError("GLM-5 nnScaler tracing requires a precomputed 4D causal mask")
    return attention_mask


def _rotary_embedding_forward(self, x, position_ids):
    inverse_frequencies = self.inv_freq[None, :, None].float()
    positions = position_ids[:, None, :].float()
    frequencies = (inverse_frequencies * positions).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos() * self.attention_scaling
    sine = embedding.sin() * self.attention_scaling
    return cosine.to(dtype=x.dtype), sine.to(dtype=x.dtype)


def install_glm5_nnscaler_adapters() -> None:
    # Transformers 5.3 decorates this helper with config-aware Python logic
    # that torch.fx cannot serialize. The wrapper always supplies the same 4D
    # additive mask that the helper would return.
    modeling_glm_moe_dsa.create_causal_mask = _precomputed_causal_mask
    modeling_glm_moe_dsa.GlmMoeDsaRotaryEmbedding.forward = _rotary_embedding_forward
    modeling_glm_moe_dsa.GlmMoeDsaIndexer.forward = _indexer_forward
    modeling_glm_moe_dsa.GlmMoeDsaMoE.route_tokens_to_experts = _route_tokens_to_experts


@torch.no_grad()
def _indexer_forward(
    self,
    hidden_states,
    q_resid,
    position_embeddings,
    attention_mask,
    use_cache=False,
):
    batch_size, sequence_length, _ = hidden_states.shape
    cosine, sine = position_embeddings
    query = self.wq_b(q_resid).view(
        batch_size,
        sequence_length,
        self.n_heads,
        self.head_dim,
    )
    query_rotary, query_pass = torch.split(
        query,
        [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
        dim=-1,
    )
    query_rotary = modeling_glm_moe_dsa.apply_rotary_pos_emb(
        query_rotary,
        cosine,
        sine,
        unsqueeze_dim=2,
    )
    query = torch.cat((query_rotary, query_pass), dim=-1)

    key = self.k_norm(self.wk(hidden_states))
    key_rotary, key_pass = torch.split(
        key,
        [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
        dim=-1,
    )
    key_rotary = modeling_glm_moe_dsa.apply_rotary_pos_emb(
        key_rotary.unsqueeze(2),
        cosine,
        sine,
        unsqueeze_dim=2,
    ).squeeze(2)
    key = torch.cat((key_rotary, key_pass), dim=-1)

    if sequence_length > 1:
        self._cached_keys = None
    if use_cache:
        key = torch.cat((self._cached_keys, key), dim=1) if self._cached_keys is not None else key
        self._cached_keys = key

    scores = torch.einsum("bshd,btd->bsht", query.float(), key.float()) * self.softmax_scale
    weights = self.weights_proj(hidden_states).float()
    weights = weights * (self.n_heads**-0.5)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    top_k = min(self.index_topk, index_scores.shape[-1])
    return torch.argsort(index_scores, dim=-1)[..., -top_k:]


def _route_tokens_to_experts(self, router_logits):
    scores = router_logits.sigmoid()
    topk_indices = select_experts(
        scores,
        self.gate.e_score_correction_bias,
        self.n_group,
        self.n_routed_experts,
        self.topk_group,
        self.top_k,
    )
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_indices, topk_weights * self.routed_scaling_factor


class GLM5TrainingModel(nn.Module):
    """GLM-5 CausalLM wrapper with a scalar training loss for nnScaler."""

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.model = GlmMoeDsaForCausalLM(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)
        causal_mask = torch.full(
            (
                input_ids.shape[0],
                1,
                sequence_length,
                sequence_length,
            ),
            torch.finfo(self.model.dtype).min,
            dtype=self.model.dtype,
            device=input_ids.device,
        ).triu(diagonal=1)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=position_ids[0],
            use_cache=False,
            return_dict=False,
        )[0]
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            input_ids[:, 1:].contiguous().view(-1),
        )


def reduced_glm5_config() -> GlmMoeDsaConfig:
    """Keep GLM-5's DSA and sparse-MoE paths while making the probe small."""

    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=24,
        n_group=1,
        topk_group=1,
        max_position_embeddings=32,
        mlp_layer_types=["sparse"],
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        experts_implementation="batched_mm",
        use_cache=False,
    )


def compile_model(
    plan_ngpus: int,
    runtime_ngpus: int,
    output_dir: Path,
    batch_size: int = 2,
    sequence_length: int = 8,
) -> None:
    install_glm5_nnscaler_adapters()
    config = reduced_glm5_config()
    model = GLM5TrainingModel(config)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
    )
    compute_config = nnscaler.ComputeConfig(
        plan_ngpus=plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        use_zero=True,
        use_end2end=True,
        pas_config={"mem_constraint": 40},
    )
    nnscaler.parallelize(
        model,
        {"input_ids": input_ids},
        "autodist",
        compute_config,
        gen_savedir=output_dir,
        load_module=False,
    )


def run_model(
    plan_ngpus: int,
    runtime_ngpus: int,
    output_dir: Path,
    steps: int,
) -> None:
    require_generated_output(output_dir)
    rank = init_distributed(runtime_ngpus)
    torch.manual_seed(0)
    config = reduced_glm5_config()
    vocab_size = config.vocab_size
    model = GLM5TrainingModel(config)
    compute_config = nnscaler.ComputeConfig(
        plan_ngpus=plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        use_zero=True,
        use_end2end=True,
        pas_config={"mem_constraint": 40},
    )
    model = nnscaler.parallelize(
        model,
        {"input_ids": torch.zeros((2, 8), dtype=torch.long)},
        "autodist",
        compute_config,
        gen_savedir=output_dir,
    ).cuda()
    optimizer = nnscaler.build_optimizer(model, torch.optim.AdamW, lr=1e-4)
    replica_rank = rank // plan_ngpus

    for step in range(steps):
        generator = torch.Generator(device="cuda").manual_seed(1000 + replica_rank * steps + step)
        input_ids = torch.randint(
            0,
            vocab_size,
            (2, 8),
            generator=generator,
            device="cuda",
        )
        losses = model.train_step([input_ids])
        assert_finite_tensors(losses, "GLM-5 losses")
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("GLM-5 run produced no gradients")
        assert_finite_tensors(gradients, "GLM-5 gradients")
        optimizer.step()
        optimizer.zero_grad()

    print_rank0(f"GLM-5 completed {steps} distributed training step(s)")
    finish_distributed()


def eager_check() -> None:
    install_glm5_nnscaler_adapters()
    config = reduced_glm5_config()
    model = GLM5TrainingModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(input_ids)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"GLM-5 eager check returned invalid loss: {loss}")
    print(f"GLM-5 reduced eager check passed (loss={loss.item():.6f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an operator-equivalent reduced GLM-5 with nnScaler."
    )
    parser.add_argument(
        "--mode",
        choices=("eager", "compile", "run"),
        default="compile",
    )
    parser.add_argument("--plan-ngpus", type=int, default=2)
    parser.add_argument("--runtime-ngpus", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("glm5"),
    )
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "eager":
        eager_check()
        return
    if args.plan_ngpus < 1 or args.runtime_ngpus < 1:
        raise ValueError("GPU counts must be positive")
    if args.runtime_ngpus % args.plan_ngpus != 0:
        raise ValueError("runtime-ngpus must be a multiple of plan-ngpus")
    if args.mode == "compile":
        compile_model(args.plan_ngpus, args.runtime_ngpus, args.output_dir)
        print(f"Generated nnScaler code in {args.output_dir}")
    else:
        run_model(
            args.plan_ngpus,
            args.runtime_ngpus,
            args.output_dir,
            args.steps,
        )


if __name__ == "__main__":
    main()
