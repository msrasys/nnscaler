# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
from transformers.models.deepseek_v32 import modeling_deepseek_v32

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
        raise ValueError("DeepSeek-V3.2 tracing requires a precomputed 4D causal mask")
    return attention_mask


def _rotary_embedding_forward(self, x, position_ids):
    inverse_frequencies = self.inv_freq[None, :, None].float()
    positions = position_ids[:, None, :].float()
    frequencies = (inverse_frequencies * positions).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos() * self.attention_scaling
    sine = embedding.sin() * self.attention_scaling
    return cosine.to(dtype=x.dtype), sine.to(dtype=x.dtype)


@torch.no_grad()
def _indexer_forward(
    self,
    hidden_states,
    q_resid,
    position_embeddings,
    attention_mask,
    position_ids,
    past_key_values=None,
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
    key = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    key_rotary, key_pass = torch.split(
        key,
        [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
        dim=-1,
    )
    query_rotary, key_rotary = modeling_deepseek_v32.apply_rotary_pos_emb(
        query_rotary,
        key_rotary,
        cosine,
        sine,
        unsqueeze_dim=2,
    )
    query = torch.cat((query_rotary, query_pass), dim=-1)
    key = torch.cat((key_rotary, key_pass), dim=-1).squeeze(2)
    scores = torch.einsum("bshd,btd->bsht", query, key) * self.softmax_scale
    scores = F.relu(scores)
    weights = self.weights_proj(hidden_states)
    weights = weights * (self.n_heads**-0.5)
    index_scores = torch.einsum("bsh,bsht->bst", weights, scores)
    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    else:
        key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))
    top_k = min(self.index_topk, index_scores.shape[-1])
    return torch.argsort(index_scores, dim=-1)[..., -top_k:]


def _expand_kv(self, kv_nope, key_rotary):
    batch_size, _, sequence_length, _ = kv_nope.shape
    key_value = self.kv_b_proj(kv_nope).view(
        batch_size,
        sequence_length,
        -1,
        self.qk_nope_head_dim + self.v_head_dim,
    ).transpose(1, 2)
    key_nope, value_states = torch.split(
        key_value,
        [self.qk_nope_head_dim, self.v_head_dim],
        dim=-1,
    )
    key_rotary = key_rotary.expand(-1, key_nope.shape[1], -1, -1)
    return torch.cat((key_nope, key_rotary), dim=-1), value_states


def _router_forward(self, hidden_states):
    hidden_states = hidden_states.view(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    scores = router_logits.sigmoid()
    topk_indices = select_experts(
        scores,
        self.e_score_correction_bias,
        self.num_group,
        self.num_experts,
        self.topk_group,
        self.top_k,
    )
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices


def _experts_forward(self, hidden_states, top_k_index, top_k_weights):
    num_tokens = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    token_indices = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    expert_indices = top_k_index.reshape(-1)
    selected_states = hidden_states[token_indices]
    gate_up_weights = self.gate_up_proj[expert_indices]
    gate_up = torch.bmm(gate_up_weights, selected_states.unsqueeze(-1)).squeeze(-1)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = self.act_fn(gate) * up
    down_weights = self.down_proj[expert_indices]
    routed = torch.bmm(down_weights, activated.unsqueeze(-1)).squeeze(-1)
    routed = routed * top_k_weights.reshape(-1, 1)
    return routed.view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def install_deepseek_v32_nnscaler_adapters() -> None:
    modeling_deepseek_v32.create_causal_mask = _precomputed_causal_mask
    modeling_deepseek_v32.DeepseekV32RotaryEmbedding.forward = _rotary_embedding_forward
    modeling_deepseek_v32.DeepseekV32Indexer.forward = _indexer_forward
    modeling_deepseek_v32.DeepseekV32Attention.expand_kv = _expand_kv
    modeling_deepseek_v32.DeepseekV32TopkRouter.forward = _router_forward
    modeling_deepseek_v32.DeepseekV32Experts.forward = _experts_forward


def reduced_deepseek_v32_config() -> DeepseekV32Config:
    return DeepseekV32Config(
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


class DeepSeekV32TrainingModel(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.model = DeepseekV32ForCausalLM(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        causal_mask = torch.full(
            (input_ids.shape[0], 1, sequence_length, sequence_length),
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


def build_model() -> DeepSeekV32TrainingModel:
    install_deepseek_v32_nnscaler_adapters()
    return DeepSeekV32TrainingModel(reduced_deepseek_v32_config())


def eager_check() -> None:
    model = build_model().cuda()
    input_ids = torch.randint(0, model.model.config.vocab_size, (2, 8), device="cuda")
    loss = model(input_ids)
    loss.backward()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"DeepSeek-V3.2 eager check returned invalid loss: {loss}")
    print(f"DeepSeek-V3.2 reduced eager check passed (loss={loss.item():.6f})")


def compile_model(plan_ngpus: int, runtime_ngpus: int, output_dir: Path) -> None:
    model = build_model()
    input_ids = torch.randint(0, model.model.config.vocab_size, (2, 8))
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
    config = reduced_deepseek_v32_config()
    vocab_size = config.vocab_size
    install_deepseek_v32_nnscaler_adapters()
    model = DeepSeekV32TrainingModel(config)
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
        generator = torch.Generator(device="cuda").manual_seed(2000 + replica_rank * steps + step)
        input_ids = torch.randint(
            0,
            vocab_size,
            (2, 8),
            generator=generator,
            device="cuda",
        )
        losses = model.train_step([input_ids])
        assert_finite_tensors(losses, "DeepSeek-V3.2 losses")
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("DeepSeek-V3.2 run produced no gradients")
        assert_finite_tensors(gradients, "DeepSeek-V3.2 gradients")
        optimizer.step()
        optimizer.zero_grad()

    print_rank0(f"DeepSeek-V3.2 completed {steps} distributed training step(s)")
    finish_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an operator-equivalent reduced DeepSeek-V3.2 with nnScaler."
    )
    parser.add_argument("--mode", choices=("eager", "compile", "run"), default="compile")
    parser.add_argument("--plan-ngpus", type=int, default=2)
    parser.add_argument("--runtime-ngpus", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("deepseek-v32"),
    )
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "eager":
        eager_check()
        return
    if args.plan_ngpus < 1 or args.runtime_ngpus % args.plan_ngpus != 0:
        raise ValueError("runtime-ngpus must be a positive multiple of plan-ngpus")
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
