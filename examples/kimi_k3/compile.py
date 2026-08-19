# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

import nnscaler

from examples.kimi_k3.adapters import (
    batched_sparse_moe_forward,
    eager_attention_forward,
    nnscaler_chunk_kda,
    precomputed_kimi_causal_mask,
)
from examples.model_runtime import (
    assert_finite_tensors,
    default_output_dir,
    finish_distributed,
    init_distributed,
    print_rank0,
    require_generated_output,
)


MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


def reduced_kimi_k3_config():
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    text_config = config.text_config
    text_config._name_or_path = MODEL_ID
    text_config.vocab_size = 128
    text_config.pad_token_id = 0
    text_config.bos_token_id = 1
    text_config.eos_token_id = 2
    text_config.hidden_size = 64
    text_config.intermediate_size = 128
    text_config.moe_intermediate_size = 32
    text_config.routed_expert_hidden_size = 32
    text_config.num_hidden_layers = 2
    text_config.num_attention_heads = 4
    text_config.num_key_value_heads = 4
    text_config.q_lora_rank = 32
    text_config.kv_lora_rank = 16
    text_config.qk_nope_head_dim = 8
    text_config.qk_rope_head_dim = 8
    text_config.v_head_dim = 16
    text_config.num_experts = 4
    text_config.num_experts_per_token = 2
    text_config.num_shared_experts = 1
    text_config.first_k_dense_replace = 0
    text_config.num_expert_group = 1
    text_config.topk_group = 1
    text_config.max_position_embeddings = 64
    text_config.attn_res_block_size = None
    text_config.linear_attn_config = {
        "full_attn_layers": [1],
        "kda_layers": [0],
        "gate_lower_bound": -5.0,
        "head_dim": 16,
        "num_heads": 4,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    }
    text_config.quantization_config = None
    text_config.use_cache = False
    text_config._attn_implementation = "eager"
    return text_config


def install_transformers_compatibility() -> None:
    # Kimi K3's reference code still imports these symbols from
    # transformers.utils.generic, while Transformers 5 moved output recording
    # to a dedicated module and removed the input decorator.
    from transformers.utils import generic
    from transformers.utils.output_capturing import OutputRecorder

    generic.OutputRecorder = OutputRecorder
    if not hasattr(generic, "check_model_inputs"):
        generic.check_model_inputs = lambda function: function


def install_kimi_k3_nnscaler_adapters(model) -> None:
    modeling_module = sys.modules[type(model).__module__]
    modeling_module.chunk_kda = nnscaler_chunk_kda
    modeling_module.create_causal_mask = precomputed_kimi_causal_mask
    modeling_module.eager_attention_forward = eager_attention_forward
    modeling_module.KimiSparseMoeBlock.forward = batched_sparse_moe_forward
    model.config._attn_implementation = "eager"


class KimiK3InferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        install_transformers_compatibility()
        self.model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
        install_kimi_k3_nnscaler_adapters(self.model)
        self.model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def build_model() -> KimiK3InferenceModel:
    return KimiK3InferenceModel(reduced_kimi_k3_config())


def eager_check() -> None:
    model = build_model().cuda()
    input_ids = torch.randint(0, model.model.config.vocab_size, (2, 16), device="cuda")
    with torch.no_grad():
        logits = model(input_ids)
    if logits.shape != (2, 16, model.model.config.vocab_size) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Kimi K3 eager check returned invalid logits: {logits.shape}")
    print("Kimi K3 reduced eager check passed")


def compile_model(plan_ngpus: int, runtime_ngpus: int, output_dir: Path) -> None:
    model = build_model()
    input_ids = torch.randint(0, model.model.config.vocab_size, (2, 16))
    compute_config = nnscaler.ComputeConfig(
        plan_ngpus=plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        inference_only=True,
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
    config = reduced_kimi_k3_config()
    vocab_size = config.vocab_size
    model = KimiK3InferenceModel(config)
    compute_config = nnscaler.ComputeConfig(
        plan_ngpus=plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        inference_only=True,
        pas_config={"mem_constraint": 40},
    )
    model = nnscaler.parallelize(
        model,
        {"input_ids": torch.zeros((2, 16), dtype=torch.long)},
        "autodist",
        compute_config,
        gen_savedir=output_dir,
    ).cuda()
    model.eval()
    replica_rank = rank // plan_ngpus

    with torch.inference_mode():
        for step in range(steps):
            generator = torch.Generator(device="cuda").manual_seed(3000 + replica_rank * steps + step)
            input_ids = torch.randint(
                0,
                vocab_size,
                (2, 16),
                generator=generator,
                device="cuda",
            )
            logits = model(input_ids)
            if logits.shape != (2, 16, vocab_size):
                raise RuntimeError(f"Kimi K3 returned unexpected shape: {logits.shape}")
            assert_finite_tensors(logits, "Kimi K3 logits")

    print_rank0(f"Kimi K3 completed {steps} distributed inference step(s)")
    finish_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an operator-equivalent reduced Kimi K3 with nnScaler."
    )
    parser.add_argument("--mode", choices=("eager", "compile", "run"), default="compile")
    parser.add_argument("--plan-ngpus", type=int, default=2)
    parser.add_argument("--runtime-ngpus", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("kimi-k3"),
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
