# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hardware-decoupled zero-GPU compilation & verification bundle CLI for DeepSeek models.

Example usage on a CPU-only laptop/VM (0 GPUs):
  PYTHONPATH=. python examples/Deepseek/compile.py \
      --model deepseek-v3 \
      --strategy examples/Deepseek/strategies/v3_ep8_pp4.json \
      --out-dir /tmp/deepseek_v3_verify_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from nnscaler.parallel import parallelize

from examples.Deepseek.audit import (
    audit_post_pas_graph,
    audit_pre_pas_graph,
    export_verification_bundle,
)
from examples.Deepseek.configs import (
    build_deepseek_model,
    build_dummy_samples,
    list_model_presets,
    load_model_config_dict,
)
from examples.Deepseek.ops import register_all_deepseek_ops
from examples.Deepseek.policy import (
    StrategyConfig,
    StrategyValidationError,
    build_compute_config,
    build_pas_policy,
    load_strategy_config,
)


def compile_deepseek(
    model_preset_or_path: str | Path | dict[str, Any],
    strategy_source: str | Path | dict[str, Any] | StrategyConfig,
    out_dir: str | Path,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    weightless: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    num_layers_override: int | None = None,
) -> dict[str, Any]:
    """Compile any DeepSeek model + declarative JSON strategy into an auditable verification bundle."""
    t0 = time.perf_counter()
    out_path = Path(out_dir)
    nnscaler_gen_dir = out_path / "_nnscaler_gen"
    nnscaler_gen_dir.mkdir(parents=True, exist_ok=True)

    op_reg_status = register_all_deepseek_ops()

    strategy = (
        strategy_source
        if isinstance(strategy_source, StrategyConfig)
        else load_strategy_config(strategy_source)
    )
    model_cfg_dict = load_model_config_dict(model_preset_or_path)
    if num_layers_override is not None:
        model_cfg_dict["num_hidden_layers"] = int(num_layers_override)

    wrapper, model_meta = build_deepseek_model(
        model_cfg_dict,
        rewrite_cfg=strategy.rewrites,
        weightless=weightless,
        dtype=dtype,
    )
    model_meta["registered_ops"] = op_reg_status

    samples = build_dummy_samples(
        model_cfg_dict,
        batch_size=batch_size,
        seq_len=seq_len,
        device="cpu",
    )

    hook_sink: dict[str, Any] = {}

    def _on_pre_pas(graph):
        hook_sink["pre_pas_audit"] = audit_pre_pas_graph(graph)

    def _on_post_pas(graph):
        hook_sink["post_pas_audit"] = audit_post_pas_graph(graph)

    hook_sink["pre_pas_hook"] = _on_pre_pas
    hook_sink["post_pas_hook"] = _on_post_pas

    trace_strategy = "meta" if weightless else "cpu"
    compute_cfg = build_compute_config(
        strategy,
        inference_only=False,
        trace_strategy=trace_strategy,
        weightless=weightless,
    )
    pas_callable = build_pas_policy(strategy, hook_sink=hook_sink)

    try:
        parallelize(
            wrapper,
            {"samples": samples},
            pas_policy=pas_callable,
            compute_config=compute_cfg,
            gen_savedir=str(nnscaler_gen_dir),
            reuse="override",
            load_module=False,
            broadcast_strategy="none",
        )
    except RuntimeError as exc:
        if isinstance(exc.__cause__, StrategyValidationError):
            raise exc.__cause__ from exc
        raise

    elapsed_s = round(time.perf_counter() - t0, 3)
    model_meta["compile_walltime_sec"] = elapsed_s

    manifest = export_verification_bundle(
        out_path,
        model_meta=model_meta,
        model_config_dict=model_cfg_dict,
        strategy_dict=strategy.to_dict(),
        policy_report=hook_sink.get("policy_report", {}),
        pre_pas_audit=hook_sink["pre_pas_audit"],
        post_pas_audit=hook_sink["post_pas_audit"],
        nnscaler_gen_dir=nnscaler_gen_dir,
        plan_ngpus=strategy.mesh.plan_ngpus,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile DeepSeek models into nnScaler distributed verification bundles.")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3",
        help=f"Preset name ({', '.join(list_model_presets())}) or path to HF config.json",
    )
    parser.add_argument("--strategy", type=str, required=True, help="Path to declarative strategy JSON file")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for verification artifact bundle")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-microbatch sequence count")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for static graph tracing")
    parser.add_argument("--num-layers", type=int, default=None, help="Optional override for num_hidden_layers")
    parser.add_argument(
        "--expand-moe-ops",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Optional CLI override for strategy.rewrites.expand_moe_ops (default: false)",
    )
    parser.add_argument("--eager-cpu", action="store_true", help="Materialize real CPU weights instead of weightless meta")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strategy = load_strategy_config(args.strategy)
    if args.expand_moe_ops is not None:
        strategy.rewrites.expand_moe_ops = args.expand_moe_ops == "true"

    manifest = compile_deepseek(
        args.model,
        strategy,
        args.out_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        weightless=not args.eager_cpu,
        num_layers_override=args.num_layers,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "model_class": manifest["model_meta"]["model_class"],
                "compile_walltime_sec": manifest["model_meta"]["compile_walltime_sec"],
                "pre_pas_summary": manifest["pre_pas_summary"],
                "post_pas_summary": manifest["post_pas_summary"],
                "codegen_validation": manifest["codegen_validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
