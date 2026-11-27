# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Zero-GPU Weightless Compilation & Verification CLI for the GLM model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from nnscaler.parallel import ComputeConfig, ParamInitStrategy, parallelize

from examples.GLM.audit import (
    audit_codegen_dir,
    audit_post_pas,
    audit_pre_pas,
    dump_ir_text,
)
from examples.GLM.configs import build_glm_model, build_glm_sample, list_presets
from examples.GLM.policy import StrategyConfig, build_json_pas, get_last_policy_report, load_strategy_config
from examples.GLM.rewrites import GlmCausalLMTaskWrapper, RewriteConfig, apply_glm_rewrites


def compile_glm(
    preset: str,
    strategy_or_path: str | Path | StrategyConfig,
    out_dir: str | Path,
    *,
    batch_size: int = 2,
    seq_len: int = 32,
    weightless: bool = True,
    expand_moe_ops: bool | None = None,
    num_layers_override: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    strategy = (
        strategy_or_path
        if isinstance(strategy_or_path, StrategyConfig)
        else load_strategy_config(strategy_or_path)
    )
    rw_cfg = RewriteConfig(
        expand_moe_ops=strategy.rewrites.expand_moe_ops if expand_moe_ops is None else expand_moe_ops,
        sum_ce_loss=strategy.rewrites.sum_ce_loss,
        decouple_row_tp_bias=strategy.rewrites.decouple_row_tp_bias,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gen_dir = out_path / "04_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    raw_model = build_glm_model(preset, weightless=weightless, num_layers_override=num_layers_override)
    rewrite_counts = apply_glm_rewrites(raw_model, rw_cfg)
    wrapped = GlmCausalLMTaskWrapper(raw_model, sum_ce_loss=rw_cfg.sum_ce_loss)
    sample = build_glm_sample(preset, batch_size=batch_size, seq_len=seq_len)

    mesh = strategy.mesh
    compute_config = ComputeConfig(
        plan_ngpus=mesh.plan_ngpus,
        runtime_ngpus=mesh.resolved_runtime_ngpus(),
        use_end2end=True,
        use_zero=mesh.zero_stage > 0,
        zero_ngroups=1,
        trace_strategy="meta" if weightless else "cpu",
        param_init_strategy=ParamInitStrategy.SHARD if weightless else ParamInitStrategy.FULL,
        pas_config={
            "pipeline_nstages": mesh.pipeline_nstages,
            "pipeline_size": mesh.resolved_pipeline_size(),
            "pipeline_nmicros": mesh.pipeline_nmicros,
            "pipeline_scheduler": mesh.pipeline_scheduler,
        },
    )

    audit_sink: dict[str, Any] = {}

    def _on_pre_pas(graph: Any) -> None:
        dump_ir_text(graph, out_path / "02_ir_pre_pas.txt")
        pre_audit = audit_pre_pas(graph)
        (out_path / "02_ir_audit.json").write_text(json.dumps(pre_audit, indent=2), encoding="utf-8")
        audit_sink["pre_audit"] = pre_audit

    def _on_post_pas(graph: Any) -> None:
        dump_ir_text(graph, out_path / "03_ir_post_pas.txt")
        post_audit = audit_post_pas(graph)
        (out_path / "03_adapters.json").write_text(json.dumps(post_audit, indent=2), encoding="utf-8")
        audit_sink["post_audit"] = post_audit

    pas_fn = build_json_pas(strategy, pre_pas_hook=_on_pre_pas, post_pas_hook=_on_post_pas)
    parallelize(
        wrapped,
        {"samples": sample},
        pas_fn,
        compute_config,
        gen_savedir=str(gen_dir),
        load_module=False,
        broadcast_strategy="none",
        reuse="override",
    )
    pre_audit = audit_sink["pre_audit"]
    post_audit = audit_sink["post_audit"]

    codegen_audit = audit_codegen_dir(gen_dir, expected_ranks=mesh.plan_ngpus)
    policy_report = get_last_policy_report()
    elapsed = round(time.perf_counter() - t0, 3)

    manifest = {
        "status": "ok",
        "model_meta": {
            "preset": preset,
            "model_class": type(raw_model).__name__,
            "weightless_meta": weightless,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "compile_walltime_sec": elapsed,
            "rewrite_counts": rewrite_counts,
        },
        "strategy": strategy.to_dict(),
        "policy_report": policy_report,
        "pre_pas_summary": pre_audit["summary"],
        "post_pas_summary": post_audit["summary"],
        "codegen_validation": codegen_audit,
    }
    (out_path / "00_model_and_strategy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-GPU nnScaler compiler for the GLM family")
    parser.add_argument("--model", required=True, choices=list_presets())
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--expand-moe-ops", choices=("true", "false"), default=None)
    parser.add_argument("--num-layers-override", type=int, default=None)
    args = parser.parse_args()

    expand = None if args.expand_moe_ops is None else (args.expand_moe_ops == "true")
    manifest = compile_glm(
        args.model,
        args.strategy,
        args.out_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        weightless=True,
        expand_moe_ops=expand,
        num_layers_override=args.num_layers_override,
    )
    print(json.dumps({
        "status": manifest["status"],
        "model_class": manifest["model_meta"]["model_class"],
        "compile_walltime_sec": manifest["model_meta"]["compile_walltime_sec"],
        "pre_pas_summary": manifest["pre_pas_summary"],
        "post_pas_summary": manifest["post_pas_summary"],
        "codegen_validation": {
            "expected_ranks": manifest["codegen_validation"]["expected_ranks"],
            "found_gencode_files": manifest["codegen_validation"]["found_gencode_files"],
            "all_ranks_present": manifest["codegen_validation"]["all_ranks_present"],
            "total_collective_calls_across_ranks": manifest["codegen_validation"]["total_collective_calls_across_ranks"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
