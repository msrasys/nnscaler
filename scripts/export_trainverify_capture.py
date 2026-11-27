# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Zero-GPU TrainVerify Capture Exporter for Qwen3.5, DeepSeek, Kimi, and GLM.

Generates `sm_mgener.pkl` (1-GPU reference) and `pm_mgener.pkl` (N-GPU parallel)
using `trace_strategy="meta"` and `ParamInitStrategy.SHARD` into
`TrainVerify_Lean_dev/trainverify_capture/captures/<capture_name>/` so `scripts.gen`
and `autoprove` can run without GPU capture servers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile

import dill

from nnscaler.codegen import ModuleCodeGen
from nnscaler.parallel import ComputeConfig, ParamInitStrategy, parallelize


@contextmanager
def _capture_mgener(out_pkl: Path):
    orig_init = ModuleCodeGen.__init__

    def _wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_pkl, "wb") as f:
            dill.dump(self, f)

    ModuleCodeGen.__init__ = _wrapped_init
    try:
        yield
    finally:
        ModuleCodeGen.__init__ = orig_init


def _build_family(family: str, preset: str, strategy_path: Path, num_layers_override: int | None, batch_size: int, seq_len: int):
    if family == "glm":
        from examples.GLM.configs import build_glm_model, build_glm_sample
        from examples.GLM.policy import StrategyConfig, build_json_pas, load_strategy_config
        from examples.GLM.rewrites import GlmCausalLMTaskWrapper, RewriteConfig, apply_glm_rewrites

        strat = load_strategy_config(strategy_path)
        rw_cfg = RewriteConfig(
            expand_moe_ops=strat.rewrites.expand_moe_ops,
            sum_ce_loss=strat.rewrites.sum_ce_loss,
            decouple_row_tp_bias=strat.rewrites.decouple_row_tp_bias,
        )
        raw_sm = build_glm_model(preset, weightless=True, num_layers_override=num_layers_override)
        apply_glm_rewrites(raw_sm, rw_cfg)
        sm_mod = GlmCausalLMTaskWrapper(raw_sm, sum_ce_loss=rw_cfg.sum_ce_loss)

        raw_pm = build_glm_model(preset, weightless=True, num_layers_override=num_layers_override)
        apply_glm_rewrites(raw_pm, rw_cfg)
        pm_mod = GlmCausalLMTaskWrapper(raw_pm, sum_ce_loss=rw_cfg.sum_ce_loss)

        sample = build_glm_sample(preset, batch_size=batch_size, seq_len=seq_len)

        sm_strat = StrategyConfig(name=f"{strat.name}_sm", rewrites=rw_cfg)
        sm_strat.mesh.plan_ngpus = 1
        sm_strat.mesh.runtime_ngpus = 1
        sm_strat.mesh.pipeline_nstages = 1
        sm_strat.mesh.pipeline_size = 1
        sm_strat.mesh.pipeline_nmicros = 1
        sm_strat.mesh.zero_stage = 0
        return sm_mod, pm_mod, sample, build_json_pas(sm_strat), build_json_pas(strat), strat

    if family == "deepseek":
        from examples.Deepseek.configs import build_dummy_samples, build_deepseek_model, load_model_config_dict
        from examples.Deepseek.policy import build_pas_policy, load_strategy_config

        strat = load_strategy_config(strategy_path)
        cfg_dict = load_model_config_dict(preset)
        if num_layers_override is not None:
            cfg_dict["num_hidden_layers"] = int(num_layers_override)
        sm_mod, _ = build_deepseek_model(cfg_dict, rewrite_cfg=strat.rewrites, weightless=True)
        pm_mod, _ = build_deepseek_model(cfg_dict, rewrite_cfg=strat.rewrites, weightless=True)
        sample = build_dummy_samples(cfg_dict, batch_size=batch_size, seq_len=seq_len, device="cpu")
        sm_strat = copy.deepcopy(strat)
        sm_strat.mesh.plan_ngpus = 1
        sm_strat.mesh.runtime_ngpus = 1
        sm_strat.mesh.pipeline_nstages = 1
        sm_strat.mesh.pipeline_size = 1
        sm_strat.mesh.pipeline_nmicros = 1
        sm_strat.mesh.zero_stage = 0
        sm_strat.stage_splits = []
        sm_strat.rules = []
        return sm_mod, pm_mod, sample, build_pas_policy(sm_strat), build_pas_policy(strat), strat

    if family == "kimi":
        from examples.Kimi.configs import build_kimi_model, build_kimi_sample
        from examples.Kimi.policy import StrategyConfig, build_json_pas, load_strategy_config
        from examples.Kimi.rewrites import KimiCausalLMTaskWrapper, RewriteConfig, apply_kimi_rewrites

        strat = load_strategy_config(strategy_path)
        rw_cfg = RewriteConfig(
            expand_moe_ops=strat.rewrites.expand_moe_ops,
            sum_ce_loss=strat.rewrites.sum_ce_loss,
            decouple_row_tp_bias=strat.rewrites.decouple_row_tp_bias,
        )
        raw_sm = build_kimi_model(preset, weightless=True, num_layers_override=num_layers_override)
        apply_kimi_rewrites(raw_sm, rw_cfg)
        sm_mod = KimiCausalLMTaskWrapper(raw_sm, sum_ce_loss=rw_cfg.sum_ce_loss)

        raw_pm = build_kimi_model(preset, weightless=True, num_layers_override=num_layers_override)
        apply_kimi_rewrites(raw_pm, rw_cfg)
        pm_mod = KimiCausalLMTaskWrapper(raw_pm, sum_ce_loss=rw_cfg.sum_ce_loss)

        sample = build_kimi_sample(preset, batch_size=batch_size, seq_len=seq_len)
        sm_strat = StrategyConfig(name=f"{strat.name}_sm", rewrites=rw_cfg)
        sm_strat.mesh.plan_ngpus = 1
        sm_strat.mesh.runtime_ngpus = 1
        sm_strat.mesh.pipeline_nstages = 1
        sm_strat.mesh.pipeline_size = 1
        sm_strat.mesh.pipeline_nmicros = 1
        sm_strat.mesh.zero_stage = 0
        return sm_mod, pm_mod, sample, build_json_pas(sm_strat), build_json_pas(strat), strat

    if family == "qwen3p5":
        from examples.qwen3p5.configs import build_qwen3p5_model, build_dummy_samples, load_model_config_dict
        from examples.qwen3p5.policy import build_pas_policy, load_strategy_config

        strat = load_strategy_config(strategy_path)
        cfg_dict = load_model_config_dict(preset)
        if num_layers_override is not None:
            cfg_dict["num_hidden_layers"] = int(num_layers_override)
        sm_mod, _ = build_qwen3p5_model(cfg_dict, rewrite_cfg=strat.rewrites, weightless=True)
        pm_mod, _ = build_qwen3p5_model(cfg_dict, rewrite_cfg=strat.rewrites, weightless=True)
        sample = build_dummy_samples(cfg_dict, batch_size=batch_size, seq_len=seq_len, device="cpu")
        sm_strat = copy.deepcopy(strat)
        sm_strat.mesh.plan_ngpus = 1
        sm_strat.mesh.runtime_ngpus = 1
        sm_strat.mesh.pipeline_nstages = 1
        sm_strat.mesh.pipeline_size = 1
        sm_strat.mesh.pipeline_nmicros = 1
        sm_strat.mesh.zero_stage = 0
        sm_strat.stage_splits = []
        sm_strat.rules = []
        return sm_mod, pm_mod, sample, build_pas_policy(sm_strat), build_pas_policy(strat), strat

    raise ValueError(f"Unknown family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export zero-GPU SM/PM ModuleCodeGen pickles for TrainVerify_Lean")
    parser.add_argument("--family", required=True, choices=("glm", "deepseek", "kimi", "qwen3p5"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--capture-name", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-layers-override", type=int, default=None)
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=Path("/home/v-xuasong/nnscaler/TrainVerify_Lean_dev/trainverify_capture/captures"),
    )
    args = parser.parse_args()

    sm_mod, pm_mod, sample, sm_pas, pm_pas, strat = _build_family(
        args.family, args.model, args.strategy, args.num_layers_override, args.batch_size, args.seq_len
    )
    cap_dir = args.captures_root / args.capture_name
    cap_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        sm_cfg = ComputeConfig(
            plan_ngpus=1,
            runtime_ngpus=1,
            use_end2end=True,
            use_zero=False,
            trace_strategy="meta",
            param_init_strategy=ParamInitStrategy.SHARD,
        )
        with _capture_mgener(cap_dir / "sm_mgener.pkl"):
            parallelize(
                sm_mod, {"samples": sample}, sm_pas, sm_cfg,
                gen_savedir=str(Path(td) / "sm"), load_module=False, broadcast_strategy="none", reuse="override",
            )

        pm_mesh = strat.mesh
        pm_cfg = ComputeConfig(
            plan_ngpus=pm_mesh.plan_ngpus,
            runtime_ngpus=pm_mesh.resolved_runtime_ngpus(),
            use_end2end=True,
            use_zero=False,
            trace_strategy="meta",
            param_init_strategy=ParamInitStrategy.SHARD,
        )
        with _capture_mgener(cap_dir / "pm_mgener.pkl"):
            parallelize(
                pm_mod, {"samples": sample}, pm_pas, pm_cfg,
                gen_savedir=str(Path(td) / "pm"), load_module=False, broadcast_strategy="none", reuse="override",
            )

    print(f"Exported SM & PM ModuleCodeGen pickles to {cap_dir}")


if __name__ == "__main__":
    main()
