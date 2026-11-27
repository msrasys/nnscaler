# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 5 Zero-GPU Weightless End-to-End Compilation & Static IR/AST Audit Tests for Kimi."""

from __future__ import annotations

from pathlib import Path
import tempfile

from examples.Kimi.compile import compile_kimi
from examples.Kimi.policy import StrategyValidationError, load_strategy_config

STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "strategies"


def _assert_clean_bundle(manifest: dict, expected_ranks: int) -> None:
    pre = manifest["pre_pas_summary"]
    assert pre["num_opaque_fw_ops"] == 0, f"Unexpected opaque ops: {pre}"
    assert pre["num_dynamic_tensors"] == 0, f"Unexpected dynamic tensors: {pre}"
    assert pre["num_mean_ce_bottlenecks"] == 0, f"Unexpected mean CE bottleneck: {pre}"
    assert pre["audit_passed"] is True

    rep = manifest["policy_report"]
    assert rep["warnings"] == [], f"Unexpected partition fallback warnings: {rep['warnings']}"

    cg = manifest["codegen_validation"]
    assert cg["all_ranks_present"] is True
    assert cg["found_gencode_files"] == expected_ranks


def test_strategy_validator_catches_indivisible_heads() -> None:
    bad_strat = load_strategy_config(STRATEGIES_DIR / "kimi_linear_48b_ep8_tp8.json")
    with tempfile.TemporaryDirectory() as td:
        try:
            compile_kimi("micro-kimi-linear", bad_strat, td, batch_size=2, seq_len=16, weightless=True)
        except (StrategyValidationError, RuntimeError) as exc:
            err_text = str(exc) + " " + str(exc.__cause__)
            assert "not divisible by tp_size=8" in err_text, err_text
            print("  [PASS] StrategyValidationError caught indivisible KDA/MLA heads (4 vs tp_size=8)")
            return
    raise AssertionError("Expected StrategyValidationError for 4 heads on tp_size=8")


def test_compile_micro_kimi_linear_expanded_etp4() -> None:
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_kimi(
            "micro-kimi-linear",
            STRATEGIES_DIR / "kimi_linear_expanded_etp4.json",
            td,
            batch_size=2,
            seq_len=16,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=4)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["kda_conv1d_head_tp4"] == 9  # 3 KDA layers * 3 (Q, K, V)
        assert hits["kda_chunk_head_tp4"] == 3
        assert hits["mla_concat_kv_head_tp4"] == 1
        assert hits["moe_expanded_down_etp_row"] == 3
        assert hits["moe_expanded_gate_up_etp_col"] == 3
        print("  [PASS] micro-kimi-linear (3 KDA + 1 MLA, expand_moe=True, ETP=4) compiled cleanly")


def test_compile_full_kimi_linear_48b_ep8_tp8() -> None:
    """Full 27-layer Kimi-Linear-48B-A3B (20 KDA + 7 MLA, 256 experts) on 8 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_kimi(
            "kimi-linear-48b",
            STRATEGIES_DIR / "kimi_linear_48b_ep8_tp8.json",
            td,
            batch_size=2,
            seq_len=64,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=8)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["lm_head_vocab_tp"] == 1
        assert hits["kda_conv1d_head_tp"] == 21 * 3  # 21 KDA layers (layers 0..26 except 3,7,11,15,19,23)
        assert hits["kda_chunk_head_tp"] == 21
        assert hits["mla_concat_kv_head_tp"] == 6
        assert hits["mla_sdpa_head_tp"] == 6
        assert hits["moe_experts_ep"] == 26
        total_params = manifest["pre_pas_summary"]["total_parameter_elements"]
        assert total_params > 40_000_000_000, f"Expected ~48B params, got {total_params}"
        print(
            f"  [PASS] Full Kimi-Linear-48B-A3B (27L: 21 KDA + 6 MLA, 256E, {total_params / 1e9:.2f}B params) "
            f"EP=8/TP=8 compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_full_kimi_linear_48b_pp2_ep4() -> None:
    """Full 27-layer Kimi-Linear-48B-A3B in PP=2 x EP/TP=4 on 8 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_kimi(
            "kimi-linear-48b",
            STRATEGIES_DIR / "kimi_linear_48b_ep4_pp2.json",
            td,
            batch_size=2,
            seq_len=64,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=8)
        assert manifest["post_pas_summary"]["num_fw_segments"] == 2
        print(
            f"  [PASS] Full Kimi-Linear-48B-A3B (27L, PP=2 x EP/TP=4, 8 GPUs) "
            f"compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_micro_kimi_k25_multimodal_tp4_ep4() -> None:
    """Multimodal Kimi-K2.5 VLM (ViT + DeepSeek-V3 MLA/MoE) compiled with pixel_values + image_grid_thw."""
    strat = load_strategy_config(STRATEGIES_DIR / "kimi_k25_ep8_pp4.json")
    strat.mesh.plan_ngpus = 4
    strat.mesh.runtime_ngpus = 4
    strat.mesh.pipeline_nstages = 1
    strat.mesh.pipeline_size = 1
    strat.mesh.pipeline_nmicros = 1
    strat.stage_splits = []
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_kimi(
            "micro-kimi-k25",
            strat,
            td,
            batch_size=2,
            seq_len=16,
            multimodal=True,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=4)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["vision_varlen_sdpa_head_tp8"] == 2
        assert hits["mla_concat_kv_head_tp8"] == 2
        assert hits["moe_experts_ep8"] == 1
        print("  [PASS] micro-kimi-k25 Multimodal VLM (ViT + DeepSeek-V3 MLA/MoE, TP=4/EP=4) compiled cleanly")


def test_compile_frontier_kimi_k25_1t_scale_pp4_ep8() -> None:
    """1T-class Kimi-K2.5 VLM (27L ViT + 384-expert DeepSeek-V3 text, 8L slice, PP=4 x EP/TP=8, 32 GPUs)."""
    strat = load_strategy_config(STRATEGIES_DIR / "kimi_k25_ep8_pp4.json")
    strat.stage_splits = [
        type(strat.stage_splits[0])(layers="0..1", stage_id=0),
        type(strat.stage_splits[0])(layers="2..3", stage_id=1),
        type(strat.stage_splits[0])(layers="4..5", stage_id=2),
        type(strat.stage_splits[0])(layers="6..7", stage_id=3),
    ]
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_kimi(
            "kimi-k2.5",
            strat,
            td,
            batch_size=2,
            seq_len=32,
            multimodal=True,
            weightless=True,
            num_layers_override=8,
        )
        _assert_clean_bundle(manifest, expected_ranks=32)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["vision_varlen_sdpa_head_tp8"] == 27
        assert hits["mla_concat_kv_head_tp8"] == 8
        assert hits["moe_experts_ep8"] == 7
        assert manifest["post_pas_summary"]["num_fw_segments"] == 4
        print(
            f"  [PASS] Kimi-K2.5 1T-class VLM (27L ViT + 8L 384E text, PP=4xEP/TP=8, 32 GPUs) "
            f"compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


if __name__ == "__main__":
    print("=== Running Kimi Zero-GPU Weightless Compilation Suite ===")
    test_strategy_validator_catches_indivisible_heads()
    test_compile_micro_kimi_linear_expanded_etp4()
    test_compile_micro_kimi_k25_multimodal_tp4_ep4()
    test_compile_full_kimi_linear_48b_ep8_tp8()
    test_compile_full_kimi_linear_48b_pp2_ep4()
    test_compile_frontier_kimi_k25_1t_scale_pp4_ep8()
    print("=== ALL KIMI ZERO-GPU COMPILATION TESTS PASSED ===")
