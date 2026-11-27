# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 5 Zero-GPU Weightless End-to-End Compilation & Static IR/AST Audit Tests for GLM."""

from __future__ import annotations

from pathlib import Path
import tempfile

from examples.GLM.compile import compile_glm
from examples.GLM.policy import StrategyValidationError, load_strategy_config

STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "strategies"


def _assert_clean_bundle(manifest: dict, expected_ranks: int, *, allow_warnings: bool = False) -> None:
    pre = manifest["pre_pas_summary"]
    assert pre["num_opaque_fw_ops"] == 0, f"Unexpected opaque ops: {pre}"
    assert pre["num_dynamic_tensors"] == 0, f"Unexpected dynamic tensors: {pre}"
    assert pre["num_mean_ce_bottlenecks"] == 0, f"Unexpected mean CE bottleneck: {pre}"
    assert pre["audit_passed"] is True

    if not allow_warnings:
        rep = manifest["policy_report"]
        assert rep["warnings"] == [], f"Unexpected partition fallback warnings: {rep['warnings']}"

    cg = manifest["codegen_validation"]
    assert cg["all_ranks_present"] is True
    assert cg["found_gencode_files"] == expected_ranks


def test_strategy_validator_catches_indivisible_heads() -> None:
    bad_strat = load_strategy_config(STRATEGIES_DIR / "glm5_dsa_744b_ep8_pp4.json")
    bad_strat.mesh.pipeline_nstages = 1
    bad_strat.mesh.pipeline_size = 1
    bad_strat.mesh.plan_ngpus = 8
    bad_strat.mesh.runtime_ngpus = 8
    bad_strat.stage_splits = []
    with tempfile.TemporaryDirectory() as td:
        try:
            compile_glm("micro-glm-moe-dsa", bad_strat, td, batch_size=2, seq_len=16, weightless=True)
        except (StrategyValidationError, RuntimeError) as exc:
            err_text = str(exc) + " " + str(exc.__cause__)
            assert "not divisible by tp_size=8" in err_text, err_text
            print("  [PASS] StrategyValidationError caught indivisible MLA heads (4 vs tp_size=8)")
            return
    raise AssertionError("Expected StrategyValidationError for 4 MLA heads on tp_size=8")


def test_compile_glm4_dense_tp4() -> None:
    """Verify full 40-layer GLM-4-9B Dense (interleaved `gate_up_proj` + GQA `fallback_replicate`) on 4 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_glm(
            "glm4-9b",
            STRATEGIES_DIR / "glm4_dense_tp4.json",
            td,
            batch_size=2,
            seq_len=32,
            weightless=True,
        )
        # num_key_value_heads=2 < tp_size=4, so attn_kv_col_tp4 falls back to replicate cleanly
        _assert_clean_bundle(manifest, expected_ranks=4, allow_warnings=True)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["lm_head_vocab_tp4"] == 1
        assert hits["attn_q_col_tp4"] == 40
        assert hits["attn_sdpa_head_tp4"] == 40
        assert hits["attn_o_row_tp4"] == 40
        assert hits["mlp_gate_up_col_tp4"] == 40
        assert hits["mlp_down_row_tp4"] == 40
        print(
            f"  [PASS] Full GLM-4-9B Dense (40L, GQA + Interleaved MLP, TP=4) "
            f"compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_full_glm4_moe_355b_pp4_ep8() -> None:
    """Verify full 46-layer GLM-4.5/4.6 355B-A32B (`PP=4 x EP/TP=8`, 32 GPUs) on 0 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_glm(
            "glm4-moe-355b",
            STRATEGIES_DIR / "glm4_moe_355b_ep8_pp4.json",
            td,
            batch_size=2,
            seq_len=32,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=32)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["lm_head_vocab_tp8"] == 1
        assert hits["attn_sdpa_head_tp8"] == 46
        assert hits["moe_experts_ep8"] == 45
        assert manifest["post_pas_summary"]["num_fw_segments"] == 4
        total_params = manifest["pre_pas_summary"]["total_parameter_elements"]
        assert total_params > 100_000_000_000, f"Expected ~107B params, got {total_params}"
        print(
            f"  [PASS] Full GLM-4.5-Air (46L, 128E, {total_params / 1e9:.2f}B params, "
            f"32 GPUs PP=4xEP/TP=8) compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_glm5_744b_dsa_pp4_ep8() -> None:
    """Verify 744B-width GLM-5 with DSA Sparse Indexer (`PP=4 x EP/TP=8`, 32 GPUs, 8L slice)."""
    strat = load_strategy_config(STRATEGIES_DIR / "glm5_dsa_744b_ep8_pp4.json")
    strat.stage_splits = [
        type(strat.stage_splits[0])(layers="0..1", stage_id=0),
        type(strat.stage_splits[0])(layers="2..3", stage_id=1),
        type(strat.stage_splits[0])(layers="4..5", stage_id=2),
        type(strat.stage_splits[0])(layers="6..7", stage_id=3),
    ]
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_glm(
            "glm5-744b-dsa",
            strat,
            td,
            batch_size=2,
            seq_len=32,
            weightless=True,
            num_layers_override=8,
        )
        _assert_clean_bundle(manifest, expected_ranks=32)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["dsa_indexer_mask_rep"] == 8
        assert hits["mla_concat_kv_head_tp8"] == 8
        assert hits["mla_sdpa_head_tp8"] == 8
        assert hits["moe_experts_ep8"] == 5  # first_k_dense_replace=3 -> layers 3..7
        assert manifest["post_pas_summary"]["num_fw_segments"] == 4
        print(
            f"  [PASS] GLM-5 744B-width DSA (256E, 8L PP=4xEP/TP=8, 32 GPUs) "
            f"compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_micro_glm5_next_hybrid_ep4() -> None:
    """Verify GLM-5-Next Hybrid KDA + NoPE-MLA/DSA + HyperConnections + MoE on 4 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_glm(
            "micro-glm5-next",
            STRATEGIES_DIR / "glm5_next_hybrid_ep4.json",
            td,
            batch_size=2,
            seq_len=16,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=4)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["kda_conv1d_head_tp4"] == 9  # 3 KDA layers * 3
        assert hits["kda_chunk_head_tp4"] == 3
        assert hits["dsa_indexer_mask_rep"] == 1
        assert hits["mla_sdpa_head_tp4"] == 1
        assert hits["moe_experts_ep4"] == 3
        print("  [PASS] micro-glm5-next (3 KDA + 1 NoPE-MLA/DSA + HyperConnections + MoE, EP=4/TP=4) compiled cleanly")


if __name__ == "__main__":
    print("=== Running GLM Zero-GPU Weightless Compilation Suite ===")
    test_strategy_validator_catches_indivisible_heads()
    test_compile_micro_glm5_next_hybrid_ep4()
    test_compile_glm4_dense_tp4()
    test_compile_glm5_744b_dsa_pp4_ep8()
    test_compile_full_glm4_moe_355b_pp4_ep8()
    print("=== ALL GLM ZERO-GPU COMPILATION TESTS PASSED ===")
