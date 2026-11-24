"""End-to-end zero-GPU weightless (`meta`) compilation tests across the Qwen3.5 family (`0.8B` .. `397B-A17B` + VL)."""

from __future__ import annotations

from pathlib import Path

from examples.qwen3p5.compile import compile_qwen3_5
from examples.qwen3p5.policy.json_pas import StrategyValidationError, load_strategy_config


STRAT_DIR = Path(__file__).resolve().parents[1] / "strategies"


def test_compile_dense_tp2_and_pp2_tp2(tmp_path: Path) -> None:
    # 1. `0.8B` with 2-stage PP x TP=2 (4 GPUs)
    manifest = compile_qwen3_5(
        "qwen3.5-0.8b",
        STRAT_DIR / "dense_pp2_tp2.json",
        tmp_path / "qwen3_5_0p8b_pp2_tp2",
        batch_size=4,
        seq_len=64,
        weightless=True,
    )
    assert manifest["pre_pas_summary"]["audit_passed"] is True
    assert manifest["pre_pas_summary"]["num_opaque_fw_ops"] == 0
    assert manifest["pre_pas_summary"]["num_dynamic_tensors"] == 0
    assert manifest["pre_pas_summary"]["num_mean_ce_bottlenecks"] == 0
    assert manifest["post_pas_summary"]["num_fw_segments"] == 2
    assert manifest["codegen_validation"]["all_ranks_present"] is True
    assert manifest["codegen_validation"]["found_gencode_files"] == 4


def test_compile_moe_fused_default_and_expanded_opt_in(tmp_path: Path) -> None:
    # Default `expand_moe_ops=False` on `micro-moe` with EP=4
    m_fused = compile_qwen3_5(
        "micro-moe",
        STRAT_DIR / "moe_ep4_tp1.json",
        tmp_path / "moe_fused",
        batch_size=4,
        seq_len=32,
        weightless=True,
    )
    assert m_fused["pre_pas_summary"]["audit_passed"] is True
    assert m_fused["codegen_validation"]["found_gencode_files"] == 4
    # micro-moe has linear_num_key_heads=2 < tp_size=4 -> fallback_replicate warnings recorded cleanly
    assert len(m_fused["policy_report"]["warnings"]) > 0

    # Opt-in `expand_moe_ops=True` on `micro-moe` with EP=2
    m_exp = compile_qwen3_5(
        "micro-moe",
        STRAT_DIR / "moe_expanded_ep2.json",
        tmp_path / "moe_expanded",
        batch_size=2,
        seq_len=32,
        weightless=True,
    )
    assert m_exp["pre_pas_summary"]["audit_passed"] is True
    assert m_exp["codegen_validation"]["found_gencode_files"] == 2


def test_compile_frontier_397b_a17b_weightless_on_cpu(tmp_path: Path) -> None:
    """Compile the flagship 397B-A17B MoE architecture (512 experts, 60 layers, 32 GPUs: PP=4 x EP/TP=8) on 0 GPUs!"""
    manifest = compile_qwen3_5(
        "qwen3.5-397b-a17b",
        STRAT_DIR / "moe_ep8_pp4.json",
        tmp_path / "qwen3_5_397b_a17b_ep8_pp4",
        batch_size=8,
        seq_len=128,
        weightless=True,
    )
    assert manifest["pre_pas_summary"]["audit_passed"] is True
    assert manifest["pre_pas_summary"]["num_opaque_fw_ops"] == 0
    assert manifest["pre_pas_summary"]["num_dynamic_tensors"] == 0
    assert manifest["post_pas_summary"]["num_fw_segments"] == 4
    assert manifest["codegen_validation"]["all_ranks_present"] is True
    assert manifest["codegen_validation"]["found_gencode_files"] == 32
    hits = manifest["policy_report"]["rule_hit_counts"]
    assert hits["moe_routed_experts_ep8"] == 60
    assert hits["gdn_conv_and_fla_head_tp8"] == 90
    assert hits["attn_sdpa_head_tp8"] == 15
    assert hits["vocab_parallel_lm_head"] == 1


def test_compile_multimodal_vl_tp2(tmp_path: Path) -> None:
    manifest = compile_qwen3_5(
        "micro-vl",
        STRAT_DIR / "vl_tp2.json",
        tmp_path / "vl_tp2",
        batch_size=2,
        seq_len=32,
        multimodal=True,
        weightless=True,
    )
    assert manifest["pre_pas_summary"]["audit_passed"] is True
    assert manifest["codegen_validation"]["all_ranks_present"] is True
    assert manifest["codegen_validation"]["found_gencode_files"] == 2


def test_strategy_validator_catches_strict_indivisible_gqa_heads(tmp_path: Path) -> None:
    strat = load_strategy_config(STRAT_DIR / "moe_ep4_tp1.json")
    # Force strict `on_indivisible="error"` on `gdn_conv_and_fla_head_tp` where `micro-moe` has `linear_num_key_heads=2 < tp_size=4`
    for r in strat.rules:
        if r.name == "gdn_conv_and_fla_head_tp":
            r.on_indivisible = "error"

    caught = False
    try:
        compile_qwen3_5(
            "micro-moe",
            strat,
            tmp_path / "should_fail",
            batch_size=2,
            seq_len=16,
            weightless=True,
        )
    except StrategyValidationError as exc:
        assert "not divisible by tp_size=4" in str(exc)
        caught = True
    assert caught, "Expected StrategyValidationError when tp_size=4 > num_key_value_heads=2"
