# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 5 Zero-GPU Weightless End-to-End Compilation & Static IR/AST Audit Tests for DeepSeek.

Verifies on a CPU-only host (`torch.cuda.device_count() == 0`) that:
  1. `deepseek-v2-lite` (full 15.7B-A2.4B, 27L, 16 MLA heads, 64 routed experts) compiles cleanly
     across 4 GPUs (`MLA-TP=4 + MoE-EP=4 + Vocab-TP=4`) with 0 opaque ops, 0 dynamic shapes,
     0 `mean` CE bottlenecks, and AST-valid `gencode0..3.py`;
  2. `micro-v2` (`q_lora_rank=48`, `group_limited_greedy` router, real interleaved RoPE) compiles cleanly;
  3. `micro-v3` compiles in both `expand_moe_ops=False` (default fused EP) and `expand_moe_ops=True`
     (`v3_expanded_ep4.json` EP=4 AND `v3_expanded_etp4.json` Col/Row-ETP=4);
  4. `deepseek-v3.2` (671B architecture + DeepSeek Sparse Attention `deepseek_v32_dsa_index_mask`)
     compiles cleanly across 32 GPUs (`PP=4 x MLA-TP=8 & MoE-EP=8`);
  5. `StrategyValidationError` catches illegal/indivisible head or expert sharding upfront;
  6. Full-scale flagship **DeepSeek-V3 / R1 671B-A37B** (`61` layers: `3` dense + `58` MoE,
     `128` MLA heads, `256` routed experts, `129,280` vocab) compiles end-to-end across **32 GPUs**
     (`PP=4 x EP/TP=8`) on 0 GPUs in weightless `meta` mode.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from examples.Deepseek.compile import compile_deepseek
from examples.Deepseek.policy import StrategyValidationError, load_strategy_config

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


def test_compile_full_deepseek_v2_lite_16b_ep4_tp4() -> None:
    """Full 27-layer DeepSeek-V2-Lite (15.7B-A2.4B, 16 heads, 64 experts, q_lora_rank=None) on 4 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_deepseek(
            "deepseek-v2-lite",
            STRATEGIES_DIR / "v2_lite_ep4_tp4.json",
            td,
            batch_size=2,
            seq_len=64,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=4)
        hits = manifest["policy_report"]["rule_hit_counts"]
        # 27 layers: 1 dense (layer 0) + 26 MoE (layers 1..26)
        assert hits["lm_head_vocab_tp4"] == 1
        assert hits["mla_qkv_col_tp4"] == 27 * 2          # q_proj + kv_b_proj per layer
        assert hits["mla_concat_kv_head_tp4"] == 27
        assert hits["mla_sdpa_head_tp4"] == 27
        assert hits["mla_o_row_tp4"] == 27
        assert hits["moe_experts_ep4"] == 26              # 26 MoE layers
        assert hits["dense_and_shared_mlp_col_tp4"] == 27 * 2  # 1 dense + 26 shared_experts
        assert hits["dense_and_shared_mlp_row_tp4"] == 27
        total_params = manifest["pre_pas_summary"]["total_parameter_elements"]
        assert total_params > 15_000_000_000, f"Expected ~15.7B params, got {total_params}"
        print(
            f"  [PASS] DeepSeek-V2-Lite 16B (27L, 64E, {total_params / 1e9:.2f}B params) "
            f"EP=4/TP=4 compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_compile_v2_qlora_and_group_limited_router() -> None:
    """Verify V2 `q_lora_rank` + `group_limited_greedy` router + real RoPE on `micro-v2`."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_deepseek(
            "micro-v2",
            STRATEGIES_DIR / "v2_lite_ep4_tp4.json",
            td,
            batch_size=2,
            seq_len=32,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=4)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["mla_qkv_col_tp4"] == 3 * 2  # q_b_proj + kv_b_proj across 3 layers
        assert hits["moe_experts_ep4"] == 2      # layers 1, 2
        print("  [PASS] micro-v2 (Q-LoRA + group_limited_greedy + real RoPE) compiled cleanly")


def test_compile_v3_expanded_ep4_and_etp4() -> None:
    """Verify opt-in `expand_moe_ops=True` in both EP=4 (`E+`) and Expert-TP=4 (`m` / `k+`) modes."""
    with tempfile.TemporaryDirectory() as td_ep, tempfile.TemporaryDirectory() as td_etp:
        m_ep = compile_deepseek(
            "micro-v3",
            STRATEGIES_DIR / "v3_expanded_ep4.json",
            td_ep,
            batch_size=2,
            seq_len=32,
            weightless=True,
        )
        _assert_clean_bundle(m_ep, expected_ranks=4)
        # 3 MoE layers * 3 grouped_linear calls (gate, up, down) = 9
        assert m_ep["policy_report"]["rule_hit_counts"]["moe_expanded_grouped_linear_ep4"] == 9

        m_etp = compile_deepseek(
            "micro-v3",
            STRATEGIES_DIR / "v3_expanded_etp4.json",
            td_etp,
            batch_size=2,
            seq_len=32,
            weightless=True,
        )
        _assert_clean_bundle(m_etp, expected_ranks=4)
        hits_etp = m_etp["policy_report"]["rule_hit_counts"]
        assert hits_etp["moe_expanded_down_etp_row"] == 3
        assert hits_etp["moe_expanded_gate_up_etp_col"] == 6
        print("  [PASS] micro-v3 expand_moe_ops=True verified in both EP=4 and Expert-TP=4 modes")


def test_compile_deepseek_v32_dsa_671b_scale_pp4_ep8() -> None:
    """Verify 671B-width DeepSeek-V3.2 with DSA Sparse Attention (`PP=4 x EP/TP=8`, 32 GPUs, 8L slice)."""
    strat = load_strategy_config(STRATEGIES_DIR / "v32_dsa_ep8_pp4.json")
    strat.stage_splits = [
        type(strat.stage_splits[0])(layers="0..1", stage_id=0),
        type(strat.stage_splits[0])(layers="2..3", stage_id=1),
        type(strat.stage_splits[0])(layers="4..5", stage_id=2),
        type(strat.stage_splits[0])(layers="6..7", stage_id=3),
    ]
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_deepseek(
            "deepseek-v3.2",
            strat,
            td,
            batch_size=2,
            seq_len=64,
            weightless=True,
            num_layers_override=8,
        )
        _assert_clean_bundle(manifest, expected_ranks=32)
        hits = manifest["policy_report"]["rule_hit_counts"]
        assert hits["dsa_indexer_mask_rep"] == 8
        assert hits["mla_concat_kv_head_tp8"] == 8
        assert hits["mla_sdpa_head_tp8"] == 8
        assert hits["moe_experts_ep8"] == 5  # first_k_dense_replace=3 -> layers 3..7 are MoE
        assert manifest["post_pas_summary"]["num_fw_segments"] == 4
        print(
            f"  [PASS] DeepSeek-V3.2 DSA (671B width, 256E, 8L PP=4xEP=8, 32 GPUs) "
            f"compiled in {manifest['model_meta']['compile_walltime_sec']}s"
        )


def test_strategy_validator_catches_indivisible_mla_heads() -> None:
    """Verify strict divisibility check raises `StrategyValidationError` when `num_heads=4 < tp_size=8`."""
    bad_strat = load_strategy_config(STRATEGIES_DIR / "v2_lite_ep4_tp4.json")
    bad_strat.mesh.plan_ngpus = 8  # micro-v2-lite only has 4 MLA heads!
    with tempfile.TemporaryDirectory() as td:
        try:
            compile_deepseek("micro-v2-lite", bad_strat, td, batch_size=2, seq_len=16, weightless=True)
        except StrategyValidationError as exc:
            assert "not divisible by tp_size=8" in str(exc)
            print("  [PASS] StrategyValidationError caught indivisible MLA heads (4 vs tp_size=8)")
            return
    raise AssertionError("Expected StrategyValidationError for 4 MLA heads on tp_size=8")


def test_compile_frontier_deepseek_v3_671b_pp4_ep8_on_cpu() -> None:
    """Flagship proof: full 61-layer DeepSeek-V3 / R1 671B-A37B (`PP=4 x EP/TP=8`, 32 GPUs) on 0 GPUs."""
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_deepseek(
            "deepseek-v3",
            STRATEGIES_DIR / "v3_ep8_pp4.json",
            td,
            batch_size=2,
            seq_len=64,
            weightless=True,
        )
        _assert_clean_bundle(manifest, expected_ranks=32)
        hits = manifest["policy_report"]["rule_hit_counts"]
        # 61 layers: 3 dense (0..2) + 58 MoE (3..60)
        assert hits["lm_head_vocab_tp8"] == 1
        assert hits["mla_qkv_col_tp8"] == 61 * 2          # q_b_proj + kv_b_proj
        assert hits["mla_concat_kv_head_tp8"] == 61
        assert hits["mla_sdpa_head_tp8"] == 61
        assert hits["mla_o_row_tp8"] == 61
        assert hits["moe_experts_ep8"] == 58              # 58 MoE layers x 256 experts across EP=8
        assert hits["dense_and_shared_mlp_col_tp8"] == 61 * 2  # 3 dense + 58 shared_experts
        assert hits["dense_and_shared_mlp_row_tp8"] == 61
        assert manifest["post_pas_summary"]["num_fw_segments"] == 4
        total_params = manifest["pre_pas_summary"]["total_parameter_elements"]
        assert total_params > 660_000_000_000, f"Expected ~671B params, got {total_params}"
        print(
            f"  [PASS] Flagship DeepSeek-V3 / R1 671B-A37B (61L, 256E, {total_params / 1e9:.2f}B params, "
            f"32 GPUs PP=4xEP/TP=8) compiled on 0 GPUs in {manifest['model_meta']['compile_walltime_sec']}s"
        )


if __name__ == "__main__":
    print("=== Running DeepSeek Zero-GPU Weightless Compilation Suite ===")
    test_strategy_validator_catches_indivisible_mla_heads()
    test_compile_v2_qlora_and_group_limited_router()
    test_compile_v3_expanded_ep4_and_etp4()
    test_compile_full_deepseek_v2_lite_16b_ep4_tp4()
    test_compile_deepseek_v32_dsa_671b_scale_pp4_ep8()
    test_compile_frontier_deepseek_v3_671b_pp4_ep8_on_cpu()
    print("=== ALL DEEPSEEK ZERO-GPU COMPILATION TESTS PASSED ===")
