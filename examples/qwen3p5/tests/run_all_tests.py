"""Single-command test runner for Qwen3.5 zero-GPU support & compilation suite.

Usage:
    PYTHONPATH=. python -m examples.qwen3p5.tests.run_all_tests
    PYTHONPATH=. python -m examples.qwen3p5.tests.run_all_tests --include-397b
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path

from examples.qwen3p5.tests import (
    test_core_patches,
    test_ops_and_rewrites_parity,
    test_weightless_compile_e2e,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Qwen3.5 zero-GPU test suite')
    parser.add_argument(
        '--include-397b',
        action='store_true',
        help='Also run full 60-layer Qwen3.5-397B-A17B 32-GPU compilation test (~5-12 min on CPU).',
    )
    args = parser.parse_args()

    logging.getLogger('nnscaler').setLevel(logging.ERROR)

    t_suite = time.time()
    passed = 0

    def _run(label: str, fn, *fn_args) -> None:
        nonlocal passed
        t0 = time.time()
        print(f'[RUN ] {label}', flush=True)
        fn(*fn_args)
        print(f'[PASS] {label} ({time.time() - t0:.1f}s)', flush=True)
        passed += 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1. Core nnScaler patches
        _run('core_patches::test_safe_repr_memory_format', test_core_patches.test_safe_repr_memory_format)
        for fn_name in [
            'test_softplus_and_broadcast_masked_fill_parse_as_dimops',
            'test_weightless_meta_module_parallelize',
        ]:
            _run(
                f'core_patches::{fn_name}',
                getattr(test_core_patches, fn_name),
                root / fn_name,
            )

        # 2. Custom operators & CPU HF golden numerical parity
        for fn_name in [
            'test_gdn_ops_tp_split_forward_backward_parity',
            'test_moe_fused_and_expanded_ep_etp_parity',
            'test_vision_ops_split_parity',
            'test_rewritten_micro_models_match_hf_eager_golden',
        ]:
            _run(f'ops_and_parity::{fn_name}', getattr(test_ops_and_rewrites_parity, fn_name))

        # 3. End-to-end weightless compilation & static IR/AST audits
        e2e_tests = [
            'test_compile_dense_tp2_and_pp2_tp2',
            'test_compile_moe_fused_default_and_expanded_opt_in',
            'test_compile_multimodal_vl_tp2',
            'test_strategy_validator_catches_strict_indivisible_gqa_heads',
        ]
        if args.include_397b:
            e2e_tests.append('test_compile_frontier_397b_a17b_weightless_on_cpu')

        for fn_name in e2e_tests:
            _run(
                f'weightless_compile_e2e::{fn_name}',
                getattr(test_weightless_compile_e2e, fn_name),
                root / fn_name,
            )

    print(f'\nAll {passed} tests PASSED in {time.time() - t_suite:.1f}s.')


if __name__ == '__main__':
    main()
