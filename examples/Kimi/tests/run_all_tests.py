# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run all Kimi CPU golden parity and zero-GPU weightless compilation tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent.parent


def main() -> None:
    suites = [
        TESTS_DIR / "test_ops_and_rewrites_parity.py",
        TESTS_DIR / "test_weightless_compile_e2e.py",
    ]
    for suite in suites:
        print(f"\n>>> Running {suite.name} ...")
        res = subprocess.run(
            [sys.executable, str(suite)],
            cwd=str(REPO_ROOT),
            check=False,
        )
        if res.returncode != 0:
            sys.exit(res.returncode)
    print("\nAll Kimi adaptation test suites PASSED.")


if __name__ == "__main__":
    main()
