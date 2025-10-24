#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Zigzag attention correctness tests.

This module contains correctness tests for the zigzag attention implementation.
Note: Zigzag attention only supports causal=True and window_size=(-1, -1).

Usage:
    python -m pytest test_zigzag_attn.py -v
    python -m pytest test_zigzag_attn.py::TestZigzagAttn::test_zigzag_attn_tiny_bf16 -v
"""

import pytest
from test_base import RingAttnTestBase


class TestZigzagAttn(RingAttnTestBase):
    """Test class for zigzag attention correctness testing"""

    @property
    def runner_script_name(self) -> str:
        return "zigzag_attn_runner.py"

    @property
    def test_name_prefix(self) -> str:
        return "zigzag_attn"

    # Basic correctness tests
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    def test_zigzag_attn_tiny(self, dtype):
        """Test zigzag attention with tiny configuration"""
        self.run_correctness_basic(dtype, "zigzag_tiny")

    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    def test_zigzag_attn_small(self, dtype):
        """Test zigzag attention with small configuration"""
        self.run_correctness_basic(dtype, "zigzag_small")

    @pytest.mark.parametrize("dtype", ["bf16"])
    def test_zigzag_attn_medium(self, dtype):
        """Test zigzag attention with medium configuration"""
        self.run_correctness_basic(dtype, "zigzag_medium")

    # Multi-GPU tests
    @pytest.mark.parametrize("num_gpus", [2, 4])
    def test_zigzag_attn_multi_gpu_small(self, num_gpus):
        """Test zigzag attention with small config on multiple GPUs"""
        self.run_multi_gpu_scaling(num_gpus, "zigzag_small")

    @pytest.mark.parametrize("num_gpus", [2, 4])
    def test_zigzag_attn_multi_gpu_medium(self, num_gpus):
        """Test zigzag attention with medium config on multiple GPUs"""
        self.run_multi_gpu_scaling(num_gpus, "zigzag_medium")

    # GQA test
    def test_zigzag_attn_gqa(self):
        """Test zigzag attention with GQA configuration"""
        self.run_gqa_correctness("bf16", "zigzag_gqa")


if __name__ == "__main__":
    # For direct execution, run a simple test
    test_instance = TestZigzagAttn()
    test_instance.run_correctness_basic("bf16", "zigzag_tiny")