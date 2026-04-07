#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Sliding Window Attention Correctness Tests

This module tests the correctness of context parallel sliding window attention.
It uses the shared test base framework.
"""

import pytest
import torch

# Skip all tests if flash_attn_varlen_func is not available
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    pytest.skip("flash_attn_varlen_func not available", allow_module_level=True)

from .test_base import RingAttnTestBase, create_parametrized_tests
from .configs import DEFAULT_CORRECTNESS_CONFIGS


class SlidingWindowAttnTest(RingAttnTestBase):
    """Test class for sliding window CP attention"""

    @property
    def runner_script_name(self) -> str:
        return "sliding_window_attn_runner.py"

    @property
    def test_function_name(self) -> str:
        return "sliding_window_attn_test"

    @property
    def test_name_prefix(self) -> str:
        return "sliding_window_attn"


# Sliding window tests use window configs
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("config_name", ["small_window", "medium_window"])
def test_sliding_window_attn_correctness(dtype, config_name):
    """Test sliding window CP attention correctness"""
    instance = SlidingWindowAttnTest()
    instance.run_sliding_window(dtype, config_name)


@pytest.mark.parametrize("num_gpus", [2, 4])
@pytest.mark.parametrize("config_name", ["small_window", "medium_window"])
def test_sliding_window_attn_multi_gpu(num_gpus, config_name):
    """Test sliding window CP attention with different numbers of GPUs"""
    instance = SlidingWindowAttnTest()
    instance.run_multi_gpu_scaling(num_gpus, config_name)


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("config_name", ["large_window"])
def test_sliding_window_attn_large(dtype, config_name):
    """Test sliding window CP attention with large config"""
    instance = SlidingWindowAttnTest()
    instance.run_sliding_window(dtype, config_name)


if __name__ == "__main__":
    test_instance = SlidingWindowAttnTest()
    test_instance.run_sliding_window("bf16", "small_window")
