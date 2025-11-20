#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Correctness Tests

This module tests the correctness of regular ring attention (non-variable length).
It uses the shared test base framework to avoid code duplication.
"""

import pytest
import torch

# Skip all tests if flash_attn_func is not available
try:
    from flash_attn import flash_attn_func
except ImportError:
    pytest.skip("flash_attn_func not available", allow_module_level=True)

from .test_base import RingAttnTestBase, create_parametrized_tests
from .configs import DEFAULT_CORRECTNESS_CONFIGS, DEFAULT_MULTI_GPU_CONFIGS, DEFAULT_GQA_CONFIGS


class RingAttnTest(RingAttnTestBase):
    """Test class for regular ring attention"""

    @property
    def runner_script_name(self) -> str:
        return "ring_attn_runner.py"

    @property
    def test_function_name(self) -> str:
        return "ring_attn_test"

    @property
    def test_name_prefix(self) -> str:
        return "ring_attn"


# Create parametrized test functions using the factory
test_functions = create_parametrized_tests(RingAttnTest)

# Assign test functions to module globals for pytest discovery
test_ring_attn_correctness = test_functions['test_ring_attn_correctness']
test_ring_attn_multi_gpu = test_functions['test_ring_attn_multi_gpu']
test_ring_attn_all_configs = test_functions['test_ring_attn_all_configs']
test_ring_attn_gqa_correctness = test_functions['test_ring_attn_gqa_correctness']
test_ring_attn_sliding_window = test_functions['test_ring_attn_sliding_window']


if __name__ == "__main__":
    # Run specific test if called directly
    test_instance = RingAttnTest()
    test_instance.run_correctness_basic("bf16", "small")

    # Example of running GQA test
    # test_instance.run_gqa_correctness("bf16", "qwen3_4b")

    # Example of running sliding window test
    # test_instance.run_sliding_window("bf16", "small_window")