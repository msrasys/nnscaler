#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Variable Length Correctness Tests

This module tests the correctness of ring attention with variable length sequences.
It uses the shared test base framework to avoid code duplication.
"""

import pytest
import torch

from test_base import RingAttnTestBase, create_parametrized_tests
from configs import DEFAULT_CORRECTNESS_CONFIGS, DEFAULT_MULTI_GPU_CONFIGS, DEFAULT_GQA_CONFIGS


class RingAttnVarlenTest(RingAttnTestBase):
    """Test class for ring attention variable length"""

    @property
    def runner_script_name(self) -> str:
        return "ring_attn_varlen_runner.py"

    @property
    def test_name_prefix(self) -> str:
        return "ring_attn_varlen"


# Create parametrized test functions using the factory
test_functions = create_parametrized_tests(RingAttnVarlenTest)

# Assign test functions to module globals for pytest discovery
test_ring_attn_varlen_correctness = test_functions['test_ring_attn_varlen_correctness']
test_ring_attn_varlen_multi_gpu = test_functions['test_ring_attn_varlen_multi_gpu']
test_ring_attn_varlen_all_configs = test_functions['test_ring_attn_varlen_all_configs']
test_ring_attn_varlen_gqa_correctness = test_functions['test_ring_attn_varlen_gqa_correctness']
test_ring_attn_varlen_sliding_window = test_functions['test_ring_attn_varlen_sliding_window']


if __name__ == "__main__":
    # Run specific test if called directly
    test_instance = RingAttnVarlenTest()
    test_instance.run_correctness_basic("bf16", "small")

    # Example of running GQA test
    # test_instance.run_gqa_correctness("bf16", "qwen3_4b")
