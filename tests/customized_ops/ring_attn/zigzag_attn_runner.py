#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Zigzag attention test runner implementation.
This module provides the specific runner for testing zigzag attention.
Note: Zigzag attention only supports causal=True and window_size=(-1, -1).
"""

import os
import sys
from typing import Dict, Any

import torch
import torch.nn as nn

from nnscaler.customized_ops.ring_attention.zigzag_attn import wrap_zigzag_attn_func
from runner_base import RingAttnRunnerBase


class ZigzagAttnRunner(RingAttnRunnerBase):
    """Zigzag attention test runner"""

    @property
    def function_signature(self) -> str:
        return "wrap_zigzag_attn_func"

    @property
    def function_name(self) -> str:
        return "wrap_zigzag_attn_func"

    def create_test_module(self, config) -> torch.nn.Module:
        """Create test module for zigzag attention"""
        class TestModule(nn.Module):
            def __init__(self, causal=True, window_size=(-1, -1)):
                super().__init__()
                # Zigzag attention only supports causal=True and window_size=(-1, -1)
                assert causal is True, "Zigzag attention only supports causal=True"
                assert window_size == (-1, -1), "Zigzag attention only supports window_size=(-1, -1)"
                self.causal = causal
                self.window_size = window_size

            def forward(self, q, k, v):
                # Note: zigzag_attn always uses causal=True and window_size=(-1, -1)
                return wrap_zigzag_attn_func(q, k, v, causal=self.causal, window_size=self.window_size)

        return TestModule(causal=config.causal, window_size=config.window_size)

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare inputs for zigzag attention"""
        batch_size = config.batch_size
        max_seqlen = config.max_seqlen
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim

        # Create input tensors
        q = torch.randn(batch_size, max_seqlen, num_heads, head_dim, device=device, dtype=torch_dtype)
        k = torch.randn(batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
        v = torch.randn(batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=torch_dtype)

        return {
            'q': q,
            'k': k,
            'v': v
        }

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation"""
        # Note: zigzag_attn always uses causal=True and window_size=(-1, -1)
        output = wrap_zigzag_attn_func(
            inputs['q'], inputs['k'], inputs['v'],
            causal=config.causal, window_size=config.window_size)
        output.retain_grad()

        return output, [inputs['q'], inputs['k'], inputs['v']]

    def get_dummy_forward_args(self, inputs) -> Dict[str, Any]:
        """Get dummy forward arguments for model parallelization"""
        return {
            'q': inputs['q'],
            'k': inputs['k'],
            'v': inputs['v']
        }


def main():
    """Main entry point for command line execution"""
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    
    runner = ZigzagAttnRunner()
    runner.main(**kwargs)


if __name__ == "__main__":
    main()