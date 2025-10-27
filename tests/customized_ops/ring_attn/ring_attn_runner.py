#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Correctness Test Runner Script

This script runs ring attention correctness tests in a distributed environment.
It compares the outputs of single-GPU and multi-GPU ring attention to ensure correctness.
"""

import sys
import torch

from runner_base import RingAttnRunnerBase
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_func


class TestModule(torch.nn.Module):
    """Test module for ring attention"""
    def __init__(self, causal=True, window_size=(-1, -1)):
        super(TestModule, self).__init__()
        self.causal = causal
        self.window_size = window_size

    def forward(self, q, k, v):
        result = wrap_ring_attn_func(
            q, k, v,
            causal=self.causal,
            window_size=self.window_size
        )
        return result


class RingAttnRunner(RingAttnRunnerBase):
    """Runner for ring attention tests"""

    @property
    def function_signature(self) -> str:
        return 'nnscaler.customized_ops.ring_attention.ring_attn.wrap_ring_attn_func'

    @property
    def function_name(self) -> str:
        return 'wrap_ring_attn_func'

    def create_test_module(self, config) -> torch.nn.Module:
        return TestModule(causal=config.causal, window_size=config.window_size)

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare regular inputs with shape [batch_size, seq_len, num_heads, head_dim]"""
        q = torch.randn(
            config.batch_size,
            config.max_seqlen,
            config.num_heads,
            config.head_dim,
            device=device,
            dtype=torch_dtype
        )

        k = torch.randn(
            config.batch_size,
            config.max_seqlen,
            config.num_kv_heads,
            config.head_dim,
            device=device,
            dtype=torch_dtype
        )

        v = torch.randn(
            config.batch_size,
            config.max_seqlen,
            config.num_kv_heads,
            config.head_dim,
            device=device,
            dtype=torch_dtype
        )

        return {'q': q, 'k': k, 'v': v}

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation"""
        # Run single GPU version (this should call flash_attn internally when no process_group)
        single_out = wrap_ring_attn_func(
            inputs['q'], inputs['k'], inputs['v'],
            causal=config.causal,
            window_size=config.window_size
        )
        return single_out, [inputs['q'], inputs['k'], inputs['v']]

    def get_dummy_forward_args(self, inputs):
        """Get dummy forward arguments for model parallelization"""
        return {
            "q": inputs["q"],
            "k": inputs["k"],
            "v": inputs["v"],
        }


def run_correctness_test(**kwargs):
    """Legacy function for backward compatibility"""
    runner = RingAttnRunner()
    runner.run_correctness_test(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    runner = RingAttnRunner()
    runner.main(**kwargs)