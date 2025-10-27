#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Variable Length Correctness Test Runner

This script runs ring attention variable length correctness tests in a distributed environment.
It compares the outputs of single-GPU and multi-GPU ring attention to ensure correctness.
"""

import sys
import torch

from runner_base import RingAttnRunnerBase
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_varlen_func


class TestModule(torch.nn.Module):
    def __init__(self, causal=True, window_size=(-1, -1)):
        super(TestModule, self).__init__()
        self.causal = causal
        self.window_size = window_size

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k):
        out = wrap_ring_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, None,
            causal=self.causal,
            window_size=self.window_size
        )
        return out


class RingAttnVarlenRunner(RingAttnRunnerBase):
    """Runner for ring attention variable length tests"""

    @property
    def function_signature(self) -> str:
        return 'nnscaler.customized_ops.ring_attention.ring_attn_varlen.wrap_ring_attn_varlen_func'

    @property
    def function_name(self) -> str:
        return 'ring_attn_varlen_func'

    def create_test_module(self, config) -> torch.nn.Module:
        return TestModule(causal=config.causal, window_size=config.window_size)

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare variable length inputs with cu_seqlens"""
        cu_seqlens_tensor = torch.tensor(config.cu_seqlens, dtype=torch.int32, device=device)
        total_seqlen = config.cu_seqlens[-1]

        # Create inputs with total sequence length (don't set requires_grad here, base class handles it)
        q = torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)
        k = torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)
        v = torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)

        return {
            'q': q,
            'k': k,
            'v': v,
            'cu_seqlens_q': cu_seqlens_tensor,
            'cu_seqlens_k': cu_seqlens_tensor
        }

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation"""
        single_out = wrap_ring_attn_varlen_func(
            inputs['q'], inputs['k'], inputs['v'],
            inputs['cu_seqlens_q'], inputs['cu_seqlens_k'], None,
            causal=config.causal,
            window_size=config.window_size
        )
        single_out.retain_grad()
        return single_out, [inputs['q'], inputs['k'], inputs['v']]

    def get_dummy_forward_args(self, inputs):
        """Get dummy forward arguments for model parallelization"""
        return {
            "q": inputs["q"],
            "k": inputs["k"],
            "v": inputs["v"],
            'cu_seqlens_q': inputs['cu_seqlens_q'],
            'cu_seqlens_k': inputs['cu_seqlens_k']
        }


def run_ring_attn_correctness_test(**kwargs):
    """Legacy function for backward compatibility"""
    runner = RingAttnVarlenRunner()
    runner.run_correctness_test(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    runner = RingAttnVarlenRunner()
    runner.main(**kwargs)
