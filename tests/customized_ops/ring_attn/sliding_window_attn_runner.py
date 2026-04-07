#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Sliding Window Attention Correctness Test Runner

This script runs sliding window CP attention correctness tests in a distributed environment.
It compares the outputs of single-GPU and multi-GPU sliding window attention to ensure correctness.
"""

import sys
from typing import Tuple
import torch

from runner_base import RingAttnRunnerBase
from nnscaler.customized_ops.ring_attention import wrap_sliding_window_attn_func


class TestModule(torch.nn.Module):
    def __init__(self, causal=True, window_size=(-1, -1)):
        super(TestModule, self).__init__()
        self.causal = causal
        self.window_size = window_size

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k):
        out = wrap_sliding_window_attn_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, None,
            causal=self.causal,
            window_size=self.window_size
        )
        return out


class SlidingWindowAttnRunner(RingAttnRunnerBase):
    """Runner for sliding window CP attention tests"""

    @property
    def function_signature(self) -> str:
        return 'nnscaler.customized_ops.ring_attention.sliding_window_attn.wrap_sliding_window_attn_func'

    @property
    def partition_position(self) -> Tuple[int, int]:
        return 0, 0

    @property
    def function_name(self) -> str:
        return 'sliding_window_attn_func'

    def create_test_module(self, config) -> torch.nn.Module:
        return TestModule(causal=config.causal, window_size=config.window_size)

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare variable length inputs with cu_seqlens"""
        cu_seqlens_tensor = torch.tensor(config.cu_seqlens, dtype=torch.int32, device=device)
        total_seqlen = config.cu_seqlens[-1]

        q = torch.clamp(torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype), min=-1, max=1)
        k = torch.clamp(torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype), min=-1, max=1)
        v = torch.clamp(torch.randn(total_seqlen, config.num_heads, config.head_dim, device=device, dtype=torch_dtype), min=-1, max=1)

        return {
            'q': q,
            'k': k,
            'v': v,
            'cu_seqlens_q': cu_seqlens_tensor,
            'cu_seqlens_k': cu_seqlens_tensor
        }

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation"""
        single_out = wrap_sliding_window_attn_func(
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


def sliding_window_attn_test(dtype="bf16", config_name="small_window", **kwargs):
    """Pure test function that can be used with torchrun"""
    runner = SlidingWindowAttnRunner()
    return runner.run_correctness_test(dtype=dtype, config_name=config_name, **kwargs)


def run_sliding_window_correctness_test(**kwargs):
    """Legacy function for backward compatibility"""
    runner = SlidingWindowAttnRunner()
    runner.run_correctness_test(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    runner = SlidingWindowAttnRunner()
    runner.main(**kwargs)
