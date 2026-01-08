#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Variable Length Performance Benchmark
Uses the shared benchmark framework to reduce code duplication.
"""

import argparse
import sys
import os
import torch

# Import the benchmark base class
from benchmark_base import RingAttnBenchmarkBase

# Import ring attention implementation
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_varlen_func
from nnscaler.customized_ops.ring_attention.core.utils import set_seed

# Import test configuration (via the base class path setup)
tests_dir = os.path.join(os.path.dirname(__file__), "../tests/customized_ops/ring_attn")
sys.path.insert(0, tests_dir)
from configs import DEFAULT_PERFORMANCE_CONFIGS


class RingAttnVarlenBenchmark(RingAttnBenchmarkBase):
    """Benchmark for Ring Attention Variable Length"""

    @property
    def function_signature(self) -> str:
        return 'nnscaler.customized_ops.ring_attention.ring_attn_varlen.wrap_ring_attn_varlen_func'

    @property
    def function_name(self) -> str:
        return "ring_attn_varlen"

    def get_benchmark_name(self) -> str:
        return "Ring Attention Variable Length"

    def create_test_module(self, config) -> torch.nn.Module:
        """Create test module for variable length ring attention."""
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k):
                return wrap_ring_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, None,
                    causal=getattr(config, 'causal', True),
                    window_size=getattr(config, 'window_size', (-1, -1))
                )

        return TestModule()

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare input tensors for variable length sequence attention."""
        set_seed(42)
        
        # Get cu_seqlens from config or create default
        if hasattr(config, 'cu_seqlens'):
            cu_seqlens = config.cu_seqlens
        else:
            # Create default variable length sequences
            seqlen = config.max_seqlen
            cu_seqlens = [0, seqlen // 8, seqlen // 4, seqlen // 2, seqlen]

        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        total_tokens = cu_seqlens[-1]

        # Create input tensors
        q = torch.randn(total_tokens, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)
        k = torch.randn(total_tokens, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)
        v = torch.randn(total_tokens, config.num_heads, config.head_dim, device=device, dtype=torch_dtype)

        return {
            'q': q,
            'k': k, 
            'v': v,
            'cu_seqlens_q': cu_seqlens_tensor,
            'cu_seqlens_k': cu_seqlens_tensor
        }

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation."""
        q, k, v = inputs['q'], inputs['k'], inputs['v']
        cu_seqlens_q = inputs['cu_seqlens_q']
        cu_seqlens_k = inputs['cu_seqlens_k']
        
        output = wrap_ring_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, None,
            causal=getattr(config, 'causal', True),
            window_size=getattr(config, 'window_size', (-1, -1))
        )
        
        return output, [q, k, v]

    def get_dummy_forward_args(self, inputs):
        """Get dummy forward arguments for model parallelization."""
        dummy_args = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    dummy_args[k] = v.detach().clone().requires_grad_()
                else:
                    dummy_args[k] = v.detach().clone()
            else:
                dummy_args[k] = v
        return dummy_args

    def _create_legacy_config(self, **kwargs):
        """Create a legacy configuration from individual parameters for varlen."""
        class LegacyVarlenConfig:
            def __init__(self, **kwargs):
                self.name = "legacy_varlen_custom"
                self.max_seqlen = kwargs.get('seqlen', 16384)
                self.num_heads = kwargs.get('nheads', 24)
                self.head_dim = kwargs.get('head_dim', 128)
                self.batch_size = kwargs.get('batch_size', 4)
                self.dtype = "bf16"
                self.causal = True
                self.window_size = (-1, -1)
                
                # Create variable length sequences
                seqlen = self.max_seqlen
                self.cu_seqlens = kwargs.get('cu_seqlens', [0, seqlen // 8, seqlen // 4, seqlen // 2, seqlen])
                self.total_tokens = self.cu_seqlens[-1]

        return LegacyVarlenConfig(**kwargs)


def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(description="Ring Attention Variable Length Performance Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Predefined configuration name. Use --list-configs to see available options.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available predefined configurations",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Data type for inputs",
    )
    # Legacy parameters for custom configurations
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Total sequence length (overridden by --config)",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=None,
        help="Number of attention heads (overridden by --config)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Head dimension (overridden by --config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (number of sequences) (overridden by --config)",
    )
    # Timing parameters
    parser.add_argument(
        "--timing-method",
        type=str,
        default="warmup",
        choices=["simple", "profiler", "warmup"],
        help="Timing method: simple (basic timing), profiler (torch.profiler with detailed analysis), warmup (recommended: warm-up + multiple runs)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warm-up runs before timing (for warmup method)",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=5,
        help="Number of timing runs to average (for warmup method)",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = RingAttnVarlenBenchmark()

    if args.list_configs:
        benchmark.list_configurations()
    else:
        benchmark.run_performance_benchmark(
            config_name=args.config,
            dtype=args.dtype,
            timing_method=args.timing_method,
            warmup_runs=args.warmup_runs,
            timing_runs=args.timing_runs,
            # Legacy parameters
            seqlen=args.seqlen,
            nheads=args.nheads,
            head_dim=args.head_dim,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()