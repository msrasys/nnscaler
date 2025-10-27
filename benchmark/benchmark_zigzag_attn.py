#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Zigzag Attention Performance Benchmark
Uses the shared benchmark framework to reduce code duplication.
"""

import argparse
import sys
import os
import torch

# Import the benchmark base class
from benchmark_base import RingAttnBenchmarkBase

# Import zigzag attention implementation
from nnscaler.customized_ops.ring_attention import wrap_zigzag_attn_func
from nnscaler.customized_ops.ring_attention.core.utils import set_seed

# Import test configuration (via the base class path setup)
tests_dir = os.path.join(os.path.dirname(__file__), "../tests/customized_ops/ring_attn")
sys.path.insert(0, tests_dir)
from configs import DEFAULT_PERFORMANCE_CONFIGS, ZIGZAG_CONFIGS


class ZigzagAttnBenchmark(RingAttnBenchmarkBase):
    """Benchmark for Zigzag Attention"""

    @property
    def function_signature(self) -> str:
        return 'nnscaler.customized_ops.ring_attention.zigzag_attn.wrap_zigzag_attn_func'

    @property
    def function_name(self) -> str:
        return "zigzag_attn"

    def get_benchmark_name(self) -> str:
        return "Zigzag Attention"

    def create_test_module(self, config) -> torch.nn.Module:
        """Create test module for zigzag attention."""
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, q, k, v):
                # Zigzag attention only supports causal=True and window_size=(-1,-1)
                return wrap_zigzag_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(-1, -1)
                )

        return TestModule()

    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare input tensors for zigzag attention."""
        set_seed(42)
        
        # Create input tensors with standard batch format
        q = torch.randn(config.batch_size, config.max_seqlen, config.num_heads, config.head_dim, 
                       device=device, dtype=torch_dtype)
        k = torch.randn(config.batch_size, config.max_seqlen, config.num_heads, config.head_dim, 
                       device=device, dtype=torch_dtype)
        v = torch.randn(config.batch_size, config.max_seqlen, config.num_heads, config.head_dim, 
                       device=device, dtype=torch_dtype)

        return {
            'q': q,
            'k': k, 
            'v': v
        }

    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation."""
        q, k, v = inputs['q'], inputs['k'], inputs['v']
        
        # Zigzag attention constraints
        output = wrap_zigzag_attn_func(
            q, k, v,
            causal=True,
            window_size=(-1, -1)
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
        """Create a legacy configuration from individual parameters for zigzag attention."""
        class LegacyZigzagAttnConfig:
            def __init__(self, **kwargs):
                self.name = "legacy_zigzag_attn_custom"
                self.max_seqlen = kwargs.get('seqlen', 16384)
                self.num_heads = kwargs.get('nheads', 24)
                self.head_dim = kwargs.get('head_dim', 128)
                self.batch_size = kwargs.get('batch_size', 4)
                self.total_tokens = self.batch_size * self.max_seqlen
                self.dtype = "bf16"
                # Zigzag attention constraints
                self.causal = True
                self.window_size = (-1, -1)

        return LegacyZigzagAttnConfig(**kwargs)

    def run_performance_benchmark(self, config_name: str = None, dtype: str = "bf16", 
                                 timing_method: str = "warmup", warmup_runs: int = 3, 
                                 timing_runs: int = 5, **legacy_kwargs):
        """Override to validate zigzag attention constraints."""
        # Validate configuration for zigzag constraints
        if config_name:
            from configs import get_config
            config = get_config(config_name)
            if not config.causal:
                print(f"WARNING: Config '{config_name}' has causal=False, but zigzag attention requires causal=True")
                print("Proceeding with causal=True for zigzag attention...")
            if config.window_size != (-1, -1):
                print(f"WARNING: Config '{config_name}' has window_size={config.window_size}, but zigzag attention requires (-1, -1)")
                print("Proceeding with window_size=(-1, -1) for zigzag attention...")

        # Call parent implementation
        super().run_performance_benchmark(
            config_name=config_name, dtype=dtype, timing_method=timing_method,
            warmup_runs=warmup_runs, timing_runs=timing_runs, **legacy_kwargs
        )

    def list_configurations(self):
        """List configurations suitable for zigzag attention."""
        print("Available Zigzag Attention Configurations:")
        print("=" * 50)
        print("NOTE: Zigzag attention only supports causal=True and window_size=(-1,-1)")
        print("Configurations listed below will be automatically adjusted for these constraints.\n")

        # Call parent method but with zigzag-specific note
        super().list_configurations()
        
        print(f"\nZIGZAG-SPECIFIC CONFIGS: {list(ZIGZAG_CONFIGS.keys())}")
        print("These configs are specifically designed for zigzag attention.")


def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(description="Zigzag Attention Performance Benchmark")
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
        help="Sequence length (overridden by --config)",
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
        help="Batch size (overridden by --config)",
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
    benchmark = ZigzagAttnBenchmark()

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