#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Base benchmark framework for ring attention performance tests.
This module extends the test framework to support performance benchmarking.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

# Add tests directory to path to import test framework
tests_dir = os.path.join(os.path.dirname(__file__), "../tests/customized_ops/ring_attn")
sys.path.insert(0, tests_dir)

from runner_base import RingAttnRunnerBase
from configs import get_config, get_configs_by_category, DEFAULT_PERFORMANCE_CONFIGS


class RingAttnBenchmarkBase(RingAttnRunnerBase):
    """Base class for ring attention performance benchmarks"""

    def __init__(self):
        super().__init__()
        self.timing_method = "warmup"
        self.warmup_runs = 3
        self.timing_runs = 5

    @abstractmethod
    def get_benchmark_name(self) -> str:
        """Return the benchmark name for display"""
        pass

    def run_timing_with_warmup(self, forward_fn: Callable, backward_fn: Callable, 
                              warmup_runs: int = None, timing_runs: int = None) -> Tuple[float, float, Any]:
        """Run timing with warm-up runs to get accurate measurements."""
        warmup_runs = warmup_runs or self.warmup_runs
        timing_runs = timing_runs or self.timing_runs
        
        # Warm-up runs
        for _ in range(warmup_runs):
            torch.cuda.synchronize()
            output = forward_fn()
            torch.cuda.synchronize()
            backward_fn(output)
            torch.cuda.synchronize()

        # Timing runs
        forward_times = []
        backward_times = []

        for _ in range(timing_runs):
            # Forward timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = forward_fn()
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - start
            forward_times.append(forward_time)

            # Backward timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            backward_fn(output)
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - start
            backward_times.append(backward_time)

        # Return average times
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        return avg_forward, avg_backward, output

    def run_timing_with_profiler(self, forward_fn: Callable, backward_fn: Callable, 
                                rank_id: int = 0) -> Tuple[float, float, Any]:
        """Run timing using torch.profiler for detailed analysis."""
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        # Run profiler with timing
        torch.cuda.synchronize()

        with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            output = forward_fn()
            torch.cuda.synchronize()
            forward_end = time.perf_counter()

            torch.cuda.synchronize()
            backward_start = time.perf_counter()
            backward_fn(output)
            torch.cuda.synchronize()
            backward_end = time.perf_counter()

        torch.cuda.synchronize()

        # Calculate timing from our measurements
        forward_time = forward_end - forward_start
        backward_time = backward_end - backward_start

        if rank_id == 0:
            self._print_profiler_results(prof)

        return forward_time, backward_time, output

    def run_timing_simple(self, forward_fn: Callable, backward_fn: Callable) -> Tuple[float, float, Any]:
        """Run simple timing without warmup or profiling."""
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        output = forward_fn()
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - forward_start

        torch.cuda.synchronize()
        backward_start = time.perf_counter()
        backward_fn(output)
        torch.cuda.synchronize()
        backward_time = time.perf_counter() - backward_start

        return forward_time, backward_time, output

    def _print_profiler_results(self, prof):
        """Print profiler results with fallback for different PyTorch versions."""
        print("\n" + "="*60)
        print("TORCH PROFILER RESULTS")
        print("="*60)

        try:
            # Try the most common sorting options
            events = prof.key_averages()
            table_str = events.table(sort_by="self_cuda_time_total", row_limit=20)
            print(table_str)
        except Exception as e1:
            try:
                table_str = events.table(sort_by="cuda_time_total", row_limit=20)
                print(table_str)
            except Exception as e2:
                try:
                    table_str = events.table(sort_by="self_cpu_time_total", row_limit=20)
                    print(table_str)
                except Exception as e3:
                    print(f"Warning: Could not generate profiler table due to API differences")
                    print(f"Errors: {e1}, {e2}, {e3}")

                    # Fallback: print basic event info
                    print("Available profiler events:")
                    for i, event in enumerate(events):
                        if i >= 10:  # Limit output
                            break
                        try:
                            print(f"  {event.key}: CPU time = {getattr(event, 'cpu_time_total', 'N/A')} us")
                        except:
                            print(f"  {event.key}: [timing info unavailable]")

        print("="*60 + "\n")

    def create_timing_functions(self, inputs, config, dout_tensor):
        """Create timing functions for single and parallel execution."""
        # Single mode functions
        def single_forward():
            single_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        single_inputs[k] = v.detach().clone().requires_grad_()
                    else:
                        single_inputs[k] = v.detach().clone()
                else:
                    single_inputs[k] = v
            
            # Run single GPU reference
            output, grad_tensors = self.run_single_gpu_reference(single_inputs, config)
            return output, (single_inputs, grad_tensors)

        def single_backward(outputs):
            output, (single_inputs, grad_tensors) = outputs
            output.backward(dout_tensor)
            return dout_tensor

        # Parallel mode functions
        model = self.create_test_module(config)
        dummy_args = self.get_dummy_forward_args(inputs)
        
        from nnscaler.parallel import parallelize, ComputeConfig, ReuseType
        world_size = dist.get_world_size()
        
        parallel_model = parallelize(
            model,
            dummy_forward_args=dummy_args,
            pas_policy=self.create_policy(),
            compute_config=ComputeConfig(world_size, world_size),
            reuse=ReuseType.OVERRIDE
        )
        parallel_model = parallel_model.cuda()
        parallel_model.train()

        def parallel_forward():
            para_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        para_inputs[k] = v.detach().clone().requires_grad_()
                    else:
                        para_inputs[k] = v.detach().clone()
                else:
                    para_inputs[k] = v
            
            output = parallel_model(**para_inputs)
            return output, para_inputs

        def parallel_backward(outputs):
            output, para_inputs = outputs
            output.backward(dout_tensor)
            parallel_model.sync_grad()
            return dout_tensor

        return single_forward, single_backward, parallel_forward, parallel_backward

    def calculate_throughput_metrics(self, config, forward_time: float, backward_time: float) -> Dict[str, float]:
        """Calculate throughput and efficiency metrics."""
        total_time = forward_time + backward_time
        
        # Calculate total tokens processed
        if hasattr(config, 'total_tokens'):
            total_tokens = config.total_tokens
        else:
            total_tokens = config.batch_size * config.max_seqlen
        
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'throughput_tokens_per_sec': throughput,
            'total_time': total_time,
            'forward_time': forward_time,
            'backward_time': backward_time
        }

    def print_benchmark_results(self, config_name: str, config, dtype: str, 
                               single_metrics: Dict[str, float], 
                               parallel_metrics: Dict[str, float], 
                               world_size: int, rank_id: int):
        """Print comprehensive benchmark results."""
        if rank_id != 0:
            return

        print("\n" + "="*80)
        print(f"{self.get_benchmark_name().upper()} PERFORMANCE BENCHMARK ({self.timing_method.upper()} METHOD)")
        print(f"Configuration: {config_name} - {config.name}")
        print(f"  Sequence length: {config.max_seqlen}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Head dim: {config.head_dim}")
        print(f"  Data type: {dtype}")
        print(f"  World size: {world_size} GPUs")
        print(f"  Total tokens: {single_metrics['total_tokens']:,}")
        
        if self.timing_method == "warmup":
            print(f"  (Warmup runs: {self.warmup_runs}, Timing runs: {self.timing_runs})")
        print("="*80)

        # Timing results
        print(f"Single Mode:")
        print(f"  Forward time:  {single_metrics['forward_time']:.6f} seconds")
        print(f"  Backward time: {single_metrics['backward_time']:.6f} seconds")
        print(f"  Total time:    {single_metrics['total_time']:.6f} seconds")
        print(f"  Throughput:    {single_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

        print(f"\nParallel Mode:")
        print(f"  Forward time:  {parallel_metrics['forward_time']:.6f} seconds")
        print(f"  Backward time: {parallel_metrics['backward_time']:.6f} seconds")
        print(f"  Total time:    {parallel_metrics['total_time']:.6f} seconds")
        print(f"  Throughput:    {parallel_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

        # Speedup calculations
        forward_speedup = single_metrics['forward_time'] / parallel_metrics['forward_time'] if parallel_metrics['forward_time'] > 0 else 0
        backward_speedup = single_metrics['backward_time'] / parallel_metrics['backward_time'] if parallel_metrics['backward_time'] > 0 else 0
        total_speedup = single_metrics['total_time'] / parallel_metrics['total_time'] if parallel_metrics['total_time'] > 0 else 0
        throughput_improvement = parallel_metrics['throughput_tokens_per_sec'] / single_metrics['throughput_tokens_per_sec'] if single_metrics['throughput_tokens_per_sec'] > 0 else 0

        print(f"\nSpeedup:")
        print(f"  Forward speedup:     {forward_speedup:.2f}x")
        print(f"  Backward speedup:    {backward_speedup:.2f}x")
        print(f"  Total speedup:       {total_speedup:.2f}x")
        print(f"  Throughput improvement: {throughput_improvement:.2f}x")

        # Efficiency metrics
        theoretical_speedup = world_size
        efficiency = total_speedup / theoretical_speedup * 100 if theoretical_speedup > 0 else 0
        print(f"\nEfficiency:")
        print(f"  Theoretical speedup: {theoretical_speedup:.0f}x")
        print(f"  Actual speedup:      {total_speedup:.2f}x")
        print(f"  Parallel efficiency: {efficiency:.1f}%")
        print("="*80 + "\n")

    def run_performance_benchmark(self, config_name: str = None, dtype: str = "bf16", 
                                 timing_method: str = "warmup", warmup_runs: int = 3, 
                                 timing_runs: int = 5, **legacy_kwargs):
        """Run performance benchmark for the specific attention implementation."""
        # Setup timing parameters
        self.timing_method = timing_method
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs

        # Initialize distributed environment
        world_size, rank = self.initialize_distributed()
        rank_id = dist.get_rank()

        # Get configuration
        config = get_config(config_name) if config_name else self._create_legacy_config(**legacy_kwargs)
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        if rank_id == 0:
            print(f"Running {self.get_benchmark_name()} performance benchmark...")
            print(f"Configuration: {config.name if hasattr(config, 'name') else 'custom'}")

        # Prepare inputs
        device = torch.device(f"cuda:{rank_id}")
        inputs = self.prepare_inputs(config, device, torch_dtype)

        # Broadcast inputs to ensure consistency
        for tensor in inputs.values():
            if isinstance(tensor, torch.Tensor):
                dist.broadcast(tensor, src=0)
        dist.barrier()

        # Pre-generate dout tensor for timing consistency
        with torch.no_grad():
            dummy_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    dummy_inputs[k] = v.detach()
                else:
                    dummy_inputs[k] = v
            dummy_out, _ = self.run_single_gpu_reference(dummy_inputs, config)
            dout_tensor = torch.randn_like(dummy_out, device=device, dtype=torch_dtype)
            dist.broadcast(dout_tensor, src=0)

        # Create timing functions
        single_forward, single_backward, parallel_forward, parallel_backward = self.create_timing_functions(
            inputs, config, dout_tensor
        )

        if rank_id == 0:
            print(f"Running performance benchmark using {timing_method} method...", end="")

        # Run timing based on method
        if timing_method == "profiler":
            single_forward_time, single_backward_time, _ = self.run_timing_with_profiler(
                single_forward, single_backward, rank_id
            )
            parallel_forward_time, parallel_backward_time, _ = self.run_timing_with_profiler(
                parallel_forward, parallel_backward, rank_id
            )
        elif timing_method == "warmup":
            single_forward_time, single_backward_time, _ = self.run_timing_with_warmup(
                single_forward, single_backward, warmup_runs, timing_runs
            )
            parallel_forward_time, parallel_backward_time, _ = self.run_timing_with_warmup(
                parallel_forward, parallel_backward, warmup_runs, timing_runs
            )
        else:  # simple
            single_forward_time, single_backward_time, _ = self.run_timing_simple(
                single_forward, single_backward
            )
            parallel_forward_time, parallel_backward_time, _ = self.run_timing_simple(
                parallel_forward, parallel_backward
            )

        if rank_id == 0:
            print(" Done!")

        # Calculate metrics and print results
        single_metrics = self.calculate_throughput_metrics(config, single_forward_time, single_backward_time)
        parallel_metrics = self.calculate_throughput_metrics(config, parallel_forward_time, parallel_backward_time)

        self.print_benchmark_results(
            config_name or "custom", config, dtype, 
            single_metrics, parallel_metrics, world_size, rank_id
        )

        # Cleanup
        dist.destroy_process_group()

    def _create_legacy_config(self, **kwargs):
        """Create a legacy configuration from individual parameters."""
        class LegacyConfig:
            def __init__(self, **kwargs):
                self.name = "legacy_custom"
                self.max_seqlen = kwargs.get('seqlen', 16384)
                self.num_heads = kwargs.get('nheads', 24)
                self.head_dim = kwargs.get('head_dim', 128)
                self.batch_size = kwargs.get('batch_size', 4)
                self.total_tokens = self.batch_size * self.max_seqlen
                self.dtype = "bf16"
                # Add other default attributes as needed

        return LegacyConfig(**kwargs)

    def list_configurations(self):
        """List all available configurations for benchmarking."""
        print("Available Ring Attention Configurations:")
        print("=" * 50)

        for category in ["small", "medium", "large", "gqa"]:
            print(f"\n{category.upper()} CONFIGS:")
            configs = get_configs_by_category(category)
            if configs:
                for name, config in configs.items():
                    tokens_k = config.total_tokens // 1000
                    gqa_info = f" (GQA {config.num_heads}->{config.num_kv_heads})" if config.is_gqa else ""
                    causal_info = " [Causal]" if config.causal else " [Non-causal]"
                    window_info = f" [Window={config.window_size[0]},{config.window_size[1]}]" if config.window_size != (-1, -1) else ""
                    print(f"  {name:20s} - {config.batch_size}x{config.num_heads}x{config.head_dim}, seq={config.max_seqlen}, tokens={tokens_k}K, {config.dtype}{gqa_info}{causal_info}{window_info}")
            else:
                print("  No configurations in this category")

        print(f"\nDEFAULT PERFORMANCE CONFIGS: {DEFAULT_PERFORMANCE_CONFIGS}")
        print(f"\nUsage: Use --config <config_name> to specify a configuration")