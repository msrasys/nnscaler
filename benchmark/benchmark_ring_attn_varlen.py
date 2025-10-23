#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import torch
import nnscaler
import argparse
import time
from torch.profiler import profile, ProfilerActivity
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType
import torch.distributed as dist

import nnscaler.graph
import nnscaler.graph.function

# Import ring attention from nnscaler.customized_ops
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_varlen_func
from nnscaler.customized_ops.ring_attention.core.utils import set_seed, log

# Import configurations from the tests directory
configs_dir = os.path.join(os.path.dirname(__file__), "../tests/customized_ops/ring_attn")
sys.path.insert(0, configs_dir)
from configs import get_config, list_configs, DEFAULT_PERFORMANCE_CONFIGS, get_configs_by_category, DEFAULT_GQA_CONFIGS


def run_timing_with_warmup(forward_fn, backward_fn, warmup_runs=3, timing_runs=5):
    """Run timing with warm-up runs to get accurate measurements."""
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


def run_timing_with_profiler(forward_fn, backward_fn, rank_id=0):
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
        # Print profiler summary
        print("\n" + "="*60)
        print("TORCH PROFILER RESULTS")
        print("="*60)

        # Try different table formats depending on available attributes
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

        # Print our manual timing measurements (most reliable)
        print(f"Manual timing measurements (most accurate):")
        print(f"  Forward:  {forward_time:.6f} seconds")
        print(f"  Backward: {backward_time:.6f} seconds")
        print(f"  Total:    {forward_time + backward_time:.6f} seconds")
        print()

    return forward_time, backward_time, output


class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k):
        out = wrap_ring_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, None)
        return out


def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if not partitioned and node.signature == 'nnscaler.customized_ops.ring_attention.ring_attn_varlen.wrap_ring_attn_varlen_func':
            print('\nPartitioned node: ', node, '\n')
            sub_nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=ngpus)
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    assert partitioned, f'expect ring_attn_varlen_func in graph, but not found.'
    return graph


def run_performance_benchmark(
    config_name=None,
    dtype="bf16",
    # Legacy parameters for backward compatibility
    seqlen=None,
    nheads=None,
    head_dim=None,
    batch_size=None,
    # Timing parameters
    timing_method="warmup",
    warmup_runs=3,
    timing_runs=5,
):
    """Run performance benchmark for ring attention"""

    nnscaler.init()
    rank_id = torch.distributed.get_rank()
    world_size = dist.get_world_size()

    # Get configuration
    if config_name is not None:
        config = get_config(config_name)
        seqlen = config.max_seqlen
        nheads = config.num_heads
        head_dim = config.head_dim
        batch_size = config.batch_size
        cu_seqlens = config.cu_seqlens
        if rank_id == 0:
            print(f"Using predefined config '{config_name}': {config.name}")
    else:
        # Use provided parameters or defaults
        seqlen = seqlen or 16384
        nheads = nheads or 24
        head_dim = head_dim or 128
        batch_size = batch_size or 4
        cu_seqlens = [0, seqlen // 8, seqlen // 4, seqlen // 2, seqlen]
        if rank_id == 0:
            print(f"Using custom config: seq={seqlen}, heads={nheads}, dim={head_dim}, batch={batch_size}")

    set_seed(rank_id)
    device = torch.device(f"cuda:{rank_id}")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # Prepare input data
    q = torch.randn(seqlen, nheads, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(seqlen, nheads, head_dim, device=device, dtype=torch_dtype)
    v = torch.randn(seqlen, nheads, head_dim, device=device, dtype=torch_dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.barrier()

    # Setup models
    model = TestModule()

    # Create parallel model
    qq_dummy = q.detach().clone().requires_grad_()
    kk_dummy = k.detach().clone().requires_grad_()
    vv_dummy = v.detach().clone().requires_grad_()

    parallel_model = parallelize(
        model,
        dummy_forward_args={
            "q": qq_dummy,
            "k": kk_dummy,
            "v": vv_dummy,
            'cu_seqlens_q': cu_seqlens_tensor,
            'cu_seqlens_k': cu_seqlens_tensor
        },
        pas_policy=policy,
        compute_config=ComputeConfig(world_size, world_size),
        reuse=ReuseType.OVERRIDE
    )
    parallel_model = parallel_model.cuda()
    parallel_model.train()

    # Pre-generate dout tensor for timing tests to avoid including generation time
    # First get output shape by running a dummy forward pass
    with torch.no_grad():
        q_dummy = q.detach()
        k_dummy = k.detach()
        v_dummy = v.detach()
        dummy_out = wrap_ring_attn_varlen_func(q_dummy, k_dummy, v_dummy, cu_seqlens_tensor, cu_seqlens_tensor, None)
        dout_timing = torch.randn_like(dummy_out, device=device, dtype=torch_dtype)
        dist.broadcast(dout_timing, src=0)  # Ensure consistency across ranks

    # Define timing functions
    def single_forward():
        q_t = q.detach().clone().requires_grad_()
        k_t = k.detach().clone().requires_grad_()
        v_t = v.detach().clone().requires_grad_()
        out = wrap_ring_attn_varlen_func(q_t, k_t, v_t, cu_seqlens_tensor, cu_seqlens_tensor, None)
        return out, (q_t, k_t, v_t)

    def single_backward(outputs):
        out, tensors = outputs
        # Use pre-generated dout tensor to avoid timing overhead
        out.backward(dout_timing)
        return dout_timing

    def parallel_forward():
        qq_t = q.detach().clone().requires_grad_()
        kk_t = k.detach().clone().requires_grad_()
        vv_t = v.detach().clone().requires_grad_()
        out = parallel_model(qq_t, kk_t, vv_t, cu_seqlens_tensor, cu_seqlens_tensor)
        return out, (qq_t, kk_t, vv_t)

    def parallel_backward(outputs):
        out, tensors = outputs
        # Use pre-generated dout tensor to avoid timing overhead
        out.backward(dout_timing)
        parallel_model.sync_grad()
        return dout_timing

    print(f"Running performance benchmark using {timing_method} method..." if rank_id == 0 else "", end="")

    # Run timing based on method
    if timing_method == "profiler":
        single_forward_time, single_backward_time, _ = run_timing_with_profiler(
            single_forward, single_backward, rank_id
        )
        parallel_forward_time, parallel_backward_time, _ = run_timing_with_profiler(
            parallel_forward, parallel_backward, rank_id
        )
    elif timing_method == "warmup":
        single_forward_time, single_backward_time, _ = run_timing_with_warmup(
            single_forward, single_backward, warmup_runs, timing_runs
        )
        parallel_forward_time, parallel_backward_time, _ = run_timing_with_warmup(
            parallel_forward, parallel_backward, warmup_runs, timing_runs
        )
    else:  # simple
        torch.cuda.synchronize()
        single_forward_start = time.perf_counter()
        single_outputs = single_forward()
        torch.cuda.synchronize()
        single_forward_time = time.perf_counter() - single_forward_start

        torch.cuda.synchronize()
        single_backward_start = time.perf_counter()
        single_backward(single_outputs)
        torch.cuda.synchronize()
        single_backward_time = time.perf_counter() - single_backward_start

        torch.cuda.synchronize()
        parallel_forward_start = time.perf_counter()
        parallel_outputs = parallel_forward()
        torch.cuda.synchronize()
        parallel_forward_time = time.perf_counter() - parallel_forward_start

        torch.cuda.synchronize()
        parallel_backward_start = time.perf_counter()
        parallel_backward(parallel_outputs)
        torch.cuda.synchronize()
        parallel_backward_time = time.perf_counter() - parallel_backward_start

    # Print timing statistics
    if rank_id == 0:
        print(" Done!")
        print("\n" + "="*80)
        print(f"RING ATTENTION PERFORMANCE BENCHMARK ({timing_method.upper()} METHOD)")
        print(f"Configuration: seq_len={seqlen}, heads={nheads}, head_dim={head_dim}, dtype={dtype}")
        print(f"World size: {world_size} GPUs")
        if timing_method == "warmup":
            print(f"(Warmup runs: {warmup_runs}, Timing runs: {timing_runs})")
        print("="*80)
        print(f"Single Mode:")
        print(f"  Forward time:  {single_forward_time:.6f} seconds")
        print(f"  Backward time: {single_backward_time:.6f} seconds")
        print(f"  Total time:    {single_forward_time + single_backward_time:.6f} seconds")
        print(f"\nParallel Mode:")
        print(f"  Forward time:  {parallel_forward_time:.6f} seconds")
        print(f"  Backward time: {parallel_backward_time:.6f} seconds")
        print(f"  Total time:    {parallel_forward_time + parallel_backward_time:.6f} seconds")
        print(f"\nSpeedup:")
        single_total = single_forward_time + single_backward_time
        parallel_total = parallel_forward_time + parallel_backward_time
        speedup = single_total / parallel_total if parallel_total > 0 else 0
        print(f"  Forward speedup:  {single_forward_time / parallel_forward_time:.2f}x")
        print(f"  Backward speedup: {single_backward_time / parallel_backward_time:.2f}x")
        print(f"  Total speedup:    {speedup:.2f}x")

        # Calculate throughput
        total_tokens = seqlen * batch_size
        single_throughput = total_tokens / single_total
        parallel_throughput = total_tokens / parallel_total
        print(f"\nThroughput:")
        print(f"  Single mode:   {single_throughput:.0f} tokens/sec")
        print(f"  Parallel mode: {parallel_throughput:.0f} tokens/sec")
        print("="*80 + "\n")


if __name__ == "__main__":
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

    if args.list_configs:
        print("Available Ring Attention Configurations:")
        print("=" * 50)

        for category in ["small", "medium", "large", "gqa"]:
            print(f"\n{category.upper()} CONFIGS:")
            configs = get_configs_by_category(category)
            if configs:  # Only print if category has configs
                for name, config in configs.items():
                    tokens_k = config.total_tokens // 1000
                    gqa_info = f" (GQA {config.num_heads}->{config.num_kv_heads})" if config.is_gqa else ""
                    causal_info = " [Causal]" if config.causal else " [Non-causal]"
                    window_info = f" [Window={config.window_size[0]},{config.window_size[1]}]" if config.window_size != (-1, -1) else ""
                    print(f"  {name:20s} - {config.batch_size}x{config.num_heads}x{config.head_dim}, seq={config.max_seqlen}, tokens={tokens_k}K, {config.dtype}{gqa_info}{causal_info}{window_info}")
            else:
                print("  No configurations in this category")

        print(f"\nDEFAULT PERFORMANCE CONFIGS: {DEFAULT_PERFORMANCE_CONFIGS}")
        print(f"\nUsage: python {__file__} --config <config_name>")
    else:
        run_performance_benchmark(
            config_name=args.config,
            dtype=args.dtype,
            seqlen=args.seqlen,
            nheads=args.nheads,
            head_dim=args.head_dim,
            batch_size=args.batch_size,
            timing_method=args.timing_method,
            warmup_runs=args.warmup_runs,
            timing_runs=args.timing_runs,
        )
