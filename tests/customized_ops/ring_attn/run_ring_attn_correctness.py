#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Ring Attention Correctness Test Runner Script

This script runs ring attention correctness tests in a distributed environment.
It compares the outputs of single-GPU and multi-GPU ring attention to ensure correctness.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

# Import ring attention implementation
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_func
from nnscaler.customized_ops.ring_attention.core.utils import set_seed, log

# Import configurations
from configs import get_config


class TestModule(torch.nn.Module):
    """Test module for ring attention"""
    def __init__(self, causal=True, window_size=(-1, -1)):
        super(TestModule, self).__init__()
        self.causal = causal
        self.window_size = window_size

    def forward(self, q, k, v):
        result = wrap_ring_attn_func(
            q, k, v,
            # causal=self.causal,
            # window_size=self.window_size
        )
        return result


def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if not partitioned and node.signature == 'nnscaler.customized_ops.ring_attention.ring_attn.wrap_ring_attn_func':
            print(f'\nPartitioned node: {node}\n')
            sub_nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=ngpus)  # Partition on sequence dimension
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    if not partitioned:
        print("WARNING: No wrap_ring_attn_func found in graph for partitioning")
    return graph


def run_correctness_test(
    config_name: str,
    dtype: str = "bf16",
    rtol: float = None,
    atol: float = None,
):
    """Run correctness test for ring attention"""

    # Get world size and rank from environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))

    # Initialize distributed environment
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Initialize nnscaler
    nnscaler.init()
    rank_id = torch.distributed.get_rank()

    # Get configuration
    config = get_config(config_name)

    # Set data type
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    if rank_id == 0:
        print(f"Testing ring attention correctness")
        print(f"Configuration: {config.name}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Sequence length: {config.max_seqlen}")
        print(f"  Num heads: {config.num_heads}")
        print(f"  KV heads: {config.num_kv_heads}")
        print(f"  Head dim: {config.head_dim}")
        print(f"  Data type: {dtype}")
        print(f"  Causal: {config.causal}")
        print(f"  Window size: {config.window_size}")
        print(f"  World size: {world_size}")
        print("=" * 60)

    # Set seed for reproducibility
    set_seed(42 + rank_id)
    device = torch.device(f"cuda:{rank_id}")

    # Create input tensors with shape [batch_size, seq_len, num_heads, head_dim]
    q = torch.randn(
        config.batch_size,
        config.max_seqlen,
        config.num_heads,
        config.head_dim,
        device=device,
        dtype=torch_dtype,
        requires_grad=True
    )

    k = torch.randn(
        config.batch_size,
        config.max_seqlen,
        config.num_kv_heads,
        config.head_dim,
        device=device,
        dtype=torch_dtype,
        requires_grad=True
    )

    v = torch.randn(
        config.batch_size,
        config.max_seqlen,
        config.num_kv_heads,
        config.head_dim,
        device=device,
        dtype=torch_dtype,
        requires_grad=True
    )

    # Broadcast inputs to ensure consistency across ranks
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.barrier()

    # Setup models
    model = TestModule(causal=config.causal, window_size=config.window_size)

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
        },
        pas_policy=policy,
        compute_config=ComputeConfig(world_size, world_size),
        reuse=ReuseType.OVERRIDE
    )
    parallel_model = parallel_model.cuda()
    parallel_model.train()

    # Run correctness test
    print("Running correctness test..." if rank_id == 0 else "", end="")

    # Single mode for reference (call wrap_ring_attn_func directly)
    q_single = q.detach().clone().requires_grad_()
    k_single = k.detach().clone().requires_grad_()
    v_single = v.detach().clone().requires_grad_()

    # Run single GPU version (this should call flash_attn internally when no process_group)
    single_out = wrap_ring_attn_func(q_single, k_single, v_single,)
    # single_out = wrap_ring_attn_func(q_single, k_single, v_single, causal=config.causal, window_size=config.window_size)
    single_loss = single_out.sum()
    single_loss.backward()

    # Parallel mode for correctness
    qq_corr = q.detach().clone().requires_grad_()
    kk_corr = k.detach().clone().requires_grad_()
    vv_corr = v.detach().clone().requires_grad_()

    para_out = parallel_model(qq_corr, kk_corr, vv_corr)
    para_loss = para_out.sum()
    para_loss.backward()
    parallel_model.sync_grad()

    print(" Done!" if rank_id == 0 else "")

    # Check correctness with dtype-based tolerances (like varlen test)
    if dtype == "bf16":
        default_tols = dict(atol=2.5e-2, rtol=2.5e-2)
    elif dtype == "fp16":
        default_tols = dict(atol=5e-3, rtol=5e-3)
    else:
        default_tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # Use custom tolerances if provided, otherwise use defaults
    tols = dict(
        atol=atol if atol is not None else default_tols['atol'],
        rtol=rtol if rtol is not None else default_tols['rtol']
    )

    # Verify outputs and gradients
    try:
        torch.testing.assert_close(single_out, para_out, **tols)
        torch.testing.assert_close(q_single.grad, qq_corr.grad, **tols)
        torch.testing.assert_close(k_single.grad, kk_corr.grad, **tols)
        torch.testing.assert_close(v_single.grad, vv_corr.grad, **tols)

        if rank_id == 0:
            # Compute differences for reporting
            output_diff = torch.abs(single_out - para_out)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()

            # Check relative differences
            rel_diff = output_diff / (torch.abs(single_out) + 1e-8)
            max_rel_diff = torch.max(rel_diff).item()

            print("\n" + "="*60)
            print("CORRECTNESS TEST RESULTS")
            print("="*60)
            print(f"Output comparison:")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Max relative difference: {max_rel_diff:.2e}")
            print(f"  Tolerance (absolute): {tols['atol']:.2e}")
            print(f"  Tolerance (relative): {tols['rtol']:.2e}")
            print("✅ CORRECTNESS TEST PASSED")
            print("="*60)

    except Exception as e:
        if rank_id == 0:
            print("\n" + "="*60)
            print("❌ CORRECTNESS TEST FAILED")
            print("="*60)
            print(f"Error: {str(e)}")

            # Print some debug information
            output_diff = torch.abs(single_out - para_out)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()
            rel_diff = output_diff / (torch.abs(single_out) + 1e-8)
            max_rel_diff = torch.max(rel_diff).item()

            print(f"Debug Information:")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Max relative difference: {max_rel_diff:.2e}")
            print(f"  Expected tolerance (absolute): {final_atol:.2e}")
            print(f"  Expected tolerance (relative): {final_rtol:.2e}")
            print("="*60)
        raise

    # Synchronize before finishing
    dist.barrier()

    if rank_id == 0:
        print(f"✅ Correctness test completed successfully for config '{config_name}'")


def main(**kwargs):
    # Filter out torch.distributed.launch arguments
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('--'):
            # Remove leading '--' from argument names
            k = k[2:].replace('-', '_')
        if k not in ['local_rank', 'local-rank']:  # Filter out torch.distributed.launch args
            filtered_kwargs[k] = v

    # Convert string arguments back to appropriate types
    if 'rtol' in filtered_kwargs:
        filtered_kwargs['rtol'] = float(filtered_kwargs['rtol'])
    if 'atol' in filtered_kwargs:
        filtered_kwargs['atol'] = float(filtered_kwargs['atol'])

    run_correctness_test(**filtered_kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    main(**kwargs)