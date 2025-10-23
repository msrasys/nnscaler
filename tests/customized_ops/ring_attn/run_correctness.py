#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import logging
import torch
import torch.distributed as dist
import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType

from nnscaler.customized_ops.ring_attention import wrap_ring_attn_varlen_func
from nnscaler.customized_ops.ring_attention.core.utils import set_seed, log

from configs import get_config


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
        # Check for ring_attn_varlen_func signature
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


def run_ring_attn_correctness_test(
    dtype="bf16",
    config_name=None,
    # Legacy parameters for backward compatibility
    batch_size=None,
    num_heads=None,
    head_dim=None,
    max_seqlen=None,
):
    """Test ring attention variable length correctness"""

    # Initialize distributed
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)

    print(f"[INFO] world_size:{world_size}, rank:{rank}")

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Initialize nnscaler
    nnscaler.init()
    rank_id = torch.distributed.get_rank()

    # Get configuration
    if config_name is not None:
        config = get_config(config_name)
        batch_size = config.batch_size
        num_heads = config.num_heads
        head_dim = config.head_dim
        max_seqlen = config.max_seqlen
        cu_seqlens = config.cu_seqlens
        if rank_id == 0:
            print(f"Using predefined config '{config_name}': {config.name}")
    else:
        # Use provided parameters or defaults
        batch_size = batch_size or 4
        num_heads = num_heads or 12
        head_dim = head_dim or 128
        max_seqlen = max_seqlen or 4096
        cu_seqlens = [0, max_seqlen // 8, max_seqlen // 4, max_seqlen // 2, max_seqlen]
        if rank_id == 0:
            print(f"Using custom config: b={batch_size}, h={num_heads}, d={head_dim}, seq={max_seqlen}")

    set_seed(rank_id)
    device = torch.device(f"cuda:{rank_id}")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    total_seqlen = cu_seqlens[-1]

    # Prepare input data
    q = torch.randn(total_seqlen, num_heads, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(total_seqlen, num_heads, head_dim, device=device, dtype=torch_dtype)
    v = torch.randn(total_seqlen, num_heads, head_dim, device=device, dtype=torch_dtype)

    # Ensure all ranks use the same input
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

    # Run correctness test
    print("Running correctness test..." if rank_id == 0 else "", end="")

    # Single mode for correctness
    q_corr = q.detach().clone().requires_grad_()
    k_corr = k.detach().clone().requires_grad_()
    v_corr = v.detach().clone().requires_grad_()

    single_out = wrap_ring_attn_varlen_func(q_corr, k_corr, v_corr, cu_seqlens_tensor, cu_seqlens_tensor, None)
    single_out.retain_grad()

    dout = torch.randn_like(single_out, device=device, dtype=torch_dtype)
    # Ensure dout is consistent across all ranks
    dist.broadcast(dout, src=0)
    single_out.backward(dout)

    # Parallel mode for correctness
    qq_corr = q.detach().clone().requires_grad_()
    kk_corr = k.detach().clone().requires_grad_()
    vv_corr = v.detach().clone().requires_grad_()

    para_out = parallel_model(qq_corr, kk_corr, vv_corr, cu_seqlens_tensor, cu_seqlens_tensor)
    # Use the same dout tensor for consistent gradient comparison
    para_out.backward(dout)
    parallel_model.sync_grad()

    print(" Done!" if rank_id == 0 else "")

    # Check correctness with tolerances
    if dtype == "bf16":
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    elif dtype == "fp16":
        tols = dict(atol=5e-3, rtol=5e-3)
    else:
        assert False, f"{dtype} is an unsupported dtype!"

    # Verify outputs and gradients
    try:
        torch.testing.assert_close(single_out, para_out, **tols)
        torch.testing.assert_close(q_corr.grad, qq_corr.grad, **tols)
        torch.testing.assert_close(k_corr.grad, kk_corr.grad, **tols)
        torch.testing.assert_close(v_corr.grad, vv_corr.grad, **tols)

        if rank_id == 0:
            print("✓ Correctness test PASSED!")

    except AssertionError as e:
        if rank_id == 0:
            print("✗ Correctness test FAILED!")
            # Print detailed error information
            log("single out", single_out, rank0_only=True)
            log("multi  out", para_out, rank0_only=True)
            log("out   diff", single_out - para_out, rank0_only=True)

            log("single  dq", q_corr.grad, rank0_only=True)
            log("multi   dq", qq_corr.grad, rank0_only=True)
            log("dq    diff", q_corr.grad - qq_corr.grad, rank0_only=True)

            log("single  dk", k_corr.grad, rank0_only=True)
            log("multi   dk", kk_corr.grad, rank0_only=True)
            log("dk    diff", k_corr.grad - kk_corr.grad, rank0_only=True)

            log("single  dv", v_corr.grad, rank0_only=True)
            log("multi   dv", vv_corr.grad, rank0_only=True)
            log("dv    diff", v_corr.grad - vv_corr.grad, rank0_only=True)

        raise e

    dist.destroy_process_group()


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
    if 'batch_size' in filtered_kwargs and filtered_kwargs['batch_size'] is not None:
        filtered_kwargs['batch_size'] = int(filtered_kwargs['batch_size'])
    if 'num_heads' in filtered_kwargs and filtered_kwargs['num_heads'] is not None:
        filtered_kwargs['num_heads'] = int(filtered_kwargs['num_heads'])
    if 'head_dim' in filtered_kwargs and filtered_kwargs['head_dim'] is not None:
        filtered_kwargs['head_dim'] = int(filtered_kwargs['head_dim'])
    if 'max_seqlen' in filtered_kwargs and filtered_kwargs['max_seqlen'] is not None:
        filtered_kwargs['max_seqlen'] = int(filtered_kwargs['max_seqlen'])

    run_ring_attn_correctness_test(**filtered_kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    main(**kwargs)
