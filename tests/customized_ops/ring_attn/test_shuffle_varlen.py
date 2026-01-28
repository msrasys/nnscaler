#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Simple test for shuffle_varlen and unshuffle_varlen functions.
"""

import pytest
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import List
from functools import partial

from tests.launch_torchrun import torchrun


# Skip all tests if flash_attn_func is not available
try:
    from flash_attn import flash_attn_func
except ImportError:
    pytest.skip("flash_attn_func not available", allow_module_level=True)


@dataclass
class ShuffleVarlenConfig:
    """Simple test configuration"""
    name: str
    batch_size: int
    seq_lens: List[int]
    hidden_dim: int


# Test configurations
CONFIGS = {
    "tiny": ShuffleVarlenConfig("tiny", 2, [512, 768], 64),
    "small": ShuffleVarlenConfig("small", 2, [1024, 1536], 128),
    "medium": ShuffleVarlenConfig("medium", 2, [1024, 1536], 256),
    "uneven": ShuffleVarlenConfig("uneven", 3, [256, 768, 1024], 128),
}


def shuffle_varlen_test(config_name="tiny", dtype="float32", world_size=2):
    """Test shuffle_varlen and unshuffle_varlen functions"""

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size_actual = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Testing shuffle_varlen and unshuffle_varlen functions")
        print(f"Configuration: {config_name}")
        print(f"World size: {world_size_actual}")
        print(f"Data type: {dtype}")
        print("=" * 60)

    # Get configuration
    config = CONFIGS[config_name]

    # Set up process group for context parallel
    cp_ranks = list(range(world_size_actual))
    cp_group = dist.new_group(cp_ranks)

    # Create cumulative sequence lengths (padded to be divisible by 2*world_size)
    cu_seqlens = torch.zeros(config.batch_size + 1, dtype=torch.int32, device=device)
    total_slices_per_seq = 2 * world_size_actual

    for i, seq_len in enumerate(config.seq_lens):
        # Pad sequence length to be divisible by total_slices_per_seq
        padded_seq_len = ((seq_len + total_slices_per_seq - 1) // total_slices_per_seq) * total_slices_per_seq
        cu_seqlens[i + 1] = cu_seqlens[i] + padded_seq_len

    total_seq_len = cu_seqlens[len(config.seq_lens)].item()  # Use len(config.seq_lens) instead of -1

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype)

    # Import functions from varlen_utils
    from nnscaler.customized_ops.ring_attention.varlen_utils import shuffle_varlen, unshuffle_varlen

    if rank == 0:
        print("Running shuffle/unshuffle correctness tests...")

    tolerance = 1e-5 if torch_dtype == torch.float32 else 1e-2

    # Test 1: 1D tensor (like position_ids)
    if rank == 0:
        print("  Test: 1D tensor (total_seq_len,)...")

    try:
        # Create full tensor first (on rank 0)
        if rank == 0:
            full_tensor_1d = torch.arange(total_seq_len, dtype=torch_dtype, device=device)
        else:
            full_tensor_1d = torch.empty(total_seq_len, dtype=torch_dtype, device=device)

        # Broadcast full tensor to all ranks for reference
        dist.broadcast(full_tensor_1d, src=0, group=cp_group)

        # Split tensor for local input (each rank gets a chunk)
        chunk_size = total_seq_len // world_size_actual
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size_actual - 1 else total_seq_len
        local_tensor_1d = full_tensor_1d[start_idx:end_idx].clone()

        # Test shuffle -> unshuffle
        shuffled = shuffle_varlen(local_tensor_1d, cu_seqlens, cp_ranks, cp_group)
        unshuffled = unshuffle_varlen(shuffled, cu_seqlens, cp_ranks, cp_group)

        # Compare with original local chunk
        if torch.allclose(local_tensor_1d, unshuffled, atol=tolerance):
            if rank == 0:
                print("    ✓ 1D tensor test passed")
        else:
            if rank == 0:
                print("    ✗ 1D tensor test FAILED")
            raise AssertionError("1D tensor test failed")

    except Exception as e:
        if rank == 0:
            print(f"    ✗ 1D tensor test FAILED with error: {e}")
        raise e

    # Test 2: 2D tensor (total_seq_len, hidden_dim)
    if rank == 0:
        print("  Test: 2D tensor (total_seq_len, hidden_dim)...")

    try:
        # Create full tensor first (on rank 0)
        if rank == 0:
            full_tensor_2d = torch.randn(total_seq_len, config.hidden_dim, dtype=torch_dtype, device=device)
        else:
            full_tensor_2d = torch.empty(total_seq_len, config.hidden_dim, dtype=torch_dtype, device=device)

        # Broadcast full tensor to all ranks for reference
        dist.broadcast(full_tensor_2d, src=0, group=cp_group)

        # Split tensor for local input (each rank gets a chunk)
        chunk_size = total_seq_len // world_size_actual
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size_actual - 1 else total_seq_len
        local_tensor_2d = full_tensor_2d[start_idx:end_idx].clone()

        # Test shuffle -> unshuffle
        shuffled = shuffle_varlen(local_tensor_2d, cu_seqlens, cp_ranks, cp_group)
        unshuffled = unshuffle_varlen(shuffled, cu_seqlens, cp_ranks, cp_group)

        # Compare with original local chunk
        if torch.allclose(local_tensor_2d, unshuffled, atol=tolerance):
            if rank == 0:
                print("    ✓ 2D tensor test passed")
        else:
            if rank == 0:
                print("    ✗ 2D tensor test FAILED")
            raise AssertionError("2D tensor test failed")

    except Exception as e:
        if rank == 0:
            print(f"    ✗ 2D tensor test FAILED with error: {e}")
        raise e

    dist.barrier()

    if rank == 0:
        print("✓ All shuffle/unshuffle tests PASSED!")

    dist.destroy_process_group()


class TestShuffleVarlen:
    """Simple test class for shuffle/unshuffle varlen"""

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_shuffle_varlen_tiny(self, dtype):
        """Test shuffle/unshuffle varlen with tiny configuration"""
        partial(torchrun, 2, shuffle_varlen_test, "tiny", dtype)()

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_shuffle_varlen_small(self, dtype):
        """Test shuffle/unshuffle varlen with small configuration"""
        partial(torchrun, 2, shuffle_varlen_test, "small", dtype)()

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_shuffle_varlen_medium(self, dtype):
        """Test shuffle/unshuffle varlen with medium configuration"""
        partial(torchrun, 2, shuffle_varlen_test, "medium", dtype)()

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_shuffle_varlen_uneven(self, dtype):
        """Test shuffle/unshuffle varlen with uneven sequence lengths"""
        partial(torchrun, 2, shuffle_varlen_test, "uneven", dtype)()

    @pytest.mark.parametrize("num_gpus", [2, 4])
    def test_shuffle_varlen_multi_gpu(self, num_gpus):
        """Test shuffle/unshuffle varlen on multiple GPUs"""
        partial(torchrun, num_gpus, shuffle_varlen_test, "tiny", "float32")()


# Standalone test functions for pytest discovery
@pytest.mark.parametrize("config,dtype", [
    ("tiny", "float32"), ("tiny", "float16"),
    ("small", "float32"), ("small", "float16"),
    ("uneven", "float32"), ("uneven", "float16"),
])
def test_shuffle_varlen_correctness(config, dtype):
    """Test shuffle/unshuffle varlen correctness"""
    partial(torchrun, 2, shuffle_varlen_test, config, dtype)()


@pytest.mark.parametrize("config,num_gpus", [
    ("tiny", 2), ("tiny", 4),
    ("small", 2), ("small", 4),
])
def test_shuffle_varlen_multi_gpu(config, num_gpus):
    """Test shuffle/unshuffle varlen on multiple GPUs"""
    partial(torchrun, num_gpus, shuffle_varlen_test, config, "float32")()