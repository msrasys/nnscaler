#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Base runner framework for ring attention correctness tests.
This module provides common functionality for both ring_attn and ring_attn_varlen test runners.
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union

import torch
import torch.distributed as dist
import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType

from nnscaler.customized_ops.ring_attention.core.utils import set_seed, log
from configs import get_config


class RingAttnRunnerBase(ABC):
    """Base class for ring attention test runners"""

    @property
    @abstractmethod
    def function_signature(self) -> str:
        """Return the function signature to look for in the graph"""
        pass

    @property
    @abstractmethod
    def function_name(self) -> str:
        """Return the function name for partitioning"""
        pass

    @abstractmethod
    def create_test_module(self, config) -> torch.nn.Module:
        """Create the test module with the appropriate configuration"""
        pass

    @abstractmethod
    def prepare_inputs(self, config, device, torch_dtype):
        """Prepare input tensors based on the configuration and attention type"""
        pass

    @abstractmethod
    def run_single_gpu_reference(self, inputs, config):
        """Run single GPU reference implementation"""
        pass

    @abstractmethod
    def get_dummy_forward_args(self, inputs) -> Dict[str, Any]:
        """Get dummy forward arguments for model parallelization"""
        pass

    def create_policy(self) -> callable:
        """Create partitioning policy for the specific attention type"""
        def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
            ngpus = resource.plan_ngpus
            partitioned = False
            for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
                if not partitioned and node.signature == self.function_signature:
                    print(f'\nPartitioned node: {node}\n')
                    sub_nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=ngpus)
                    partitioned = True
                else:
                    sub_nodes = graph.replicate(node, times=ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            if not partitioned:
                print(f"WARNING: No {self.function_name} found in graph for partitioning")
            return graph
        return policy

    def initialize_distributed(self):
        """Initialize distributed environment"""
        # Check CUDA availability first
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available")
            sys.exit(1)
        
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        
        # Check if we have enough GPUs
        available_gpus = torch.cuda.device_count()
        if available_gpus < world_size:
            print(f"ERROR: Test requires {world_size} GPUs, but only {available_gpus} available")
            sys.exit(1)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            device_count = torch.cuda.device_count()
            device = rank % device_count
            try:
                torch.cuda.set_device(device)
            except Exception as e:
                print(f"ERROR: Failed to set CUDA device {device}: {e}")
                sys.exit(1)

        print(f"[INFO] world_size:{world_size}, rank:{rank}, available_gpus:{available_gpus}")
        
        try:
            dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        except Exception as e:
            print(f"ERROR: Failed to initialize process group: {e}")
            sys.exit(1)

        # Initialize nnscaler
        nnscaler.init()
        return world_size, rank

    def get_tolerances(self, dtype: str) -> Dict[str, float]:
        """Get tolerance values based on data type"""
        if dtype == "bf16":
            return dict(atol=2.5e-2, rtol=2.5e-2)
        elif dtype == "fp16":
            return dict(atol=5e-3, rtol=5e-3)
        else:
            return dict(atol=2.5e-2, rtol=2.5e-2)

    def print_debug_info(self, single_out, para_out, single_grads, para_grads, rank_id):
        """Print debug information when correctness test fails"""
        if rank_id == 0:
            print("✗ Correctness test FAILED!")
            # Print detailed error information
            log("single out", single_out, rank0_only=True)
            log("multi  out", para_out, rank0_only=True)
            log("out   diff", single_out - para_out, rank0_only=True)

            for i, (single_grad, para_grad, name) in enumerate(zip(single_grads, para_grads, ['q', 'k', 'v'])):
                log(f"single  d{name}", single_grad, rank0_only=True)
                log(f"multi   d{name}", para_grad, rank0_only=True)
                log(f"d{name}    diff", single_grad - para_grad, rank0_only=True)

    def print_success_info(self, rank_id, config_name=None):
        """Print success information"""
        if rank_id == 0:
            config_suffix = f" for config '{config_name}'" if config_name else ""
            print(f"✓ Correctness test PASSED{config_suffix}!")

    def run_correctness_test(self, config_name: str, dtype: str = "bf16", **kwargs):
        """Run correctness test with the specific attention implementation"""
        # Initialize distributed
        world_size, rank = self.initialize_distributed()
        rank_id = torch.distributed.get_rank()

        # Get configuration
        config = get_config(config_name)
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        if rank_id == 0:
            print(f"Testing {self.function_name} correctness")
            print(f"Configuration: {config.name}")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Sequence length: {config.max_seqlen}")
            print(f"  Num heads: {config.num_heads}")
            print(f"  KV heads: {config.num_kv_heads}")
            print(f"  Head dim: {config.head_dim}")
            print(f"  Data type: {dtype}")
            print(f"  World size: {world_size}")
            print("=" * 60)

        # Set seed for reproducibility
        set_seed(42 + rank_id)
        device = torch.device(f"cuda:{rank_id}")

        # Prepare inputs (implementation-specific)
        inputs = self.prepare_inputs(config, device, torch_dtype)

        # Broadcast inputs to ensure consistency across ranks
        for tensor in inputs.values():
            if isinstance(tensor, torch.Tensor):
                dist.broadcast(tensor, src=0)
        dist.barrier()

        # Setup models
        model = self.create_test_module(config)

        # Create parallel model
        dummy_args = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    dummy_args[k] = v.detach().clone().requires_grad_()
                else:
                    dummy_args[k] = v.detach().clone()
            else:
                dummy_args[k] = v

        parallel_model = parallelize(
            model,
            dummy_forward_args=self.get_dummy_forward_args(dummy_args),
            pas_policy=self.create_policy(),
            compute_config=ComputeConfig(world_size, world_size),
            reuse=ReuseType.OVERRIDE
        )
        parallel_model = parallel_model.cuda()
        parallel_model.train()

        # Run correctness test
        print("Running correctness test..." if rank_id == 0 else "", end="")

        # Single mode for reference
        single_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    single_inputs[k] = v.detach().clone().requires_grad_()
                else:
                    single_inputs[k] = v.detach().clone()
            else:
                single_inputs[k] = v

        single_out, single_grad_tensors = self.run_single_gpu_reference(single_inputs, config)

        # Create gradient for backward pass
        dout = torch.randn_like(single_out, device=device, dtype=torch_dtype)
        # Ensure dout is consistent across all ranks
        dist.broadcast(dout, src=0)
        single_out.backward(dout)

        # Extract single gradients
        single_grads = [tensor.grad for tensor in single_grad_tensors]

        # Parallel mode for correctness
        para_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    para_inputs[k] = v.detach().clone().requires_grad_()
                else:
                    para_inputs[k] = v.detach().clone()
            else:
                para_inputs[k] = v

        para_out = parallel_model(**para_inputs)
        para_out.backward(dout)
        parallel_model.sync_grad()

        # Extract gradients for q, k, v tensors
        para_grads = [para_inputs[k].grad for k in ['q', 'k', 'v']]

        print(" Done!" if rank_id == 0 else "")

        # Check correctness with tolerances
        tols = self.get_tolerances(dtype)

        # Verify outputs and gradients
        try:
            torch.testing.assert_close(single_out, para_out, **tols)
            for single_grad, para_grad in zip(single_grads, para_grads):
                torch.testing.assert_close(single_grad, para_grad, **tols)

            self.print_success_info(rank_id, config_name)

        except AssertionError as e:
            self.print_debug_info(single_out, para_out, single_grads, para_grads, rank_id)
            raise e

        dist.destroy_process_group()

    def main(self, **kwargs):
        """Main entry point for the test runner"""
        # Filter out torch.distributed.launch arguments
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith('--'):
                # Remove leading '--' from argument names
                k = k[2:].replace('-', '_')
            if k not in ['local_rank', 'local-rank']:  # Filter out torch.distributed.launch args
                filtered_kwargs[k] = v

        # Convert string arguments back to appropriate types
        for numeric_arg in ['batch_size', 'num_heads', 'head_dim', 'max_seqlen']:
            if numeric_arg in filtered_kwargs and filtered_kwargs[numeric_arg] is not None:
                filtered_kwargs[numeric_arg] = int(filtered_kwargs[numeric_arg])

        for float_arg in ['rtol', 'atol']:
            if float_arg in filtered_kwargs and filtered_kwargs[float_arg] is not None:
                filtered_kwargs[float_arg] = float(filtered_kwargs[float_arg])

        self.run_correctness_test(**filtered_kwargs)