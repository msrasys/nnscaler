#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
import pytest
from typing import Dict, Tuple, List, Any

import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.graph import IRGraph

from .common import PASMegatron, CubeLinear, init_random, init_distributed, assert_equal
from ..launch_torchrun import launch_torchrun
from ..utils import clear_dir_on_rank0


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleMLP, self).__init__()
        init_random()
        self.register_buffer('buffer', torch.zeros(hidden_dim,))
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x + self.buffer
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def get_tensor_bytesize(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def pas_test_offload(graph: IRGraph, cfg: ComputeConfig):
    ngpus = cfg.plan_ngpus
    auto_multiref(graph)

    batch_dim = 0
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, list(range(ngpus)))

    found_linear = False
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if 'linear' in node.signature and not found_linear:
                found_linear = True
                algo = node.algorithm('dim')
                sub_nodes = graph.partition(
                    node, algo, idx=1, dim=1, num=ngpus)
            else:
                sub_nodes = graph.replicate(node, ngpus)

            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph


def _mem_worker():
    init_distributed()
    bsz, dim = 32, 1024
    compute_config = ComputeConfig(
        plan_ngpus=1,
        runtime_ngpus=2,
    )
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'nnscaler_test_offload_mem') as tempdir:
        module = SimpleMLP(dim, dim, dim)
        p_module = parallelize(
            module,
            {'x': torch.randn(bsz, dim)},
            'dp',
            compute_config,
            gen_savedir=tempdir,
        )

        before_mem = torch.cuda.memory_allocated()
        size_to_free = 0
        for reducer in p_module.reducers:
            assert get_tensor_bytesize(reducer._contiguous_params) == get_tensor_bytesize(reducer._contiguous_grads)
            size_to_free += get_tensor_bytesize(reducer._contiguous_params)

        for buffer in p_module.buffers():
            size_to_free += get_tensor_bytesize(buffer)

        for param in p_module.parameters():
            size_to_free += get_tensor_bytesize(param)

        p_module.offload_params()
        torch.distributed.barrier()
        after_mem = torch.cuda.memory_allocated()
        print(f"Memory before offload: {before_mem}, after offload: {after_mem}, freed: {before_mem - after_mem}")
        print(f"Total size to free: {size_to_free}")

        assert size_to_free == before_mem - after_mem, f"Expected {size_to_free}, but got {before_mem - after_mem}"


def _correctness_worker():
    init_distributed()
    bsz, dim, num_steps = 32, 1024, 5
    compute_config = ComputeConfig(
        plan_ngpus=1,
        runtime_ngpus=2,
    )
    
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'nnscaler_test_offload_correctness') as tempdir:
        # Create test data
        torch.manual_seed(42 + torch.distributed.get_rank())
        test_data = [torch.randn(bsz, dim).cuda() for _ in range(num_steps)]
        
        # Test 1: Normal execution without offload/load
        init_random()
        module1 = SimpleMLP(dim, dim, dim)
        p_module1 = parallelize(
            module1,
            {'x': torch.randn(bsz, dim)},
            'dp',
            compute_config,
            gen_savedir=tempdir,
            instance_name='normal'
        )
        optimizer1 = build_optimizer(p_module1, torch.optim.Adam, lr=0.01)
        
        results_normal = []
        for step, x in enumerate(test_data):
            p_module1.train()
            output = p_module1(x)
            loss = output.sum()
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            # Save intermediate results for comparison
            results_normal.append({
                'loss': loss.detach().cpu(),
                'output': output.detach().cpu(),
                'params': {name: param.detach().cpu().clone() for name, param in p_module1.named_parameters()}
            })
        
        torch.distributed.barrier()
        
        # Test 2: Execution with offload/load
        init_random()
        module2 = SimpleMLP(dim, dim, dim)
        p_module2 = parallelize(
            module2,
            {'x': torch.randn(bsz, dim)},
            'dp',
            compute_config,
            gen_savedir=tempdir,
            instance_name='offload'
        )
        optimizer2 = build_optimizer(p_module2, torch.optim.Adam, lr=0.01)
        
        # First offload to initialize the buffer_shape
        p_module2.offload_params()
        
        results_offload = []
        for step, x in enumerate(test_data):
            # Load params at the beginning of each step
            p_module2.load_params()
            
            p_module2.train()
            output = p_module2(x)
            loss = output.sum()
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            # Save intermediate results for comparison
            results_offload.append({
                'loss': loss.detach().cpu(),
                'output': output.detach().cpu(),
                'params': {name: param.detach().cpu().clone() for name, param in p_module2.named_parameters()}
            })
            
            # Offload params at the end of each step
            p_module2.offload_params()
        
        torch.distributed.barrier()
        
        # Compare results
        for step in range(num_steps):
            normal_result = results_normal[step]
            offload_result = results_offload[step]
            
            # Compare loss
            assert torch.equal(normal_result['loss'], offload_result['loss']), \
                f"Loss mismatch at step {step}: {normal_result['loss']} vs {offload_result['loss']}"
            
            # Compare output
            assert torch.equal(normal_result['output'], offload_result['output']), \
                f"Output mismatch at step {step}"
            
            # Compare parameters
            for param_name in normal_result['params']:
                assert torch.equal(normal_result['params'][param_name], 
                                 offload_result['params'][param_name]), \
                    f"Parameter {param_name} mismatch at step {step}"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_offload_params_mem():
    launch_torchrun(2, _mem_worker)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_offload_params_correctness():
    launch_torchrun(2, _correctness_worker)
