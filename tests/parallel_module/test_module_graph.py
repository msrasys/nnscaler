# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import pytest
import torch
import torch.distributed as dist

import nnscaler
from nnscaler import ComputeConfig, ModuleStreamRule, parallelize
from nnscaler.policies import pas_dp
from .test_weight_backward_streams import ready_anchor, ReadyAnchor, submit_ready_anchor, complete_ready_anchor
from ..utils import replace_all_device_with


@nnscaler.register_op('n h -> n h')
def combine_anchor(x):
    return ReadyAnchor.apply(x, x.detach()*2)


nnscaler.register_async_op(combine_anchor, submit=submit_ready_anchor, complete=complete_ready_anchor)


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.gate = torch.nn.Linear(8, 8)
        self.down = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.down(self.linear(x).sin() * self.gate(x))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = Branch()
        self.gate = torch.nn.Linear(8, 8)

    def forward(self, x):
        main = ready_anchor(x)
        main = combine_anchor(main.cos())
        return main + self.branch(x) + self.gate(x)


class EarlyPreparationModel(Model):
    def forward(self, x):
        value = x.relu()
        main = ready_anchor(value.tanh())
        return main + self.branch(value)


@replace_all_device_with('cpu', force=True)
def test_graph_preparation_moves_before_independent_main_work(tmp_path):
    rule = ModuleStreamRule('branch', stream='shared', launch_after='*.ready_anchor',
                            overlap_backward=True, cuda_graph='forward')
    parallelize(EarlyPreparationModel(), {'x': torch.randn(4, 8, requires_grad=True)}, pas_dp,
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path, load_module=False)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    assert source.index('.prepare(') < source.index('torch.tanh(') < source.index('.capture(')
    assert source.index('.capture(') < source.index('.submit(')
    assert source.index('.submit(') < source.index('.fork(') < source.index('run_module_graph(')


@pytest.mark.parametrize('boundary', ['ordinary', 'mutation', 'checkpoint'])
def test_graph_preparation_preserves_mutation_and_checkpoint_boundaries(boundary):
    from nnscaler.ir.operator import IRFwOperation
    from nnscaler.ir.tensor import IRFullTensor
    from nnscaler.graph.schedule.module_streams import ModuleStreamRegion, plan_module_graph_preparation

    def node(signature, inputs):
        operation = IRFwOperation(signature.rsplit('.', 1)[-1], signature, inputs, 1)
        operation.set_output(0, IRFullTensor((4, 8), requires_grad=True))
        return operation

    producer = node('torch.relu', [IRFullTensor((4, 8), requires_grad=True)])
    value = producer.output(0)
    work = node('torch.add_' if boundary == 'mutation' else 'torch.sin', [value])
    if boundary == 'checkpoint':
        producer.recompute = work.recompute = 7
    anchor = node('test.async', [work.output(0)])
    branch = node('torch.cos', [value])
    region = ModuleStreamRegion('side', 'shared', [branch], (value,), (branch.output(0),),
                                ready_before=anchor, cuda_graph='forward')
    points = plan_module_graph_preparation([producer, work, anchor, branch], {'side': region})
    assert points['side'] is (work if boundary == 'ordinary' else anchor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_coalesced_replay_preserves_bf16_input_gradient_accumulation(tmp_path):
    from nnscaler.runtime.module_graph import prepare_module_graphs
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/pg', rank=0, world_size=1)
    try:
        nnscaler.init()
        torch.manual_seed(512)
        model = Model().cuda().bfloat16()
        shared = dict(stream='shared', launch_after='*.ready_anchor', overlap_backward=True,
                      backward_on='*.combine_anchor', cuda_graph='forward')
        rules = (ModuleStreamRule('gate', **shared),
                 ModuleStreamRule('branch', split_before='down', resume_after='*.combine_anchor', **shared))
        x = torch.randn(4, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        generated = parallelize(model, {'x': x}, pas_dp, ComputeConfig(1, 1, module_streams=rules),
                                gen_savedir=tmp_path, instance_name='bf16_coalesced_order')
        expected = generated(x)
        expected.float().square().sum().backward()
        reference = [p.grad.clone() for p in (x, *generated.parameters())]
        prepare_module_graphs(generated)
        for p in (x, *generated.parameters()):
            p.grad = None
        actual = generated(x)
        actual.float().square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close([p.grad for p in (x, *generated.parameters())], reference, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('capture_mode', [True, 'forward'])
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('coalesce', [False, True])
def test_generated_graph_regions_calibrate_then_train_after_weight_updates(tmp_path, monkeypatch, workers, split, capture_mode, coalesce):
    from nnscaler.runtime.module_graph import _profiles
    from nnscaler.flags import RuntimeFlag
    # build_optimizer configures global accumulation flags for its own loop.
    monkeypatch.setattr(RuntimeFlag, 'skip_reducer', RuntimeFlag.skip_reducer)
    monkeypatch.setattr(RuntimeFlag, 'skip_zero_grad', RuntimeFlag.skip_zero_grad)
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/pg', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = Model().cuda()
        reference = copy.deepcopy(model)
        rule = ModuleStreamRule('branch', stream='shared', launch_after='*.ready_anchor',
                                overlap_backward=True, cuda_graph=capture_mode,
                                **(dict(split_before='down', resume_after='*.combine_anchor',
                                        backward_on='*.combine_anchor') if split else {}))
        rules = (rule,)
        if coalesce:
            rules = (ModuleStreamRule('gate', stream='shared', launch_after='*.ready_anchor',
                                      overlap_backward=True, cuda_graph=capture_mode,
                                      backward_on='*.combine_anchor' if split else None), rule)
        generated = parallelize(model, {'x': torch.randn(4,8,device='cuda',requires_grad=True)}, pas_dp,
            ComputeConfig(1,1,module_streams=rules), gen_savedir=tmp_path,
            instance_name=f'graphed_branch_{workers}_{split}_{capture_mode}_{coalesce}', codegen_workers=workers)
        source = next(tmp_path.rglob('gencode0.py')).read_text()
        assert source.count('nnscaler.runtime.module_graph.run_module_graph(') == (2 if split else 1)
        optimizer = nnscaler.build_optimizer(generated, torch.optim.SGD, lr=.001)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.001)
        for step in range(3):
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            if step:
                assert _profiles[generated] and all(p.pool is not None and len(p.pool.slots)==3 for p in _profiles[generated].values())
            inputs = [torch.randn(4,8,device='cuda',requires_grad=True) for _ in range(3)]
            refs = [x.detach().clone().requires_grad_() for x in inputs]
            actual, expected = [generated(x) for x in inputs], [reference(x) for x in refs]
            sum(y.square().mean() for y in actual).backward()
            sum(y.square().mean() for y in expected).backward()
            optimizer.sync_shard_grad()
            torch.testing.assert_close(actual, expected)
            for x,ref in zip(inputs,refs):torch.testing.assert_close(x.grad,ref.grad)
            for name,p in generated.named_parameters():
                torch.testing.assert_close(p.grad,reference.get_parameter(generated.fullmap[name].orig_name).grad)
            optimizer.step()
            ref_optimizer.step()
        assert all(not p.function.__code__.co_freevars for p in _profiles[generated].values())
    finally:
        torch.cuda.synchronize()
        nnscaler.uninit()
        if dist.is_initialized():dist.destroy_process_group()
