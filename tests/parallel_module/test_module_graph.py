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


@nnscaler.register_op('n h -> n h')
def combine_anchor(x):
    return ReadyAnchor.apply(x, x.detach()*2)


nnscaler.register_async_op(combine_anchor, submit=submit_ready_anchor, complete=complete_ready_anchor)


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.down = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.down(self.linear(x).sin())


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = Branch()

    def forward(self, x):
        main = ready_anchor(x)
        main = combine_anchor(main.cos())
        return main + self.branch(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('split', [False, True])
def test_generated_graph_regions_calibrate_then_train_after_weight_updates(tmp_path, monkeypatch, workers, split):
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
                                overlap_backward=True, cuda_graph=True,
                                **(dict(split_before='down', resume_after='*.combine_anchor',
                                        backward_on='*.combine_anchor') if split else {}))
        generated = parallelize(model, {'x': torch.randn(4,8,device='cuda',requires_grad=True)}, pas_dp,
            ComputeConfig(1,1,module_streams=(rule,)), gen_savedir=tmp_path,
            instance_name=f'graphed_branch_{workers}_{split}', codegen_workers=workers)
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
