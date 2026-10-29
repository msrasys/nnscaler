import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch

import nnscaler
from nnscaler.policies import OpPlan, get_pas_ops
from nnscaler.runtime.module import CubeModule
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0


def shard():
    model = CubeModule()
    model.weight = torch.nn.Parameter(torch.arange(8).reshape(2, 4).float())
    model.register_buffer('replica', torch.zeros(2, 4), persistent=False)
    model.add_full_map('weight', 0, True, 'weight', (4, 4), (slice(2, 4), slice(0, 4)), 1)
    model.add_full_map('replica', 1, False, 'reference.weight', (4, 4), (slice(2, 4), slice(0, 4)), 1)
    return model


def test_snapshot_and_full_checkpoint_slicing():
    model = shard()
    aliases = {'reference.weight': 'weight'}
    model.init_aliased_buffers(aliases)
    torch.testing.assert_close(model.replica, model.weight)
    with torch.no_grad():
        model.weight.add_(100)
    assert not torch.equal(model.replica, model.weight)
    state = {'weight': torch.arange(16).reshape(4, 4).float()}
    model.init_aliased_buffers(aliases, state)
    torch.testing.assert_close(model.replica, state['weight'][2:])
    assert 'replica' not in model.state_dict()
    assert not model.replica.requires_grad


@pytest.mark.parametrize('state,exception', [({}, RuntimeError), ({'weight': torch.ones(8, 4)}, ValueError)])
def test_replica_rejects_missing_or_incompatible_weights(state, exception):
    model = shard()
    with pytest.raises(exception):
        model.init_aliased_buffers({'reference.weight': 'weight'}, state)
    assert not model.replica.any()


def test_non_persistent_replica_resume_pages_in_local_slice(tmp_path):
    model = shard()
    full = torch.arange(16).reshape(4, 4).float()
    filename = tmp_path / 'npbuffer.pt'
    torch.save({1: full}, filename)
    with patch('nnscaler.runtime.module.torch.load', wraps=torch.load) as reader:
        model.load_np_buffer_content(str(filename))
    assert reader.call_args.kwargs['mmap'] is True
    assert reader.call_args.kwargs['map_location'] == 'cpu'
    torch.testing.assert_close(model.replica, full[2:])


class PairedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
        self.reference = torch.nn.Module()
        self.reference.register_buffer('weight', self.weight.detach().clone(), persistent=False)

    def forward(self, x, ref):
        with torch.no_grad():
            ref = torch.nn.functional.linear(ref, self.reference.weight).tanh()
            log_q = ref.log_softmax(-1)
        x = torch.nn.functional.linear(x, self.weight).tanh()
        aux = torch.nn.functional.kl_div(x.log_softmax(-1), log_q, log_target=True, reduction='sum')
        return x, ref, aux


class PairedPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([PairedBlock() for _ in range(4)])

    def forward(self, x):
        ref = x.detach()
        loss = 0.0
        for layer in self.layers:
            x, ref, aux = layer(x, ref)
            loss = loss + aux
        return x.square().sum() + loss


def policy(graph, cfg):
    interleaved = cfg.pas_config['pipeline_scheduler'] == '1f1b_interleaved'
    for op in get_pas_ops(graph):
        stage = -1
        if PairedBlock in op.module_class_chain:
            index = int(op.get_module_fqn(PairedBlock).split('.')[-1])
            stage = index if interleaved else index // 2
        yield OpPlan(op, stage_id=stage, partition=None)


def worker(scheduler):
    nnscaler.init()
    torch.manual_seed(7)
    source = PairedPipeline()
    eager = copy.deepcopy(source).cuda()
    eager_opt = torch.optim.SGD(eager.parameters(), lr=0.01)
    cfg = nnscaler.ComputeConfig(2, 2, use_end2end=True, constant_folding=False,
                                pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                            'pipeline_scheduler': scheduler})
    directory = Path(tempfile.gettempdir()) / f'paired_ref_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = nnscaler.parallelize(source, {'x': torch.ones(8, 8)}, policy, cfg,
                                    gen_savedir=tempdir, reuse='override').cuda()
        aliases = {m.orig_name: m.orig_name.replace('.reference.', '.')
                   for m in model.fullmap.values() if '.reference.' in m.orig_name}
        model.init_aliased_buffers(aliases)
        frozen = {name: value.clone() for name, value in model.get_non_persistent_buffers().items()}
        opt = nnscaler.build_optimizer(model, torch.optim.SGD, lr=0.01)
        eager_params = dict(eager.named_parameters())
        for step in range(2):
            samples = [torch.arange(64, device='cuda').reshape(8, 8).float() / 64 + micro
                       for micro in range(4)]
            eager_opt.zero_grad()
            expected = [eager(sample) for sample in samples]
            torch.stack(expected).sum().backward()
            torch.testing.assert_close(model.train_step(samples), expected)
            for name, meta in model.fullmap.items():
                if meta.is_param:
                    torch.testing.assert_close(getattr(model, name).grad,
                                               eager_params[meta.orig_name].grad[meta.slicers])
            opt.step()
            eager_opt.step()
            for name, meta in model.fullmap.items():
                if meta.is_param:
                    torch.testing.assert_close(getattr(model, name), eager_params[meta.orig_name][meta.slicers])
            for name, value in frozen.items():
                torch.testing.assert_close(getattr(model, name), value, rtol=0, atol=0)
                assert name not in model.state_dict()
            opt.zero_grad()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('scheduler', ['1f1b', '1f1b_interleaved'])
def test_pipeline_frozen_stream_matches_eager(scheduler):
    assert all(launch_torchrun(2, worker, scheduler).values())
