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


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_generated_buffer_bundle_is_relocatable_and_loads_only_local_shards(tmp_path, indexed, dtype):
    from nnscaler.graph.parser.frame import Frame, NP_BUFFER_INDEX_KEY
    from nnscaler.ir.tensor import IRFullTensor

    original = tmp_path / 'original'
    original.mkdir()
    frame = Frame()
    full = torch.arange(1024 * 256).reshape(1024, 256).to(dtype)
    if dtype.is_complex:
        full = full + 2j
    tensors = []
    for i in range(3):
        tensor = IRFullTensor(full.shape, dtype=dtype).as_buffer(persistent=False)
        tensors.append(tensor)
        frame.add_attr(tensor, full + i, f'reference.weight{i}')
    stem = original / 'fullmodel.pt'
    frame.save_attr_content(stem, params_per_file=full.numel(), max_workers=2)
    frame.save_np_buffer_content(original / 'npbuffer.pt', attr_file_stem=stem if indexed else None)
    moved = tmp_path / 'moved'
    original.rename(moved)
    filename = moved / 'npbuffer.pt'
    if indexed:
        assert filename.stat().st_size < full.numel() * full.element_size() // 10
        index = torch.load(filename, weights_only=True)
        assert index[NP_BUFFER_INDEX_KEY] == 1

    model = CubeModule()
    model.register_buffer('reference', torch.empty(2, 4, dtype=dtype), persistent=False)
    model.add_full_map('reference', tensors[1].tid, False, 'reference.weight1',
                       tuple(full.shape), (slice(2, 4), slice(3, 7)), 1)
    with patch('nnscaler.runtime.module.torch.load', wraps=torch.load) as reader:
        model.load_np_buffer_content(str(filename))
    torch.testing.assert_close(model.reference, (full + 1)[2:4, 3:7])
    paths = [Path(call.args[0]).name for call in reader.call_args_list]
    assert paths == (['npbuffer.pt', 'fullmodel.pt.1'] if indexed else ['npbuffer.pt'])
    assert all(call.kwargs['map_location'] == 'cpu' for call in reader.call_args_list)
    assert 'reference' not in model.state_dict()


@pytest.mark.parametrize('version,files', [(2, {}), (1, {}), (1, {1: '../outside.pt'})])
def test_buffer_index_rejects_unknown_version_or_missing_local_source(tmp_path, version, files):
    from nnscaler.graph.parser.frame import NP_BUFFER_INDEX_KEY

    filename = tmp_path / 'npbuffer.pt'
    torch.save({NP_BUFFER_INDEX_KEY: version, 'files': files}, filename)
    with pytest.raises(RuntimeError):
        shard().load_np_buffer_content(str(filename))


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
        # A resumed module skips the initial parameter shards, so its frozen
        # buffers must load through the new npbuffer index (legacy archives
        # remain covered by the CPU tests). Preserve the original teacher.
        resumed = type(model)(init_params=False).cuda()
        resumed.load_state_dict(model.state_dict())
        resumed_opt = nnscaler.build_optimizer(resumed, torch.optim.SGD, lr=0.01)
        eager_opt.zero_grad()
        expected = [eager(sample) for sample in samples]
        torch.stack(expected).sum().backward()
        torch.testing.assert_close(resumed.train_step(samples), expected)
        for name, meta in resumed.fullmap.items():
            if meta.is_param:
                torch.testing.assert_close(getattr(resumed, name).grad,
                                           eager_params[meta.orig_name].grad[meta.slicers])
        resumed_opt.step()
        eager_opt.step()
        for name, meta in resumed.fullmap.items():
            if meta.is_param:
                torch.testing.assert_close(getattr(resumed, name), eager_params[meta.orig_name][meta.slicers])
        for name, value in frozen.items():
            torch.testing.assert_close(getattr(resumed, name), value, rtol=0, atol=0)
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('scheduler', ['1f1b', '1f1b_interleaved'])
def test_pipeline_frozen_stream_matches_eager(scheduler):
    assert all(launch_torchrun(2, worker, scheduler).values())
