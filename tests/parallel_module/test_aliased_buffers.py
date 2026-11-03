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


@pytest.mark.parametrize('package_format', ['attributes', 'legacy', 'legacy_index'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_generated_buffer_bundle_is_relocatable_and_loads_only_local_shards(tmp_path, package_format, dtype):
    from nnscaler.graph.parser.frame import Frame
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
    if package_format == 'legacy':
        frame.save_np_buffer_content(original / 'npbuffer.pt')
        for path in original.glob('fullmodel.pt.*'):
            path.unlink()  # Old resume packages can contain only the sidecar.
    else:
        if package_format == 'legacy_index':
            torch.save({'__nnscaler_npbuffer_index__': 1,
                        'files': {t.tid: f'fullmodel.pt.{i}' for i, t in enumerate(tensors)}},
                       original / 'npbuffer.pt')
        # Unused shards need not be present when restoring only local buffers.
        (original / 'fullmodel.pt.0').unlink()
        (original / 'fullmodel.pt.2').unlink()
    moved = tmp_path / 'moved'
    original.rename(moved)

    model = CubeModule()
    model.register_buffer('reference', torch.empty(2, 4, dtype=dtype), persistent=False)
    model.add_full_map('reference', tensors[1].tid, False, 'reference.weight1',
                       tuple(full.shape), (slice(2, 4), slice(3, 7)), 2)
    model.weight = torch.nn.Parameter(torch.full((2, 4), 37., dtype=dtype))
    model.add_full_map('weight', tensors[0].tid, True, 'weight',
                       tuple(full.shape), (slice(2, 4), slice(3, 7)), 1)
    model.register_buffer('persistent', torch.full((2, 4), 41., dtype=dtype))
    model.add_full_map('persistent', tensors[2].tid, False, 'persistent',
                       tuple(full.shape), (slice(2, 4), slice(3, 7)), 1)
    with patch('nnscaler.runtime.module.torch.load', wraps=torch.load) as reader:
        if package_format == 'attributes':
            assert not (moved / 'npbuffer.pt').exists()
            model.load_attr_content(str(moved / 'fullmodel.pt'), non_persistent_only=True)
        else:
            model.load_np_buffer_content(str(moved / 'npbuffer.pt'))
    torch.testing.assert_close(model.reference, (full + 1)[2:4, 3:7] / 2)
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 37.))
    torch.testing.assert_close(model.persistent, torch.full_like(model.persistent, 41.))
    paths = [Path(call.args[0]).name for call in reader.call_args_list]
    expected = {'attributes': ['fullmodel.pt.index', 'fullmodel.pt.1'],
                'legacy': ['npbuffer.pt'],
                'legacy_index': ['npbuffer.pt', 'fullmodel.pt.1']}
    assert paths == expected[package_format]
    assert all(call.kwargs['map_location'] == 'cpu' for call in reader.call_args_list
               if not str(call.args[0]).endswith('.index'))
    assert 'reference' not in model.state_dict()


@pytest.mark.parametrize('version,files', [(2, {}), (1, {}), (1, {1: '../outside.pt'})])
def test_legacy_buffer_index_rejects_unknown_version_or_missing_local_source(tmp_path, version, files):
    filename = tmp_path / 'npbuffer.pt'
    torch.save({'__nnscaler_npbuffer_index__': version, 'files': files}, filename)
    with pytest.raises(RuntimeError):
        shard().load_np_buffer_content(str(filename))


def test_buffer_subset_requires_its_tensor_ids_in_fullmodel_index(tmp_path):
    torch.save({0: 0}, tmp_path / 'fullmodel.pt.index')
    with pytest.raises(RuntimeError, match='missing tensor IDs'):
        shard().load_attr_content(str(tmp_path / 'fullmodel.pt'), non_persistent_only=True)


def test_buffer_subset_requires_fullmap_metadata(tmp_path):
    model = shard()
    del model.fullmap['replica']
    with pytest.raises(RuntimeError, match='non-persistent buffers not found in fullmap'):
        model.load_attr_content(str(tmp_path / 'fullmodel.pt'), non_persistent_only=True)


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


def worker(scheduler, package_format):
    nnscaler.init()
    torch.manual_seed(7)
    source = PairedPipeline()
    eager = copy.deepcopy(source).cuda()
    eager_opt = torch.optim.SGD(eager.parameters(), lr=0.01, momentum=0.9)
    cfg = nnscaler.ComputeConfig(2, 2, use_end2end=True, constant_folding=False,
                                pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                            'pipeline_scheduler': scheduler})
    directory = Path(tempfile.gettempdir()) / f'paired_ref_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = nnscaler.parallelize(source, {'x': torch.ones(8, 8)}, policy, cfg,
                                    gen_savedir=tempdir, reuse='override').cuda()
        assert not list(Path(tempdir).rglob('npbuffer.pt'))
        aliases = {m.orig_name: m.orig_name.replace('.reference.', '.')
                   for m in model.fullmap.values() if '.reference.' in m.orig_name}
        model.init_aliased_buffers(aliases)
        frozen = {name: value.clone() for name, value in model.get_non_persistent_buffers().items()}
        opt = nnscaler.build_optimizer(model, torch.optim.SGD, lr=0.01, momentum=0.9)
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
        # Save a real per-rank checkpoint: parameters and optimizer history,
        # but no frozen reference buffers. Restore those from the generated package.
        checkpoint_file = Path(tempdir) / f'checkpoint{torch.distributed.get_rank()}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint_file)
        if package_format == 'legacy':
            local_names = {meta.tid: meta.orig_name for name, meta in model.fullmap.items()
                           if name in model.get_non_persistent_buffers()}
            names_by_rank = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(names_by_rank, local_names)
            if torch.distributed.get_rank() == 0:
                package = next(Path(tempdir).rglob('fullmodel.pt.index')).parent
                source_buffers = dict(source.named_buffers())
                values = {tid: source_buffers[name].cpu() for names in names_by_rank for tid, name in names.items()}
                torch.save(values, package / 'npbuffer.pt')
                for path in package.glob('fullmodel.pt.*'):
                    path.unlink()
            torch.distributed.barrier()
        resumed = type(model)(init_params=False).cuda()
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        resumed.load_state_dict(checkpoint['model'])
        resumed_opt = nnscaler.build_optimizer(resumed, torch.optim.SGD, lr=0.01, momentum=0.9)
        resumed_opt.load_state_dict(checkpoint['optimizer'])
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
@pytest.mark.parametrize('package_format', ['attributes', 'legacy'])
def test_pipeline_frozen_stream_matches_eager(scheduler, package_format):
    assert all(launch_torchrun(2, worker, scheduler, package_format).values())
