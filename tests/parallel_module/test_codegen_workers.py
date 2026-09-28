#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pickle
from pathlib import Path
import sys
import time

import pytest
import torch

import nnscaler
from nnscaler.codegen.serialization import dump_codegen_payload
from nnscaler.codegen.metadata import compact_worker_metadata, worker_metadata_filename
from nnscaler.parallel import (
    ComputeConfig,
    _compact_attr_meta_files,
    _gencode_in_subprocesses,
    _partition_codegen_ranks,
    parallelize,
)
from nnscaler.runtime.module import ParallelModule

from ..utils import replace_all_device_with


class _RegularModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x).relu()


class _End2EndModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, data):
        return self.linear(data).sum()


def _emit_custom_neg(node, args, kwargs, runtime_devid, plan_ndevs, runtime_ndevs):
    return f'torch.neg({args[0]})'


@nnscaler.register_op('* -> *', emit_fn=_emit_custom_neg)
def _custom_neg(x):
    return -x


class _CustomEmitModel(torch.nn.Module):
    def forward(self, x):
        return _custom_neg(x)


def _make_local_emitter():
    operation = 'torch.neg'

    def emit(node, args, kwargs, runtime_devid, plan_ndevs, runtime_ndevs):
        return f'{operation}({args[0]})'

    return emit


@nnscaler.register_op('* -> *', emit_fn=_make_local_emitter())
def _custom_local_emit_neg(x):
    return -x


class _LocalEmitModel(torch.nn.Module):
    def forward(self, x):
        return _custom_local_emit_neg(x)


def _generated_module_dir(root: Path) -> Path:
    matches = list(root.rglob('gencode0.py'))
    assert len(matches) == 1
    return matches[0].parent


def _generate(
    root: Path,
    model: torch.nn.Module,
    *,
    codegen_workers: int,
    use_end2end: bool,
    reuse: str = 'match',
) -> Path:
    parallelize(
        model,
        {'data' if use_end2end else 'x': torch.randn(2, 4)},
        'data' if use_end2end else 'dp',
        ComputeConfig(1, 2, use_end2end=use_end2end),
        gen_savedir=root,
        reuse=reuse,
        load_module=False,
        codegen_workers=codegen_workers,
    )
    return _generated_module_dir(root)


def _load_compact_raw_maps(module_dir: Path) -> tuple[dict, list[dict]]:
    with (module_dir / ParallelModule.ATTR_META_FILE).open('rb') as stream:
        compact_meta = pickle.load(stream)
    raw_maps = [
        pickle.loads(compact_meta['unique_payloads'][variant])
        for variant in compact_meta['rank_to_variant']
    ]
    return compact_meta, raw_maps


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize(
    ('model_factory', 'use_end2end'),
    [
        (_RegularModel, False),
        (_End2EndModel, True),
        (_CustomEmitModel, False),
        (_LocalEmitModel, False),
    ],
)
def test_multi_process_codegen_matches_serial(tmp_path, model_factory, use_end2end, monkeypatch):
    serializers = []
    bundles = []

    def record_serializer(payload, stream):
        serializer = dump_codegen_payload(payload, stream)
        serializers.append(serializer)
        return serializer

    monkeypatch.setattr('nnscaler.parallel.dump_codegen_payload', record_serializer)

    def inspect_bundles(staging_dir, rank_ranges, runtime_ngpus):
        assert not list(staging_dir.glob('attr_meta[0-9]*.pkl'))
        paths = list(staging_dir.glob('attr_meta_worker_*.pkl'))
        assert len(paths) == len(rank_ranges)
        bundles.extend(paths)
        return compact_worker_metadata(staging_dir, rank_ranges, runtime_ngpus)

    monkeypatch.setattr('nnscaler.parallel.compact_worker_metadata', inspect_bundles)
    serial_dir = _generate(
        tmp_path / 'serial',
        model_factory(),
        codegen_workers=1,
        use_end2end=use_end2end,
    )
    parallel_dir = _generate(
        tmp_path / 'parallel',
        model_factory(),
        codegen_workers=2,
        use_end2end=use_end2end,
    )
    assert serializers == ['cloudpickle']
    assert len(bundles) == 2

    for rank in range(2):
        assert (serial_dir / f'gencode{rank}.py').read_bytes() == (
            parallel_dir / f'gencode{rank}.py'
        ).read_bytes()

    serial_compact_meta, serial_attr_meta = _load_compact_raw_maps(serial_dir)
    parallel_compact_meta, parallel_attr_meta = _load_compact_raw_maps(parallel_dir)
    assert serial_attr_meta == parallel_attr_meta
    assert serial_compact_meta['version'] == ParallelModule.ATTR_META_FORMAT_VERSION
    assert parallel_compact_meta['version'] == ParallelModule.ATTR_META_FORMAT_VERSION
    assert not list(serial_dir.glob('attr_meta[0-9]*.pkl'))
    assert not list(parallel_dir.glob('attr_meta[0-9]*.pkl'))
    assert not list(parallel_dir.glob('attr_meta_worker_*.pkl'))
    assert (serial_dir / ParallelModule.ATTR_META_FILE).read_bytes() == (parallel_dir / ParallelModule.ATTR_META_FILE).read_bytes()

    if model_factory is _End2EndModel:
        assert '_train_step' in (parallel_dir / 'gencode0.py').read_text()
    if model_factory in (_CustomEmitModel, _LocalEmitModel):
        assert 'torch.neg' in (parallel_dir / 'gencode0.py').read_text()


@replace_all_device_with('cpu', force=True)
def test_codegen_workers_reuse_cached_custom_op_graph(tmp_path, monkeypatch):
    module_dir = _generate(tmp_path, _LocalEmitModel(), codegen_workers=1, use_end2end=False)
    expected_code = [(module_dir / f'gencode{rank}.py').read_bytes() for rank in range(2)]
    expected_meta = _load_compact_raw_maps(module_dir)
    serializers = []

    def record_serializer(payload, stream):
        serializer = dump_codegen_payload(payload, stream)
        serializers.append(serializer)
        return serializer

    monkeypatch.setattr('nnscaler.parallel.dump_codegen_payload', record_serializer)
    _generate(tmp_path, _LocalEmitModel(), codegen_workers=2, use_end2end=False, reuse='graph')
    assert serializers == ['cloudpickle']
    assert [(module_dir / f'gencode{rank}.py').read_bytes() for rank in range(2)] == expected_code
    assert _load_compact_raw_maps(module_dir) == expected_meta


@pytest.mark.parametrize('codegen_workers', [0, -1, True])
def test_codegen_workers_must_be_positive_integer(codegen_workers, tmp_path):
    with pytest.raises(ValueError, match='codegen_workers must be a positive integer'):
        parallelize(
            _RegularModel(),
            {'x': torch.randn(2, 4)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tmp_path,
            load_module=False,
            codegen_workers=codegen_workers,
        )


def test_codegen_rank_ranges_are_contiguous_and_balanced():
    assert _partition_codegen_ranks(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_compact_attr_meta_deduplicates_exact_payloads(tmp_path):
    empty_payload = pickle.dumps({})
    populated_payload = pickle.dumps({
        'weight': {
            'tid': 1,
            'is_param': True,
            'orig_name': 'weight',
            'shape': (4, 4),
            'slicers': (slice(None), slice(None)),
            'val_chunks': 1,
            'dtype': torch.float32,
            'sub_shape': (4, 4),
        },
    })
    for rank, payload in enumerate((empty_payload, empty_payload, populated_payload, populated_payload)):
        (tmp_path / f'attr_meta{rank}.pkl').write_bytes(payload)

    assert _compact_attr_meta_files(tmp_path, 4) == 2
    with (tmp_path / ParallelModule.ATTR_META_FILE).open('rb') as stream:
        compact_meta = pickle.load(stream)
    assert compact_meta['unique_payloads'] == [empty_payload, populated_payload]
    assert compact_meta['rank_to_variant'] == [0, 0, 1, 1]

    loaded_maps = ParallelModule._load_attr_meta_maps(tmp_path, 4)
    assert loaded_maps[0] is loaded_maps[1]
    assert loaded_maps[2] is loaded_maps[3]
    assert loaded_maps[0] == {}
    assert loaded_maps[2]['weight'].orig_name == 'weight'


def test_compact_attr_meta_validates_world_size_and_variant_indexes(tmp_path):
    compact_file = tmp_path / ParallelModule.ATTR_META_FILE
    compact_file.write_bytes(pickle.dumps({
        'version': ParallelModule.ATTR_META_FORMAT_VERSION,
        'unique_payloads': [pickle.dumps({})],
        'rank_to_variant': [0],
    }))
    with pytest.raises(RuntimeError, match='world size mismatch'):
        ParallelModule._load_attr_meta_maps(tmp_path, 2)

    compact_file.write_bytes(pickle.dumps({
        'version': ParallelModule.ATTR_META_FORMAT_VERSION,
        'unique_payloads': [pickle.dumps({})],
        'rank_to_variant': [0, 1],
    }))
    with pytest.raises(RuntimeError, match='rank-to-variant index'):
        ParallelModule._load_attr_meta_maps(tmp_path, 2)


@replace_all_device_with('cpu', force=True)
def test_attr_meta_legacy_loading_and_reuse(tmp_path):
    root = tmp_path / 'legacy'
    module_dir = _generate(root, _RegularModel(), codegen_workers=1, use_end2end=False)

    compact_maps = ParallelModule._load_attr_meta_maps(module_dir, 2)
    compact_meta, _ = _load_compact_raw_maps(module_dir)
    for rank, variant in enumerate(compact_meta['rank_to_variant']):
        (module_dir / f'attr_meta{rank}.pkl').write_bytes(compact_meta['unique_payloads'][variant])
    (module_dir / ParallelModule.ATTR_META_FILE).unlink()

    legacy_maps = ParallelModule._load_attr_meta_maps(module_dir, 2)
    assert legacy_maps == compact_maps
    # A complete legacy shard set remains reusable and is not regenerated.
    assert _generate(root, _RegularModel(), codegen_workers=1, use_end2end=False) == module_dir

    # A successful code-only regeneration upgrades legacy shards to the compact format.
    assert _generate(
        root,
        _RegularModel(),
        codegen_workers=1,
        use_end2end=False,
        reuse='graph',
    ) == module_dir
    assert (module_dir / ParallelModule.ATTR_META_FILE).is_file()
    assert not list(module_dir.glob('attr_meta[0-9]*.pkl'))


class _FailingModuleCodeGen:
    def gen(
            self,
            rank,
            *,
            forward_args,
            outfile,
            attach,
            as_parallel_module,
            end2end_mode,
            outfile_attr_meta_map,
        ):
        if rank == 0:
            raise RuntimeError('intentional worker failure')
        if rank == 2:
            time.sleep(30)
        Path(outfile).write_text(f'rank {rank}\n')
        pickle.dump({'rank': rank}, outfile_attr_meta_map)


class _PayloadDepthModuleCodeGen:
    def gen(
            self,
            rank,
            *,
            forward_args,
            outfile,
            attach,
            as_parallel_module,
            end2end_mode,
            outfile_attr_meta_map,
        ):
        nested = forward_args['nested']
        depth = 0
        while isinstance(nested, list):
            depth += 1
            nested = nested[0]
        Path(outfile).write_text(f'{rank}:{depth}\n')
        pickle.dump({}, outfile_attr_meta_map)


def test_multi_process_codegen_serializes_deep_payload(tmp_path):
    depth = 1500
    nested = None
    for _ in range(depth):
        nested = [nested]

    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        _gencode_in_subprocesses(
            _PayloadDepthModuleCodeGen(),
            None,
            {'nested': nested},
            ComputeConfig(1, 2),
            tmp_path,
            2,
        )
        assert sys.getrecursionlimit() == 1000
    finally:
        sys.setrecursionlimit(original_limit)

    assert (tmp_path / 'gencode0.py').read_text() == f'0:{depth}\n'
    assert (tmp_path / 'gencode1.py').read_text() == f'1:{depth}\n'
    assert (tmp_path / ParallelModule.ATTR_META_FILE).is_file()


def test_worker_failure_terminates_siblings_and_cleans_staging(tmp_path):
    final_code = tmp_path / 'gencode0.py'
    final_attr_meta = tmp_path / ParallelModule.ATTR_META_FILE
    final_code.write_text('old code')
    final_attr_meta.write_bytes(b'old metadata')

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match='intentional worker failure') as exc_info:
        _gencode_in_subprocesses(
            _FailingModuleCodeGen(),
            None,
            {},
            ComputeConfig(1, 4),
            tmp_path,
            2,
        )

    assert time.monotonic() - started_at < 15
    assert 'worker 0 ranks [0, 2)' in str(exc_info.value)
    assert not list(tmp_path.glob('.nnscaler-codegen-*'))
    assert final_code.read_text() == 'old code'
    assert final_attr_meta.read_bytes() == b'old metadata'


@pytest.mark.parametrize('failure', ['missing', 'invalid_range'])
def test_invalid_worker_metadata_preserves_previous_outputs(tmp_path, monkeypatch, failure):
    final_code = tmp_path / 'gencode0.py'
    final_meta = tmp_path / ParallelModule.ATTR_META_FILE
    final_code.write_text('old code')
    final_meta.write_bytes(b'old metadata')
    original = compact_worker_metadata

    def corrupt_and_compact(staging_dir, rank_ranges, runtime_ngpus):
        bundle_file = staging_dir / worker_metadata_filename(*rank_ranges[0])
        if failure == 'missing':
            bundle_file.unlink()
        else:
            bundle = pickle.loads(bundle_file.read_bytes())
            bundle['rank_start'] = 1
            bundle_file.write_bytes(pickle.dumps(bundle))
        return original(staging_dir, rank_ranges, runtime_ngpus)

    monkeypatch.setattr('nnscaler.parallel.compact_worker_metadata', corrupt_and_compact)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        _gencode_in_subprocesses(_PayloadDepthModuleCodeGen(), None, {'nested': None},
                                ComputeConfig(1, 2), tmp_path, 2)
    assert final_code.read_text() == 'old code'
    assert final_meta.read_bytes() == b'old metadata'
    assert not list(tmp_path.glob('.nnscaler-codegen-*'))
