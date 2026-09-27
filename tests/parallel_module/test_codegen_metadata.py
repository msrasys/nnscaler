#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pickle

import pytest
import torch

from nnscaler.codegen.metadata import AttributeMetadata, compact_worker_metadata, worker_metadata_filename
from nnscaler.parallel import _compact_attr_meta_files
from nnscaler.runtime.module import ParallelModule


def _payloads():
    return [pickle.dumps(value) for value in (
        {},
        {'weight': {'tid': 1, 'is_param': True, 'orig_name': 'weight',
                    'shape': (4, 4), 'slicers': (slice(None), slice(None)),
                    'val_chunks': 1, 'dtype': torch.float32, 'sub_shape': (4, 4)}},
    )]


def test_worker_bundles_match_per_rank_format_byte_for_byte(tmp_path):
    legacy = tmp_path / 'legacy'
    bundled = tmp_path / 'bundled'
    legacy.mkdir()
    bundled.mkdir()
    empty, weight = _payloads()
    payloads = [empty, weight, empty, weight, weight, empty, weight]
    ranges = [(0, 3), (3, 5), (5, 7)]
    for rank, payload in enumerate(payloads):
        (legacy / f'attr_meta{rank}.pkl').write_bytes(payload)
    # Simulate workers completing out of rank order, including uneven ranges.
    for start, end in reversed(ranges):
        metadata = AttributeMetadata()
        for payload in payloads[start:end]:
            metadata.add(payload)
        metadata.write_worker_bundle(bundled, start, end)
    assert len(list(bundled.iterdir())) == 3
    assert _compact_attr_meta_files(legacy, len(payloads)) == 2
    assert compact_worker_metadata(bundled, list(reversed(ranges)), len(payloads)) == 2
    assert (legacy / ParallelModule.ATTR_META_FILE).read_bytes() == (bundled / ParallelModule.ATTR_META_FILE).read_bytes()
    assert ParallelModule._load_attr_meta_maps(legacy, 7) == ParallelModule._load_attr_meta_maps(bundled, 7)


@pytest.mark.parametrize(('key', 'value'), [
    ('version', 999),
    ('rank_start', 1),
    ('rank_end', 3),
    ('unique_payloads', ['not bytes']),
    ('unique_payloads', [b'']),
    ('rank_to_variant', [0]),
    ('rank_to_variant', [0, 1]),
    ('rank_to_variant', [0, -1]),
    ('rank_to_variant', [0, True]),
])
def test_invalid_bundle_does_not_publish_compact_metadata(tmp_path, key, value):
    metadata = AttributeMetadata()
    for _ in range(2):
        metadata.add(pickle.dumps({}))
    metadata.write_worker_bundle(tmp_path, 0, 2)
    path = tmp_path / worker_metadata_filename(0, 2)
    bundle = pickle.loads(path.read_bytes())
    bundle[key] = value
    path.write_bytes(pickle.dumps(bundle))
    final = tmp_path / ParallelModule.ATTR_META_FILE
    final.write_bytes(b'previous metadata')
    with pytest.raises(RuntimeError):
        compact_worker_metadata(tmp_path, [(0, 2)], 2)
    assert final.read_bytes() == b'previous metadata'


@pytest.mark.parametrize('ranges', [[], [(1, 2)], [(0, 1)], [(0, 2), (0, 2)], [(0, 3)]])
def test_incomplete_or_overlapping_rank_coverage_is_rejected(tmp_path, ranges):
    for start, end in [(0, 1), (0, 2)]:
        metadata = AttributeMetadata()
        for _ in range(end-start):
            metadata.add(pickle.dumps({}))
        metadata.write_worker_bundle(tmp_path, start, end)
    with pytest.raises(RuntimeError):
        compact_worker_metadata(tmp_path, ranges, 2)
    assert not (tmp_path / ParallelModule.ATTR_META_FILE).exists()


def test_incomplete_worker_cannot_write_success_bundle(tmp_path):
    metadata = AttributeMetadata()
    metadata.add(pickle.dumps({}))
    with pytest.raises(ValueError, match='Incomplete worker metadata'):
        metadata.write_worker_bundle(tmp_path, 0, 2)
    assert not list(tmp_path.iterdir())


def test_bundle_write_failure_removes_temporary_file(tmp_path, monkeypatch):
    metadata = AttributeMetadata()
    metadata.add(pickle.dumps({}))

    def fail_dump(value, stream):
        stream.write(b'partial')
        raise OSError('disk full')

    monkeypatch.setattr(pickle, 'dump', fail_dump)
    with pytest.raises(OSError, match='disk full'):
        metadata.write_worker_bundle(tmp_path, 0, 1)
    assert not list(tmp_path.iterdir())
