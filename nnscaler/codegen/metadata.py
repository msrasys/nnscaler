#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Temporary worker metadata bundles and deterministic final compaction."""

from collections.abc import Iterable
import os
from pathlib import Path
import pickle
import tempfile

from nnscaler.runtime.module import ParallelModule


_WORKER_BUNDLE_VERSION = 1


def worker_metadata_filename(rank_start: int, rank_end: int) -> str:
    return f'attr_meta_worker_{rank_start}_{rank_end}.pkl'


def _atomic_pickle_dump(value, path: Path) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{path.name}.', dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            pickle.dump(value, stream)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class AttributeMetadata:
    """Keep one copy of each pickle payload, preserving first-rank order."""

    def __init__(self):
        self.unique_payloads: list[bytes] = []
        self.rank_to_variant: list[int] = []
        self._payload_to_variant: dict[bytes, int] = {}

    def add(self, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError('Expected a non-empty serialized rank metadata payload')
        variant = self._payload_to_variant.get(payload)
        if variant is None:
            variant = len(self.unique_payloads)
            self._payload_to_variant[payload] = variant
            self.unique_payloads.append(payload)
        self.rank_to_variant.append(variant)

    def write_worker_bundle(self, outdir: Path, rank_start: int, rank_end: int) -> None:
        if rank_start < 0 or rank_end <= rank_start or len(self.rank_to_variant) != rank_end - rank_start:
            raise ValueError(f'Incomplete worker metadata for ranks [{rank_start}, {rank_end})')
        bundle = {
            'version': _WORKER_BUNDLE_VERSION,
            'rank_start': rank_start,
            'rank_end': rank_end,
            'unique_payloads': self.unique_payloads,
            'rank_to_variant': self.rank_to_variant,
        }
        _atomic_pickle_dump(bundle, outdir / worker_metadata_filename(rank_start, rank_end))


def compact_metadata_payloads(staging_dir: Path, payloads: Iterable[bytes]) -> int:
    metadata = AttributeMetadata()
    for payload in payloads:
        metadata.add(payload)
    # Keep the existing public format and dictionary order byte-for-byte.
    compact = {
        'version': ParallelModule.ATTR_META_FORMAT_VERSION,
        'unique_payloads': metadata.unique_payloads,
        'rank_to_variant': metadata.rank_to_variant,
    }
    _atomic_pickle_dump(compact, staging_dir / ParallelModule.ATTR_META_FILE)
    return len(metadata.unique_payloads)


def _worker_payloads(staging_dir: Path, rank_ranges: list[tuple[int, int]], runtime_ngpus: int):
    next_rank = 0
    for rank_start, rank_end in sorted(rank_ranges):
        if rank_start != next_rank or rank_end <= rank_start or rank_end > runtime_ngpus:
            raise RuntimeError(f'Invalid worker metadata rank coverage: expected rank {next_rank}, got [{rank_start}, {rank_end})')
        path = staging_dir / worker_metadata_filename(rank_start, rank_end)
        with path.open('rb') as stream:
            bundle = pickle.load(stream)
        if not isinstance(bundle, dict) or bundle.get('version') != _WORKER_BUNDLE_VERSION:
            raise RuntimeError(f'Invalid worker metadata bundle version in {path}')
        if (type(bundle.get('rank_start')) is not int or type(bundle.get('rank_end')) is not int
                or (bundle['rank_start'], bundle['rank_end']) != (rank_start, rank_end)):
            raise RuntimeError(f'Worker metadata rank range mismatch in {path}')
        payloads = bundle.get('unique_payloads')
        variants = bundle.get('rank_to_variant')
        if not isinstance(payloads, list) or not all(isinstance(p, bytes) and p for p in payloads):
            raise RuntimeError(f'Invalid worker metadata payloads in {path}')
        if not isinstance(variants, list) or len(variants) != rank_end - rank_start:
            raise RuntimeError(f'Incomplete worker metadata rank mapping in {path}')
        if any(type(v) is not int or v < 0 or v >= len(payloads) for v in variants):
            raise RuntimeError(f'Invalid worker metadata variant index in {path}')
        for variant in variants:
            yield payloads[variant]
        next_rank = rank_end
    if next_rank != runtime_ngpus:
        raise RuntimeError(f'Incomplete worker metadata coverage: expected {runtime_ngpus} ranks, got {next_rank}')


def compact_worker_metadata(staging_dir: Path, rank_ranges: list[tuple[int, int]], runtime_ngpus: int) -> int:
    """Validate every bundle before publishing the existing compact format."""
    return compact_metadata_payloads(staging_dir, _worker_payloads(staging_dir, rank_ranges, runtime_ngpus))
