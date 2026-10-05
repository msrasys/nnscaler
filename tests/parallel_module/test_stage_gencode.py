"""Generated artifact semantics belong to NNScaler, without application imports."""
from pathlib import Path

import pytest
import torch

import nnscaler
from nnscaler.codegen import artifacts
from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.module import ParallelModule
from nnscaler.parallel import _prepare_and_check_reusable
from tests.parallel_module.test_resume_gencode_files import make_package


def package(tmp_path):
    source = tmp_path / 'source'
    leaf = source / '_parallel_modules/model'
    leaf.mkdir(parents=True)
    for name in [ParallelModule.COMPUTE_CONFIG_FILE, 'gencode0.py',
                 FxModuleParser.NON_PERSISTENT_BUFFER_FILE,
                 ParallelModule.ORIGIN_MODULE_METADATA_FILE,
                 FxModuleParser.ATTR_CONTENT_INDEX_FILE,
                 'fullmodel.pt.0', 'fullmodel.pt.1']:
        (leaf / name).write_bytes(name.encode())
    return source, leaf


def contents(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_resume_staging_preserves_buffers_metadata_and_source(tmp_path):
    source, leaf = package(tmp_path)
    before = contents(source)
    destination = tmp_path / 'stage'
    result = nnscaler.stage_gencode(source, destination, resume_only=True)
    expected = {name: value for name, value in before.items()
                if Path(name).name not in ['fullmodel.pt.0', 'fullmodel.pt.1']}
    assert contents(destination) == expected
    assert contents(source) == before
    assert result['omitted_files'] == 2
    assert result['copied_bytes'] == sum(map(len, expected.values()))
    assert result['resume_only'] is True
    assert 'checkpoint' not in result  # job/checkpoint selection is not a library responsibility


def test_full_and_dry_staging(tmp_path):
    source, _ = package(tmp_path)
    destination = tmp_path / 'stage'
    result = nnscaler.stage_gencode(source, destination, dry_run=True)
    assert result['omitted_files'] == 0
    assert not destination.exists()
    nnscaler.stage_gencode(source, destination)
    assert contents(destination) == contents(source)
    with pytest.raises(FileExistsError):
        nnscaler.stage_gencode(source, destination)


@pytest.mark.parametrize('missing', [
    ParallelModule.COMPUTE_CONFIG_FILE, 'gencode0.py',
    FxModuleParser.NON_PERSISTENT_BUFFER_FILE,
    ParallelModule.ORIGIN_MODULE_METADATA_FILE, FxModuleParser.ATTR_CONTENT_INDEX_FILE,
])
def test_invalid_package_rejected_before_copy(tmp_path, missing):
    source, leaf = package(tmp_path)
    (leaf / missing).unlink()
    with pytest.raises(ValueError):
        nnscaler.stage_gencode(source, tmp_path / 'stage', resume_only=True)
    assert not (tmp_path / 'stage').exists()


def test_skip_only_known_weight_shards_and_bytecode(tmp_path):
    source, leaf = package(tmp_path)
    (leaf / 'fullmodel.pt.notes').write_bytes(b'keep')
    (source / 'fullmodel.pt.0').write_bytes(b'not a generated module weight')
    (leaf / '__pycache__').mkdir()
    (leaf / '__pycache__/gencode0.pyc').touch()
    destination = tmp_path / 'stage'
    nnscaler.stage_gencode(source, destination, resume_only=True)
    assert (destination / 'fullmodel.pt.0').read_bytes() == b'not a generated module weight'
    staged_leaf = destination / leaf.relative_to(source)
    assert (staged_leaf / 'fullmodel.pt.notes').read_bytes() == b'keep'
    assert not (staged_leaf / 'fullmodel.pt.0').exists()
    assert not (staged_leaf / '__pycache__').exists()


def test_symlinks_and_destination_inside_source_rejected(tmp_path):
    source, leaf = package(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        nnscaler.stage_gencode(source, leaf / 'copy')
    (leaf / 'link').symlink_to(leaf / 'npbuffer.pt')
    with pytest.raises(ValueError, match='Symbolic links'):
        nnscaler.stage_gencode(source, tmp_path / 'stage')
    assert not (tmp_path / 'stage').exists()


def test_failed_copy_cleans_only_temporary_destination(monkeypatch, tmp_path):
    source, _ = package(tmp_path)
    before = contents(source)
    destination = tmp_path / 'stage'
    def fail(*args):
        raise OSError('simulated full disk')
    monkeypatch.setattr(artifacts.shutil, 'copy2', fail)
    with pytest.raises(OSError, match='full disk'):
        nnscaler.stage_gencode(source, destination)
    assert contents(source) == before
    assert not destination.exists()
    assert not list(tmp_path.glob('.stage-*'))


def test_staged_package_is_accepted_only_by_the_matching_runtime_mode(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'stage'
    config, _ = make_package(source)
    nnscaler.stage_gencode(source, destination, resume_only=True)
    assert _prepare_and_check_reusable(
        str(destination), torch.nn.Linear, config, allow_missing_init_weights=True,
    )[1]
    with pytest.raises(RuntimeError, match='do not match'):
        _prepare_and_check_reusable(str(destination), torch.nn.Linear, config)
    # Staging neither consumes nor strips the original complete package.
    assert _prepare_and_check_reusable(str(source), torch.nn.Linear, config)[1]
