# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage a generated package, optionally omitting resume-unused initial weights.

Resume-only staging requires an NNScaler runtime that explicitly accepts
missing initial weights when constructing with init_params=False. Keep the
canonical complete package: the staged copy cannot start training from scratch.
"""
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.module import ParallelModule


_INITIAL_WEIGHT_SHARD = re.compile(
    re.escape(FxModuleParser.ATTR_CONTENT_FILE_STEM) + r'\.[0-9]+$'
)


def stage_gencode(source, destination, *, resume_only: bool = False, dry_run: bool = False) -> dict:
    """Copy a generated package to a new directory without changing its source.

    ``resume_only=True`` omits numeric initial-weight shards in generated module
    directories. The caller must construct those modules with init_params=False
    and restore weights from a checkpoint before use. Checkpoint selection and
    validation belong to the training application; this API never opens one.

    Code, metadata, non-persistent buffers and all other files are preserved;
    Python bytecode caches are excluded. Symlinks inside the package and an
    existing destination are rejected. Copy failures remove the temporary copy.

    These are lightweight structural checks, not a deserialization or a complete
    compatibility check. NNScaler's reuse/configuration and checkpoint checks
    still run when the application constructs the model.

    With ``dry_run=True``, return the planned file/byte counts without writing.
    Otherwise return those counts and the copy duration after staging succeeds.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination == source or source in destination.parents:
        raise ValueError('destination must be outside the source package')
    if destination.exists():
        raise FileExistsError(destination)
    modules = sorted(source.rglob(ParallelModule.COMPUTE_CONFIG_FILE))
    if not modules:
        raise ValueError('No generated modules found')
    for config in modules:
        parent = config.parent
        if not any(parent.glob('gencode*.py')):
            raise ValueError(f'Missing generated code: {parent}')
        for name in (FxModuleParser.NON_PERSISTENT_BUFFER_FILE,
                     ParallelModule.ORIGIN_MODULE_METADATA_FILE,
                     FxModuleParser.ATTR_CONTENT_INDEX_FILE):
            if not (parent / name).is_file():
                raise ValueError(f'Missing required metadata: {parent / name}')
    module_dirs = {config.parent for config in modules}
    files, skipped = [], []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symbolic links are not supported inside the generated package: {path}')
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if resume_only and path.parent in module_dirs and _INITIAL_WEIGHT_SHARD.fullmatch(path.name):
            skipped.append(path)
        else:
            files.append(path)
    result = {
        'source': str(source), 'destination': str(destination),
        'resume_only': resume_only,
        'copied_files': len(files), 'copied_bytes': sum(p.stat().st_size for p in files),
        'omitted_files': len(skipped), 'omitted_bytes': sum(p.stat().st_size for p in skipped),
        'dry_run': dry_run,
    }
    if dry_run:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    started = time.perf_counter()
    try:
        for src in files:
            dst = temporary / src.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        result['copy_seconds'] = time.perf_counter() - started
        # Do not replace a destination created by another staging process.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return result

