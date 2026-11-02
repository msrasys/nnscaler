"""Audit deterministic trajectories through the unchanged production VL entrypoint."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import subprocess
import shutil
import sys
import time

from vl_resume_observer_v2 import install_vision_determinism


def generated_attention_audit(root):
    records = []
    for path in sorted(root.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
            if name not in ('wrap_ring_attn_varlen_func', 'wrap_sliding_window_attn_func'):
                continue
            value = next((kw.value for kw in node.keywords if kw.arg == 'deterministic'), None)
            if not isinstance(value, ast.Constant) or value.value is not True:
                raise RuntimeError(f'Attention is not deterministic: {path}:{node.lineno}')
            records.append({'path': str(path.relative_to(root)), 'line': node.lineno, 'function': name})
    if not records or not any('VisionBackbone' in row['path'] for row in records):
        raise RuntimeError('Missing generated language/Vision attention evidence')
    if not any('Language' in row['path'] for row in records):
        raise RuntimeError('Missing generated language attention evidence')
    return records


def compare_metrics(reference, actual, tolerance=1e-5):
    differences = {}
    for key in ('gnorm', 'train_loss', 'lr', 'moe_loss', 'mtp_loss'):
        left, right = float(reference[key]), float(actual[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f'Nonfinite {key}')
        differences[key] = abs(left - right)
    failures = {key: value for key, value in differences.items()
                if value > (0.0 if key == 'lr' else tolerance)}
    return differences, failures


def main():
    llm_root = Path(os.environ['VL_BENCH_LLM_ROOT']).resolve()
    nnscaler_root = Path(os.environ['NNSCALER_ROOT']).resolve()
    output = Path(os.environ['VL_BENCH_AUDIT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    horizon = int(os.environ['MAX_UPDATES'])
    sys.path.insert(0, str(llm_root / 'llm'))
    sys.path.insert(0, str(nnscaler_root))
    import numpy as np
    import torch
    import nnscaler
    from nnscaler.cli.trainer import Trainer
    from data.vl_data_loader import VLDataLoader
    from arch.vision_encoder import MoonViT3dVisionTower

    stream = (output / f'rank{rank}.jsonl').open('x', buffering=1)

    def record(kind, **values):
        row = {'kind': kind, 'rank': rank, 'pid': os.getpid(), 'time_ns': time.time_ns(), **values}
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        return row

    expected_sha = os.environ['VL_BENCH_EXPECTED_SHA']
    actual_sha = subprocess.check_output(['git', '-C', str(nnscaler_root), 'rev-parse', 'HEAD'], text=True).strip()
    if not Path(nnscaler.__file__).resolve().is_relative_to(nnscaler_root) or actual_sha != expected_sha:
        raise RuntimeError('Unexpected nnScaler source')
    llm_sha = subprocess.check_output(['git', '-C', str(llm_root), 'rev-parse', 'HEAD'], text=True).strip()
    if llm_sha != os.environ['VL_BENCH_EXPECTED_LLM_SHA']:
        raise RuntimeError('Unexpected llm-train source')

    reference_path = os.environ.get('VL_TRAJECTORY_REFERENCE')
    reference = []
    if reference_path:
        reference = [json.loads(line) for line in
                     (Path(reference_path) / f'rank{rank}.jsonl').read_text().splitlines()]

    def reference_record(kind, **filters):
        rows = [row for row in reference if row['kind'] == kind
                and all(row.get(key) == value for key, value in filters.items())]
        if len(rows) != 1:
            raise RuntimeError(f'Expected exactly one reference {kind}: {filters}, got {len(rows)}')
        return rows[0]

    def tensor_digest(value):
        value = value.detach()
        if not value.is_contiguous():
            value = value.contiguous()
        flat = value.reshape(-1)
        h = hashlib.sha256()
        # Bound transient host staging at 16 MiB per rank; never clone a full model.
        chunk = max(1, 16 * 1024 * 1024 // max(1, value.element_size()))
        for start in range(0, flat.numel(), chunk):
            cpu = flat[start:start + chunk].cpu().contiguous()
            h.update(memoryview(cpu.view(torch.uint8).numpy()).cast('B'))
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': h.hexdigest()}

    def digest(value):
        def normalized(obj):
            if hasattr(obj, 'tensor_signature'):
                signature = obj.tensor_signature()
                return dict(signature, shape=list(signature['shape']))
            if torch.is_tensor(obj):
                return tensor_digest(obj)
            if isinstance(obj, np.ndarray):
                return tensor_digest(torch.from_numpy(np.ascontiguousarray(obj)))
            if isinstance(obj, dict):
                return {str(k): normalized(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
            if isinstance(obj, (list, tuple)):
                return [normalized(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return repr(obj)
        return hashlib.sha256(json.dumps(normalized(value), sort_keys=True).encode()).hexdigest()

    def model_snapshot(trainer):
        result = {}
        for module_name, module in trainer.model.named_modules():
            metadata = getattr(module, 'fullmap', {})
            for name, value in module.named_parameters(recurse=False):
                meta = metadata.get(name)
                identity = {'module': module_name, 'name': name}
                if meta is not None:
                    identity.update(name=meta.orig_name, slices=repr(meta.slicers),
                                    full_shape=list(meta.shape), value_chunks=meta.val_chunks)
                key = json.dumps(identity, sort_keys=True)
                if key in result:
                    raise RuntimeError(f'Duplicate model snapshot identity: {key}')
                result[key] = tensor_digest(value)
        if not result:
            raise RuntimeError('No model parameters audited')
        return result

    def optimizer_snapshot(optimizer, gradients=False):
        result = {}
        children = getattr(optimizer, 'optimizers', None)
        if children is not None:
            for child_index, child in enumerate(children):
                result[str(child_index)] = optimizer_snapshot(child, gradients)
            return result
        for group_index, group in enumerate(optimizer.param_groups):
            for param_index, parameter in enumerate(group['params']):
                key = f'{group_index}:{param_index}'
                if gradients:
                    result[key] = None if parameter.grad is None else tensor_digest(parameter.grad)
                else:
                    state = optimizer.state.get(parameter, {})
                    result[key] = {str(k): tensor_digest(v) if torch.is_tensor(v) else v
                                   for k, v in state.items()}
        return result

    def snapshot(trainer, stage, step):
        if stage == 'preclip':
            state = optimizer_snapshot(trainer.optimizer, gradients=True)
        else:
            state = {'model': model_snapshot(trainer),
                     'optimizer': optimizer_snapshot(trainer.optimizer)}
        payload = {'stage': stage, 'step': step, 'state': state}
        record('snapshot', **payload)
        if reference:
            old = reference_record('snapshot', stage=stage, step=step)
            exact = old['state'] == state
            record('snapshot_comparison', stage=stage, step=step, exact=exact)
            if stage == 'initial' and not exact:
                raise RuntimeError('Initial model/optimizer state differs from baseline')

    install_vision_determinism(MoonViT3dVisionTower, record)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    original_setup, original_train = Trainer._setup, Trainer._train
    original_metrics, original_next = Trainer.log_metrics, VLDataLoader.__next__
    batch_index = 0

    def observed_setup(trainer, *args, **kwargs):
        value = original_setup(trainer, *args, **kwargs)
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError('Deterministic algorithms were disabled during setup')
        config = trainer.train_args.compute_config
        record('setup', horizon=trainer.max_train_steps,
               deterministic=True, tf32=torch.backends.cuda.matmul.allow_tf32,
               compute_config={key: getattr(config, key, None) for key in
                  ('plan_ngpus', 'runtime_ngpus', 'use_zero', 'use_async_reducer',
                   'use_fbw', 'zero_param_level_sharding')})
        if rank == 0:
            records = generated_attention_audit(Path(os.environ['GEN_SAVEDIR']))
            record('generated_attention', calls=records, verified_count=len(records))
        torch.distributed.barrier()
        if rank == 0:
            removed = 0
            for path in Path(os.environ['GEN_SAVEDIR']).rglob('*'):
                if path.is_file() and (path.name.startswith('fullmodel.pt.') or path.name == 'npbuffer.pt'):
                    removed += path.stat().st_size
                    path.unlink()
            record('loaded_attribute_files_released', bytes=removed)
        torch.distributed.barrier()
        return value

    def observed_train(trainer, *args, **kwargs):
        snapshot(trainer, 'initial', 0)
        reference_buffers = {f'{mn}:{name}': tensor_digest(value) for mn, mod in trainer.model.named_modules() for name, value in mod.named_buffers(recurse=False) if 'reference' in getattr(getattr(mod, 'fullmap', {}).get(name), 'orig_name', '')}
        record('reference_initial', state=reference_buffers)
        torch.cuda.reset_peak_memory_stats()
        rng = digest({'python': random.getstate(), 'numpy': np.random.get_state(),
                      'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()})
        record('initial_rng', sha256=rng)
        if reference and reference_record('initial_rng')['sha256'] != rng:
            raise RuntimeError('Initial RNG state differs')
        original_clip = trainer.optimizer.clip_gnorm

        def audited_clip(*clip_args, **clip_kwargs):
            step = int(trainer.train_status.finished_train_steps) + 1
            return original_clip(*clip_args, **clip_kwargs)

        trainer.optimizer.clip_gnorm = audited_clip
        return original_train(trainer, *args, **kwargs)

    def observed_next(loader):
        nonlocal batch_index
        batch = original_next(loader)
        fields = ('tokens', 'targets', 'loss_mask', 'moe_loss_mask', 'context',
                  'image_input_mask', 'image_grid_thws', 'image_pixel_values')
        fields += tuple(key for key in batch if key.startswith('mtp_'))
        fields += ('loss_index', 'tag_index')
        audited = {key: batch[key] for key in fields if key in batch}
        h = digest(audited)
        grids = batch.get('image_grid_thws')
        record('batch', index=batch_index, sha256=h, field_hashes={key: digest(value) for key, value in audited.items()},
               images=0 if grids is None else int(grids.shape[0]))
        if reference and reference_record('batch', index=batch_index)['sha256'] != h:
            raise RuntimeError(f'Input batch differs at index {batch_index}')
        batch_index += 1
        return batch

    def observed_metrics(trainer, metrics, *args, **kwargs):
        if 'cuda_gb_allocated' in metrics:
            record('allocator_metrics', step=int(trainer.train_status.finished_train_steps), tag=kwargs.get('tag'), values=metrics)
        if kwargs.get('tag') == 'train' and 'train_loss' in metrics:
            step = int(trainer.train_status.finished_train_steps)
            values = {key: float(metrics[key]) for key in
                      ('train_loss', 'gnorm', 'lr', 'bsz', 'train_wall', 'FLOPS_VISION', 'moe_loss', 'mtp_loss') if key in metrics}
            record('metrics', step=step, values=values)
            if reference:
                differences, failures = compare_metrics(reference_record('metrics', step=step)['values'], values)
                record('metric_comparison', step=step, abs_differences=differences, failures=failures)
                if failures:
                    raise RuntimeError(f'Trajectory tolerance exceeded at step {step}: {failures}')
            if step in (1, horizon):
                snapshot(trainer, 'poststep', step)
                buffers = {f'{mn}:{name}': tensor_digest(value) for mn, mod in trainer.model.named_modules() for name, value in mod.named_buffers(recurse=False) if 'reference' in getattr(getattr(mod, 'fullmap', {}).get(name), 'orig_name', '')}
                record('reference_poststep', step=step, state=buffers)
        if kwargs.get('tag') == 'train' and 'train_loss' in metrics:
            record('memory', step=int(trainer.train_status.finished_train_steps), allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        return original_metrics(trainer, metrics, *args, **kwargs)

    Trainer._setup, Trainer._train = observed_setup, observed_train
    Trainer.log_metrics, VLDataLoader.__next__ = observed_metrics, observed_next
    record('start', nnscaler_sha=actual_sha, llm_sha=llm_sha, horizon=horizon,
           deterministic_env=os.environ.get('DETERMINISTIC'), torch_version=torch.__version__,
           cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), train_arguments=sys.argv[1:])
    entry = llm_root / 'llm/nnscaler_train_vl.py'
    sys.argv[0] = str(entry)
    try:
        runpy.run_path(str(entry), run_name='__main__')
        record('complete', horizon=horizon)
    except BaseException as exc:
        record('exception', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        stream.close()


if __name__ == '__main__':
    main()
