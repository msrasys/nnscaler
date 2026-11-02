"""Observe the real VL entrypoint without changing its training or data logic."""
import argparse
import ast
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time


def install_vision_determinism(tower_class, record):
    """Select the existing constructor option without replacing the class."""
    original_init = tower_class.__init__
    signature = inspect.signature(original_init)
    if 'use_deterministic_attn' not in signature.parameters:
        raise RuntimeError('Original Vision constructor has no deterministic configuration option')

    @functools.wraps(original_init)
    def configured_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments['use_deterministic_attn'] = True
        original_init(*bound.args, **bound.kwargs)
        layers = [module for module in self.modules()
                  if type(module).__name__ == 'MoonViTEncoderLayer']
        if not layers or not all(layer.use_deterministic_attn is True for layer in layers):
            raise RuntimeError('Vision constructor did not apply deterministic attention to every layer')
        record('vision_deterministic_configuration', class_name=tower_class.__qualname__,
               class_module=tower_class.__module__, original_constructor_file=inspect.getsourcefile(original_init),
               original_class_preserved=type(self) is tower_class, layers=len(layers),
               use_deterministic_attn=True)

    tower_class.__init__ = configured_init


def verify_generated_vision_attention(gencode, audit):
    """Retain and inspect generated Vision code before heavy artifacts are removed."""
    records = []
    for source in sorted(gencode.rglob('*.py')):
        if 'VisionBackbone' not in source.parts:
            continue
        value = source.read_text()
        calls = []
        for node in ast.walk(ast.parse(value)):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            if name != 'wrap_ring_attn_varlen_func':
                continue
            keyword = next((item.value for item in node.keywords if item.arg == 'deterministic'), None)
            if not isinstance(keyword, ast.Constant) or keyword.value is not True:
                raise RuntimeError(f'Generated Vision attention is not explicitly deterministic: {source}:{node.lineno}')
            calls.append({'line': node.lineno, 'deterministic': True})
        if calls:
            relative = source.relative_to(gencode)
            target = audit / 'generated-vision-code' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
            records.append({'path': str(relative), 'sha256': hashlib.sha256(value.encode()).hexdigest(),
                            'calls': calls, 'retained_source': str(target)})
    if not records:
        raise RuntimeError('No generated VisionBackbone attention call could be verified')
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-root', type=Path, required=True)
    parser.add_argument('--nnscaler-root', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--vision-deterministic-attention', action='store_true')
    parser.add_argument('train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args[1:] if args.train_args[:1] == ['--'] else args.train_args
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.llm_root.resolve() / 'llm'))
    sys.path.insert(0, str(args.nnscaler_root.resolve()))
    import numpy as np
    import torch
    import nnscaler
    from nnscaler.cli.trainer import Trainer
    from data.vl_data_loader import VLDataLoader

    actual = Path(nnscaler.__file__).resolve()
    if not actual.is_relative_to(args.nnscaler_root.resolve()):
        raise RuntimeError(f'Unexpected nnscaler import: {actual}')
    rank = int(os.environ.get('RANK', '0'))
    stream = (args.audit_dir / f'rank{rank}.jsonl').open('x', buffering=1)

    def record(kind, **values):
        stream.write(json.dumps({'kind': kind, 'rank': rank, 'pid': os.getpid(),
                                 'time_ns': time.time_ns(), **values}, allow_nan=False) + '\n')

    if args.vision_deterministic_attention:
        from arch.vision_encoder import MoonViT3dVisionTower
        install_vision_determinism(MoonViT3dVisionTower, record)

    def digest(value):
        hasher = hashlib.sha256()

        def walk(obj):
            if torch.is_tensor(obj):
                array = obj.detach().cpu().contiguous()
                hasher.update(str((tuple(array.shape), str(array.dtype))).encode())
                if array.numel() == 0:
                    return
                data = memoryview(array.reshape(-1).view(torch.uint8).numpy()).cast('B')
                for start in range(0, len(data), 1024 * 1024):
                    hasher.update(data[start:start + 1024 * 1024])
            elif isinstance(obj, np.ndarray):
                array = np.ascontiguousarray(obj)
                hasher.update(str((array.shape, str(array.dtype))).encode())
                if array.size == 0:
                    return
                data = memoryview(array.reshape(-1)).cast('B')
                for start in range(0, len(data), 1024 * 1024):
                    hasher.update(data[start:start + 1024 * 1024])
            elif isinstance(obj, dict):
                for key in sorted(obj, key=str):
                    hasher.update(repr(key).encode())
                    walk(obj[key])
            elif isinstance(obj, (list, tuple)):
                hasher.update(type(obj).__name__.encode())
                for item in obj:
                    walk(item)
            else:
                hasher.update(repr(obj).encode())

        walk(value)
        return hasher.hexdigest()

    original_next = VLDataLoader.__next__
    original_setstate = VLDataLoader.setstate
    original_metrics = Trainer.log_metrics
    original_rng_resume = Trainer._try_resume_rng_states
    original_setup = Trainer._setup

    def observed_setup(trainer, *positional, **keyword):
        result = original_setup(trainer, *positional, **keyword)
        record('training_configuration', training_horizon=trainer.train_args.max_train_steps,
               runtime_horizon=trainer.max_train_steps,
               finished_train_steps=trainer.train_status.finished_train_steps,
               optimizer_group_lrs=[float(group['lr']) for group in trainer.optimizer.param_groups],
               lr_scheduler=trainer.lr_scheduler.state_dict(),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if rank == 0 and args.vision_deterministic_attention:
            gencode = Path(train_args[train_args.index('--gen_savedir') + 1])
            evidence = verify_generated_vision_attention(gencode, args.audit_dir)
            record('generated_vision_determinism', files=evidence,
                   verified_call_count=sum(len(item['calls']) for item in evidence))
        return result

    def observed_next(loader):
        batch = original_next(loader)
        fields = ('tokens', 'targets', 'loss_mask', 'moe_loss_mask', 'context',
                  'image_input_mask', 'image_grid_thws', 'image_pixel_values')
        audited = {key: batch[key] for key in fields if key in batch}
        grids = batch.get('image_grid_thws')
        record('batch', loader_step=int(loader._step), sha256=digest(audited),
               images=0 if grids is None else int(grids.shape[0]),
               fields=list(audited))
        return batch

    def observed_setstate(loader, state):
        original_setstate(loader, state)
        record('dataloader_restore', loader_step=int(loader._step),
               state_sha256=digest(state))

    def observed_metrics(trainer, metrics, *positional, **keyword):
        tag = keyword.get('tag')
        if tag == 'train' and 'train_loss' in metrics:
            values = {}
            for key in ('train_loss', 'loss', 'gnorm', 'lr', 'bsz', 'train_wall',
                        'FLOPS', 'FLOPS_VISION', 'peak_gb', 'reserved_gb'):
                value = metrics.get(key)
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, (int, float)):
                    values[key] = value
            record('train_metrics', step=int(trainer.train_status.finished_train_steps),
                   values=values,
                   optimizer_group_lrs_after_step=[float(group['lr']) for group in trainer.optimizer.param_groups])
        return original_metrics(trainer, metrics, *positional, **keyword)

    def observed_rng_resume(trainer):
        restored = trainer.rng_states_from_resume is not None
        original_rng_resume(trainer)
        if restored:
            record('rng_restore', step=int(trainer.train_status.finished_train_steps))

    VLDataLoader.__next__ = observed_next
    VLDataLoader.setstate = observed_setstate
    Trainer.log_metrics = observed_metrics
    Trainer._try_resume_rng_states = observed_rng_resume
    Trainer._setup = observed_setup
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    record('process_start', nnscaler_file=str(actual),
           entrypoint=str(args.llm_root / 'llm/nnscaler_train_vl.py'),
           tf32=False, deterministic_env=os.environ.get('DETERMINISTIC'),
           train_arguments=train_args,
           vision_deterministic_attention=args.vision_deterministic_attention)
    sys.argv = [str(args.llm_root / 'llm/nnscaler_train_vl.py'), *train_args]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as exc:
        if exc.code in (None, 0):
            record('process_exit', exit_code=0)
        else:
            record('process_exception', exception_type='SystemExit')
        raise
    except BaseException as exc:
        record('process_exception', exception_type=type(exc).__name__)
        raise
    else:
        record('process_exit', exit_code=0)
    finally:
        stream.close()


if __name__ == '__main__':
    main()
