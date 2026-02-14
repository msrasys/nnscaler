#!/usr/bin/env python3
"""Inspect the contents of *.ckpt.model files.

Usage:
    # Inspect a single file
    python inspect_ckpt_model.py tmp/0000-129375/0.ckpt.model

    # Inspect all files in a directory
    python inspect_ckpt_model.py tmp/0000-129375/

    # Inspect with full tensor stats (mean, std, min, max)
    python inspect_ckpt_model.py tmp/0000-129375/0.ckpt.model --stats

    # Show CUBE_EXTRA_STATE metadata
    python inspect_ckpt_model.py tmp/0000-129375/0.ckpt.model --meta

    # Show only summary (one line per file)
    python inspect_ckpt_model.py tmp/0000-129375/ --summary
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import torch


def _human_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def inspect_file(path: str, *, show_stats: bool = False, show_meta: bool = False, summary_only: bool = False):
    file_size = os.path.getsize(path)
    data = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(data, dict):
        print(f"{path}: type={type(data).__name__}, file_size={_human_bytes(file_size)}")
        return

    tensor_keys = sorted(k for k, v in data.items() if isinstance(v, torch.Tensor))
    non_tensor_keys = sorted(k for k, v in data.items() if not isinstance(v, torch.Tensor))
    total_tensor_bytes = sum(
        data[k].nelement() * data[k].element_size() for k in tensor_keys
    )

    # --- summary mode: one line ---
    if summary_only:
        print(
            f"{os.path.basename(path):>20s}  "
            f"file={_human_bytes(file_size):>10s}  "
            f"keys={len(data):>4d}  "
            f"tensors={len(tensor_keys):>4d} ({_human_bytes(total_tensor_bytes):>10s})  "
            f"non_tensor={len(non_tensor_keys)}"
        )
        return

    # --- detailed mode ---
    print(f"\n{'=' * 80}")
    print(f"File: {path}")
    print(f"  File size:     {_human_bytes(file_size)}")
    print(f"  Total keys:    {len(data)}")
    print(f"  Tensor keys:   {len(tensor_keys)} ({_human_bytes(total_tensor_bytes)})")
    print(f"  Non-tensor:    {len(non_tensor_keys)}")

    # Tensor details
    if tensor_keys:
        # Detect dtypes
        dtypes = {}
        for k in tensor_keys:
            t = data[k]
            dt = str(t.dtype)
            if dt not in dtypes:
                dtypes[dt] = {'count': 0, 'bytes': 0}
            dtypes[dt]['count'] += 1
            dtypes[dt]['bytes'] += t.nelement() * t.element_size()
        print(f"\n  Dtype breakdown:")
        for dt, info in sorted(dtypes.items()):
            print(f"    {dt:>20s}: {info['count']:>4d} tensors, {_human_bytes(info['bytes']):>10s}")

        print(f"\n  Tensors:")
        for k in tensor_keys:
            t = data[k]
            line = f"    {k}: shape={list(t.shape)} dtype={t.dtype} device={t.device}"
            if show_stats and t.is_floating_point():
                line += (
                    f" mean={t.float().mean().item():.6g}"
                    f" std={t.float().std().item():.6g}"
                    f" min={t.min().item():.6g}"
                    f" max={t.max().item():.6g}"
                )
            elif show_stats:
                line += f" min={t.min().item()} max={t.max().item()}"
            print(line)

    # Non-tensor details
    if non_tensor_keys:
        print(f"\n  Non-tensor keys:")
        for k in non_tensor_keys:
            v = data[k]
            if show_meta and isinstance(v, dict):
                print(f"    {k}: dict with {len(v)} keys")
                _print_dict(v, indent=6, max_depth=3)
            else:
                print(f"    {k}: {type(v).__name__} ({repr(v)[:120]})")


def _print_dict(d: dict, indent: int = 4, max_depth: int = 2, depth: int = 0):
    """Recursively print a dict with indentation."""
    prefix = ' ' * indent
    for k, v in d.items():
        if isinstance(v, dict) and depth < max_depth:
            print(f"{prefix}{k}: dict ({len(v)} keys)")
            _print_dict(v, indent + 2, max_depth, depth + 1)
        elif isinstance(v, (list, tuple)) and len(v) > 5:
            print(f"{prefix}{k}: {type(v).__name__}[{len(v)}]")
        elif isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: Tensor shape={list(v.shape)} dtype={v.dtype}")
        else:
            s = repr(v)
            if len(s) > 120:
                s = s[:117] + '...'
            print(f"{prefix}{k}: {s}")


def main():
    parser = argparse.ArgumentParser(description="Inspect *.ckpt.model files")
    parser.add_argument('path', help="File or directory to inspect")
    parser.add_argument('--stats', action='store_true', help="Show tensor statistics (mean/std/min/max)")
    parser.add_argument('--meta', action='store_true', help="Expand CUBE_EXTRA_STATE metadata")
    parser.add_argument('--summary', action='store_true', help="One-line-per-file summary")
    parser.add_argument('--sort', choices=['name', 'size', 'tensors'], default='name',
                        help="Sort order when scanning a directory")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        files = [str(target)]
    elif target.is_dir():
        files = sorted(glob.glob(str(target / '*.ckpt.model')))
        if not files:
            print(f"No *.ckpt.model files found in {target}")
            return
    else:
        print(f"Path not found: {target}")
        sys.exit(1)

    if args.sort == 'size':
        files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    elif args.sort == 'tensors':
        # Defer sort after loading — fall back to by-name for now in summary
        pass

    if args.summary:
        total_file_bytes = 0
        total_tensors = 0
        for f in files:
            inspect_file(f, summary_only=True)
            total_file_bytes += os.path.getsize(f)
        print(f"\n{'TOTAL':>20s}  files={len(files)}  disk={_human_bytes(total_file_bytes)}")
    else:
        for f in files:
            inspect_file(f, show_stats=args.stats, show_meta=args.meta)


if __name__ == '__main__':
    main()
