#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Subprocess entry point for multi-process per-rank code generation."""

import argparse
import logging
from pathlib import Path
import time

import dill

from nnscaler.flags import CompileFlag
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.runtime.module import ParallelModule


logger = logging.getLogger(__name__)


def _restore_compile_flags(flags: dict[str, object]) -> None:
    for name, value in flags.items():
        setattr(CompileFlag, name, value)


def generate_rank_range(payload_file: Path, outdir: Path, rank_start: int, rank_end: int) -> None:
    with payload_file.open('rb') as payload_stream:
        payload = dill.load(payload_stream)

    _restore_compile_flags(payload['compile_flags'])
    CustomizedOps.kOpEmit.clear()
    CustomizedOps.kOpEmit.update(payload['custom_op_emit_registry'])

    module_codegen = payload['module_codegen']
    schedule_codegen = payload['schedule_codegen']
    forward_args = payload['forward_args']
    end2end_mode = payload['end2end_mode']
    gencode_file_template = payload['gencode_file_template']

    for rank in range(rank_start, rank_end):
        code_file = outdir / gencode_file_template.format(rank)
        attr_meta_file = outdir / ParallelModule.ATTR_META_FILE_TEMPLATE.format(rank)
        module_codegen.gen(
            rank,
            forward_args=forward_args,
            outfile=code_file,
            attach=False,
            as_parallel_module=True,
            end2end_mode=end2end_mode,
            outfile_attr_meta_map=attr_meta_file,
        )
        if end2end_mode:
            schedule_codegen.gen(device=rank, outfile=code_file, attach=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='NNScaler multi-process codegen worker')
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--rank-start', type=int, required=True)
    parser.add_argument('--rank-end', type=int, required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    args = parser.parse_args()

    if args.rank_start < 0 or args.rank_end <= args.rank_start:
        parser.error('worker rank range must be non-empty and non-negative')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )
    started_at = time.monotonic()
    logger.info(
        'Codegen worker %d started for ranks [%d, %d)',
        args.worker_id,
        args.rank_start,
        args.rank_end,
    )
    generate_rank_range(args.payload, args.outdir, args.rank_start, args.rank_end)
    logger.info(
        'Codegen worker %d completed ranks [%d, %d) in %.2f seconds',
        args.worker_id,
        args.rank_start,
        args.rank_end,
        time.monotonic() - started_at,
    )


if __name__ == '__main__':
    main()
