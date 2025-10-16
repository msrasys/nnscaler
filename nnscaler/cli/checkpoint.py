#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Only for command line
"""

import logging
import os
import sys
from pathlib import Path

import torch.distributed

import nnscaler
from nnscaler.cli.trainer import Trainer, TrainerArgs
from nnscaler.parallel import _trim_module_merged_state_dict, _trim_optimizer_merged_state_dict


logger = logging.getLogger(__name__)


def _patch_distributed():
    groups = {}

    def is_initialized():
        return bool(groups)

    torch.distributed.is_initialized = is_initialized

    def init_process_group(*args, **kwargs):
        world_size = int(os.environ['WORLD_SIZE'])
        groups[None] = list(range(world_size))

    def get_rank(group=None):
        if group not in groups:
            raise ValueError(f"Unknown group: {group}")
        try:
            return groups[group].index(int(os.environ['RANK']))
        except ValueError:
            return -1

    def get_world_size(group=None):
        if group not in groups:
            raise ValueError(f"Unknown group: {group}")
        return len(groups[group])

    def new_group(ranks=None, *args, **kwargs):
        world_size = int(os.environ['WORLD_SIZE'])
        if ranks is None or len(ranks) == world_size:
            return
        group_id = tuple(sorted(ranks))
        if group_id in groups:
            return group_id
        groups[group_id] = ranks
        return group_id

    torch.distributed.get_rank = get_rank
    torch.distributed.get_world_size = get_world_size
    torch.distributed.init_process_group = init_process_group
    torch.distributed.destroy_process_group = lambda: None
    torch.distributed.new_group = new_group
    torch.distributed.barrier = lambda *args, **kwargs: None
    torch.distributed.all_gather = lambda *args, **kwargs: None
    torch.distributed.broadcast_object_list = lambda *args, **kwargs: None


def _trim_merged_checkpoint(train_args: TrainerArgs, merged_state_dict, rank: int):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(train_args.compute_config.runtime_ngpus)
    os.environ['GROUP_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = '1'
    os.environ['TORCHELASTIC_RUN_ID'] = '0' # fake torchrun env

    sharded_state_dict = {k: v for k, v in merged_state_dict.items()}

    trainer = Trainer(train_args=train_args)
    # enforce run mode to load module and optimizer
    trainer.train_args.run_mode = 'run'
    trainer._setup()

    sharded_state_dict['model'] = _trim_module_merged_state_dict(
        trainer.model, merged_state_dict['model'],
        device='cpu'
    )
    sharded_state_dict['optimizer'] = _trim_optimizer_merged_state_dict(
        trainer.model, trainer.optimizer._extra_state, merged_state_dict['optimizer'],
        device='cpu'
    )
    sharded_state_dict['train_args'] = train_args.to_dict()
    sharded_state_dict['train_args'].setdefault('checkpoint', {})['save_type'] = 'sharded'
    # discard rng_states for merged state dict
    sharded_state_dict.pop('rng_states', None)
    if 'dataloader' in sharded_state_dict and sharded_state_dict['dataloader'] is not None:
        # keep dataloader state only when all ranks have the same state
        dataloader_states = sharded_state_dict['dataloader']
        if all(dataloader_states[i] == dataloader_states[0] for i in range(1, len(dataloader_states))):
            sharded_state_dict['dataloader'] = dataloader_states[0]
        else:
            sharded_state_dict.pop('dataloader')

    # make it sharded checkpoint
    for module_path, m in trainer.model.named_modules():
        prefix = module_path + '.' if module_path else ''
        if isinstance(m, nnscaler.ParallelModule):
            m._add_extra_state(sharded_state_dict['model'], prefix)
    return sharded_state_dict


def _distribute_checkpoint(train_args: TrainerArgs, from_: str, to_: str):
    nnscaler.utils.set_default_logger_level(level=logging.INFO)
    _patch_distributed()
    resume_from = Path(from_)
    save_to = Path(to_)
    save_to.mkdir(parents=True, exist_ok=True)

    if resume_from.is_file():
        state_dict = torch.load(resume_from, map_location='cpu', weights_only=False)
        if convert_fn := train_args.checkpoint.resolved_convert_fn:
            state_dict = convert_fn(state_dict)
    else:
        ckpt_files = list(resume_from.glob('*.ckpt'))
        rank_ckpt_files = {int(f.stem): f for f in ckpt_files if f.stem.isdigit()}
        if set(rank_ckpt_files.keys()) != set(range(len(rank_ckpt_files))):
            raise ValueError(f"Checkpoint files in {resume_from} are not complete: {rank_ckpt_files.keys()}")
        state_dict = Trainer._merge_checkpoint(list(rank_ckpt_files.values()))

    for i in range(train_args.compute_config.runtime_ngpus):
        sharded_state_dict = _trim_merged_checkpoint(train_args, state_dict, i)
        torch.save(sharded_state_dict, save_to / f"{i}.ckpt")


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        raise ValueError("No command specified. Expected `distribute <from> <to> -f <trainer args yml> <optional other trainer args>`")
    if argv[0] == 'distribute':
        if len(argv) < 5:
            raise ValueError("Not enough arguments. Expected at least `distribute <from> <to> -f <trainer args yml> <optional other trainer args>`")
        from_ = argv[1]
        to_ = argv[2]
        train_args = TrainerArgs.from_cli(argv[3:])
        # never broadcast generated files.
        train_args.broadcast_strategy = 'none'
        train_args.checkpoint.resume_from = None
        _distribute_checkpoint(train_args, from_, to_)
    else:
        raise ValueError(f"Unknown command: {argv[0]}")
else:
    # we have patched too many things.
    # please run this script with `python -m nnscaler.cli.checkpoint`
    raise ImportError("checkpoint.py should be run as a script.")
