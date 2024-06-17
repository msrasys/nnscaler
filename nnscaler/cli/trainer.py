from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sys
import copy
import inspect

import torch
import torch.distributed
import nnscaler

from .trainer_args import TrainerArgs


@dataclass
class TrainStatus:
    epoch: int = 0
    in_epoch_pos: int = 0  # the position inside an epoch, used for resuming training


class Trainer:
    def __init__(self,
        argv: Optional[List[str]] = None,
        train_args: Optional[Union[Dict[str, Any], TrainerArgs]] = None
    ):
        """
        Args:
            argv (Optional[List[str]]): command line arguments. If not specified, sys.argv[1:] will be used
            train_args: a dict used to construct TrainerArgs or TrainerArgs object itself.
        """
        if train_args is not None:
            if argv is not None:
                raise ValueError("argv and train_args can not be specified together")
            if isinstance(train_args, TrainerArgs):
                self.train_args = train_args
            else:
                if not isinstance(train_args, dict):
                    raise ValueError(f"train_args should be a dict or TrainerArgs, got {type(train_args)}")
                self.train_args = TrainerArgs.from_dict(train_args)
        else:
            cli_args = argv or sys.argv[1:]  # remve the leading script name from sys.argv
            self.train_args = TrainerArgs.from_cli(cli_args)

        self.model = None
        self.optimizer = None
        self.dataset = {'train': None, 'val': None, 'test': None}
        self.dataloader = {'train': None, 'val': None, 'test': None}
        self.lr_scheduler = None
        self.train_status = TrainStatus()
        self.dummy_input = None
        self._setup()

    def _fix_input(self, input):
        if isinstance(input, dict):
            return {k: self._fix_input(v) for k, v in input.items()}
        elif isinstance(input, list):
            return [self._fix_input(v) for v in input]
        elif isinstance(input, tuple):
            return tuple(self._fix_input(v) for v in input)
        elif isinstance(input, torch.Tensor):
            if self.train_args.fp16:
                return input.half().cuda()
            elif self.train_args.bf16:
                return input.bfloat16().cuda()
            else:
                return input.cuda()
        return input

    def _create_dummy_forward_args(self):
        assert self.dummy_input is not None, "dummy_input is not set"
        assert self.train_args.model_type is not None, "model_type is not set"

        arg_names = list(
            inspect.signature(
                inspect.unwrap(getattr(self.train_args.model_type, 'forward'))
            ).parameters.keys()
        )
        return {arg_names[1]: self.dummy_input}  # arg_names[0] is self

    def _setup(self):
        compile_only = self.train_args.run_mode == 'compile'
        if not compile_only:
            nnscaler.init()

        def _create_model():
            model = self.train_args.create_model()
            if self.train_args.fp16:
                model = model.half()
            elif self.train_args.bf16:
                model = model.bfloat16()
            if self.train_args.ckpt_tracing:
                model.load_state_dict(torch.load(self.train_args.ckpt_tracing))
            return model

        # load a dummy input from training dataset
        if not compile_only:
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = self.train_args.create_dataset(stage)
                self.dataloader[stage] = self.train_args.create_dataloader(stage, self.dataset[stage])

            self.dummy_input = self.dataloader['train'].collate_fn(
                [self.dataset['train'][idx] for idx in range(self.train_args.micro_batch_size)]
            )
        else:
            train_dataset = self.train_args.create_dataset('train')
            self.dummy_input = self.train_args.collate_fn(
                [train_dataset[idx] for idx in range(self.train_args.micro_batch_size)]
            )
            del train_dataset

        self.dummy_input = self._fix_input(self.dummy_input)

        # setup compute config
        compute_config = copy.deepcopy(self.train_args.compute_config)
        compute_config.pas_config['__pas_name'] = self.train_args.pas_policy
        compute_config.user_config['__from_trainer_args'] = {
            'mbs': self.train_args.micro_batch_size,
            'gbs': self.train_args.global_batch_size,
            'fp16': self.train_args.fp16,
            'bf16': self.train_args.bf16,
        }

        # parallalize model
        pmodel_class = nnscaler.parallelize(
            self.train_args.model_type,
            self._create_dummy_forward_args(),
            self.train_args.pas_policy,
            compute_config,
            module_fn=_create_model,
            gen_savedir=self.train_args.gen_savedir,
            reuse='moo' if compile_only else 'match',
            instance_name=self.train_args.instance_name,
            broadcast_strategy='all',
            load_module=not compile_only,
        )
        if compile_only:
            return

        torch.distributed.barrier()

        self.model = pmodel_class()
        self.optimizer = self.train_args.create_parallel_optimizer(self.model)
        self.lr_scheduler = self.train_args.create_lr_scheduler(self.optimizer)
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not self.train_args.ckpt_load_file:
            return
        state_dict = torch.load(self.train_args.ckpt_load_file, map_location='cpu')
        ckpt_save_type = state_dict.get('train_args', {}).get('ckpt_save_type', None)

        if not ckpt_save_type: # it is a merged state dict
            nnscaler.load_merged_state_dicts(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
                )
            if 'lr_scheduler' in state_dict:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        elif ckpt_save_type == 'sharded':
            self.model.load_state_dict(state_dict['model'])
            self.model.cuda()
            self.optimizer.load_state_dict(state_dict['optimizer'])
            if 'lr_scheduler' in state_dict:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            self.train_status = TrainStatus(**state_dict['train_status'])
        elif ckpt_save_type == 'deduped':
            nnscaler.load_deduped_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
            )
            if 'lr_scheduler' in state_dict:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            self.train_status = TrainStatus(**state_dict['train_status'])
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_save_type}")

    def _save_checkpoint(self, from_end_of_epoch=True):
        if not self.train_args.ckpt_save_dir:
            return
        save_dir = Path(self.train_args.ckpt_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.train_args.ckpt_save_type == 'sharded':
            model_state_dict= self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
        elif self.train_args.ckpt_save_type == 'deduped':
            model_state_dict, optimizer_state_dict = nnscaler.deduped_state_dict(
                self.model, self.optimizer
            )
        else:
            raise ValueError(f"Unknown checkpoint type: {self.train_args.ckpt_save_type}")

        train_status = copy.deepcopy(self.train_status)
        if from_end_of_epoch:
            train_status.in_epoch_pos = 0
            train_status.epoch += 1

        state_dict = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'train_status': asdict(train_status),
            'train_args': self.train_args.to_dict(),
        }
        torch.save(state_dict, save_dir /
            f'ckpt_{train_status.epoch}_{train_status.in_epoch_pos}_rank{torch.distributed.get_rank()}.pt'
        )

    def _global_batch_iterator(self, num_skip_first = 0):
        samples = []
        for idx, sample in enumerate(self.dataloader['train']):
            if idx < num_skip_first * self.train_args.update_freq:
                continue
            sample = self._fix_input(sample)
            samples.append(sample)
            if len(samples) == self.train_args.update_freq:
                yield samples
                samples = []
        if samples:
            yield samples

    def train(self):
        num_skip_fist = self.train_status.in_epoch_pos
        for epoch in range(self.train_status.epoch, self.train_args.max_epochs):
            self.train_status.epoch = epoch
            for idx, samples in enumerate(self._global_batch_iterator(num_skip_fist)):
                self.train_status.in_epoch_pos = idx
                is_dummy_batch = [False] * len(samples)
                if len(samples) < self.train_args.update_freq:
                    gap = self.train_args.update_freq - len(samples)
                    is_dummy_batch += [True] * gap
                    samples += [self.dummy_input] * gap

                self.model.train()
                self.optimizer.zero_grad()
                losses = self.model.train_step(samples, is_dummy_batch)
                if self.train_args.clip_gnorm:
                    self.optimizer.clip_gnorm(self.train_args.clip_gnorm)
                self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)

            self._save_checkpoint(True)

            num_skip_fist = 0
