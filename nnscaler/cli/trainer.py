from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sys, os
import copy
import inspect
import warnings
import shutil
import logging

import torch
import torch.distributed
from torch.utils.data import DataLoader

from tqdm import tqdm

import nnscaler
from nnscaler.utils import enforce_zero_num_worker
import nnscaler.utils

from .trainer_args import AggregatedOutputs, TrainerArgs
from .train_hook import AggregatedTrainHook, TrainHook


logger = logging.getLogger(__name__)


# the format of the checkpoint file
# keys: epoch, step, rank
# currently it is not configurable
# TODO: make it configurable
CHECKPOINT_FILE_FORMAT: str = '{epoch:04d}-{step:04d}/{rank}.ckpt'
CHECKPOINT_LAST_DIR_NAME: str = 'last'
CHECKPOINT_BEST_DIR_NAME: str = 'best'
CHECKPOINT_LAST_FILE_FORMAT: str = 'last/{rank}.ckpt'
CHECKPOINT_BEST_FILE_FORMAT: str = 'best/{rank}.ckpt'


@dataclass
class TrainStatus:
    best_loss = float('inf')
    epoch: int = 0
    # used for resuming training
    # it is the index of the next batch in the current epoch
    # i means the i-1 batch is done, and we should resume from ith batch
    # for example
    # 0 means the epoch is not started
    # 1 means the 0th batch is done, and we should resume from 1st batch
    next_batch_index: int = 0


@dataclass
class _StepStat:
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    lr: Optional[float] = None
    gnorm: Optional[float] = None


class Trainer:
    def __init__(self,
        argv: Optional[List[str]] = None,
        *,
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
            cli_args = argv or sys.argv[1:]  # remove the leading script name from sys.argv
            self.train_args = TrainerArgs.from_cli(cli_args)

        self.rank = None
        self.sync_group = None
        self.model = None
        self.optimizer = None
        self.dataset = {'train': None, 'val': None, 'test': None}
        self.dataloader: Dict[str, Optional[DataLoader]] = {'train': None, 'val': None, 'test': None}
        self.lr_scheduler = None
        self.train_status = TrainStatus()
        self.dummy_input = None
        self.total_train_steps_per_epoch = None
        self.loggers = []
        self.hook = None
        self._setup()

    @property
    def num_train_steps(self):
        return self.train_status.epoch  * self.total_train_steps_per_epoch + self.train_status.next_batch_index

    def _fix_input(self, input):
        if isinstance(input, dict):
            return {k: self._fix_input(v) for k, v in input.items()}
        elif isinstance(input, list):
            return [self._fix_input(v) for v in input]
        elif isinstance(input, tuple):
            return tuple(self._fix_input(v) for v in input)
        elif isinstance(input, torch.Tensor):
            if input.is_floating_point() and self.train_args.input_dtype is not None:
                return input.to(self.train_args.input_dtype).cuda()
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

    def _load_dummy_input(self):
        with enforce_zero_num_worker(DataLoader):
            assert self.dataset['train'] is not None, "train dataset is not set"
            dataloader = self.train_args.create_dataloader('train', self.dataset['train'])
            assert dataloader.num_workers == 0, "The dataloader must have `num_workers=0`."
            return next(iter(dataloader))

    def _setup(self):
        self.train_args.init_env()
        compile_only = self.train_args.run_mode == 'compile'
        if not compile_only:
            nnscaler.init()
            if torch.distributed.get_rank() == 0:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)

        def _create_model():
            model = self.train_args.create_model()
            if self.train_args.param_dtype == self.train_args.buffer_dtype:
                if self.train_args.param_dtype is not None:
                    model = model.to(self.train_args.param_dtype)
            else:
                # separate param and buffer dtype
                # TODO: a little hacky. A better way?
                # 3 kinds of tensors are converted in Module._apply:
                # model parameters, its grad, and buffer
                # param_dtype controls the first two, (but grad is `None` here)
                # and buffer_dtype controls the last one
                buf_ids = { id(buf) for buf in model.buffers(recurse=True) }
                if self.train_args.param_dtype is not None:
                    model._apply(
                        lambda t: t.to(self.train_args.param_dtype)
                            if t.is_floating_point() and id(t) not in buf_ids
                            else t)
                if self.train_args.buffer_dtype is not None:
                    model._apply(
                        lambda t: t.to(self.train_args.buffer_dtype)
                            if t.is_floating_point() and id(t) in buf_ids
                            else t)
            if self.train_args.tracing_from_weights:
                model.load_state_dict(torch.load(self.train_args.tracing_from_weights))
            return model

        # create dataset and dataloader
        for stage in ['train', 'val', 'test']:
            self.dataset[stage] = self.train_args.create_dataset(stage)

        # load a dummy input from training dataset
        self.dummy_input = self._load_dummy_input()
        self.dummy_input = self._fix_input(self.dummy_input)

        for stage in ['train', 'val', 'test']:
            self.dataloader[stage] = self.train_args.create_dataloader(stage, self.dataset[stage])
            if self.dataloader[stage] is not None \
                and not self.dataloader[stage].drop_last \
                and len(self.dataset[stage]) % (self.train_args.micro_batch_size * self.train_args.scaling_factor) != 0:
                    warnings.warn(
                        f"Length of {stage} dataset ({len(self.dataset[stage])}) "
                        f"is not multiple of micro_batch_size * scale_factor ({self.train_args.micro_batch_size * self.train_args.scaling_factor}). "
                        f"In this case, the train_step for the last batch of samples can fail! "
                        f"You can specify `drop_last=True` in DataLoader to fix this problem."
                    )

        # setup compute config
        compute_config = copy.deepcopy(self.train_args.compute_config)
        compute_config.pas_config['__pas_name'] = self.train_args.pas_policy
        # autodist configs
        compute_config.pas_config['update_freq'] = self.train_args.update_freq
        compute_config.pas_config['use_bf16'] = self.train_args.param_dtype == torch.bfloat16
        compute_config.pas_config['use_fp16'] = self.train_args.param_dtype == torch.float16

        compute_config.user_config['__from_trainer_args'] = {
            'mbs': self.train_args.micro_batch_size,
            'gbs': self.train_args.global_batch_size,
            'precision': self.train_args.precision,
            'model_args': self.train_args.model.args,
        }

        # parallalize model
        pmodel_class = nnscaler.parallelize(
            self.train_args.model_type,
            self._create_dummy_forward_args(),
            self.train_args.resolved_pas_policy,
            compute_config,
            module_fn=_create_model,
            gen_savedir=self.train_args.gen_savedir,
            reuse=self.train_args.gen_reuse,
            instance_name=self.train_args.instance_name,
            broadcast_strategy='all',
            load_module=not compile_only,
        )
        if compile_only:
            return

        torch.distributed.barrier()
        self.rank = torch.distributed.get_rank()
        self.total_train_steps_per_epoch = len(self.dataloader['train']) // self.train_args.update_freq
        if len(self.dataloader['train']) % self.train_args.update_freq != 0:
            self.total_train_steps_per_epoch += 1  # will add extra dummy batches
        _, self.sync_group = self.train_args.compute_config.get_sync_group()
        self.model = pmodel_class()
        self.model.cuda()
        self.optimizer = self.train_args.create_parallel_optimizer(self.model)
        self.lr_scheduler = self.train_args.create_lr_scheduler(self.optimizer)
        self.loggers = self.train_args.create_loggers()

        supported_hook_components = [
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ]
        self.hook = AggregatedTrainHook(
            [x for x in supported_hook_components if isinstance(x, TrainHook)]
            + [self.train_args.create_hook()]
        )

        self._log_config(self.train_args.to_dict())
        self._load_checkpoint()

        self.hook.after_setup(self)

    @classmethod
    def merge_checkpoint(cls, checkpoint_files: List[str], output_file: str):
        state_dicts = [torch.load(f, map_location='cpu') for f in checkpoint_files]
        for i in range(1, len(state_dicts)):
            if state_dicts[i]['train_args'] != state_dicts[0]['train_args']:
                raise ValueError(f"train_args in {checkpoint_files[i]} is different from {checkpoint_files[0]}")

        module_state_dict, opt_state_dict = nnscaler.merge_state_dicts(
            [s['model'] for s in state_dicts],
            [s['optimizer'] for s in state_dicts]
        )
        train_args = copy.deepcopy(state_dicts[0]['train_args'])
        train_args['checkpoint']['save_type'] = 'merged'
        merged_state_dict = {
            'model': module_state_dict,
            'optimizer': opt_state_dict,
            'lr_scheduler': state_dicts[0].get('lr_scheduler', None),
            'train_status': state_dicts[0]['train_status'],
            'train_args': train_args,
        }
        torch.save(merged_state_dict, output_file)

    def _log_finalize(self):
        for logger in self.loggers:
            logger.finalize()

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def _log_config(self, config: Dict):
        for logger in self.loggers:
            logger.setup(config)

    def _load_checkpoint(self):
        resume_from = self.train_args.checkpoint.get_resume_checkpoint_dir()
        if not resume_from:
            return
        logger.info(f"Resuming from {resume_from}")
        if resume_from.is_file():
            resume_from = resume_from   # when we load from merged checkpoint
        else:
            resume_from = resume_from / f'{self.rank}.ckpt'
        state_dict = torch.load(resume_from, map_location='cpu')
        self.hook.on_load_checkpoint(self, state_dict)
        ckpt_save_type = state_dict['train_args']['checkpoint']['save_type']

        if ckpt_save_type == 'merged': # it is a merged state dict
            nnscaler.load_merged_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
                )
        elif ckpt_save_type == 'sharded':
            nnscaler.load_sharded_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
            )
        elif ckpt_save_type == 'deduped':
            nnscaler.load_deduped_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
            )
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_save_type}")

        if 'lr_scheduler' in state_dict:
            if state_dict['lr_scheduler'] and not self.lr_scheduler:
                raise ValueError("lr_scheduler is not set in the current trainer")
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.train_status = TrainStatus(**state_dict['train_status'])

    def _save_checkpoint(self, loss):
        checkpoint_config = self.train_args.checkpoint

        if checkpoint_config.no_save:
            logger.info('Skip saving checkpoint because `no_save` is set to True')
            return

        torch.distributed.barrier()
        logger.info(f"Saving checkpoint after {self.num_train_steps} steps with loss={loss:.3f}.")
        save_dir = Path(checkpoint_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_config.save_type == 'sharded':
            model_state_dict= self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
        elif checkpoint_config.save_type == 'deduped':
            model_state_dict, optimizer_state_dict = nnscaler.deduped_state_dict(
                self.model, self.optimizer
            )
        elif checkpoint_config.save_type == 'merged':
            raise ValueError("merged checkpoint is not supported for saving")
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_config.save_type}")

        state_dict = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'train_status': asdict(self.train_status),
            'train_args': self.train_args.to_dict(),
        }
        self.hook.on_save_checkpoint(self, state_dict)
        ckpt_file = save_dir / CHECKPOINT_FILE_FORMAT.format(
            epoch=self.train_status.epoch,
            step=self.num_train_steps,
            rank=self.rank,
        )
        logger.info(f"Saving checkpoint to {str(ckpt_file.parent)}")
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, ckpt_file)

        # save last
        if checkpoint_config.save_last:
            logger.info(f"Saving checkpoint as the last checkpoint.")
            last_file = save_dir / CHECKPOINT_LAST_FILE_FORMAT.format(
                rank=self.rank
            )
            last_file.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_config.symlink_best_and_last:
                # remove the old symlink or file
                if last_file.is_symlink() or last_file.exists():
                    last_file.unlink()
                # symblink as relative path
                last_file.symlink_to(Path('..') / ckpt_file.parent.name / ckpt_file.name)
                # last_file.symlink_to(ckpt_file)
            else:
                shutil.copy(ckpt_file, last_file)

        # save best
        if checkpoint_config.save_best and loss <= self.train_status.best_loss:
            logger.info(f"Best loss updated: {self.train_status.best_loss:.3f} -> {loss:.3f}")
            logger.info(f"Saving checkpoint as the best checkpoint.")
            best_file = save_dir / CHECKPOINT_BEST_FILE_FORMAT.format(
                epoch=self.train_status.epoch,
                step=self.num_train_steps,
                rank=self.rank,
            )
            best_file.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_config.symlink_best_and_last:
                # symblink as relative path
                if best_file.is_symlink() or best_file.exists():
                    best_file.unlink()
                best_file.symlink_to(Path('..') / ckpt_file.parent.name / ckpt_file.name)
                # best_file.symlink_to(ckpt_file)
            else:
                shutil.copy(ckpt_file, best_file)

        torch.distributed.barrier()
        # remove old checkpoints
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        # only the first rank in the group will do the job
        if self.rank % local_world_size == 0:
            try:
                self._expire_checkpoints()
            except Exception as e:
                logger.warning('Error when removing old checkpoints: %s. Will try later.', e)

        torch.distributed.barrier()

    def _expire_checkpoints(self):
        if not self.train_args.checkpoint.keep_last_n_checkpoints:  # keep all
            return

        save_dir = Path(self.train_args.checkpoint.save_dir)
        checkpoints = [
            p.name for p in save_dir.glob('*')
            if p.is_dir() and p.name not in [CHECKPOINT_BEST_DIR_NAME, CHECKPOINT_LAST_DIR_NAME]
        ]
        if len(checkpoints) <= self.train_args.checkpoint.keep_last_n_checkpoints:
            return

        # (step, num) pairs
        checkpoint_info = [(int(p.split('-')[1]), p) for p in checkpoints]
        checkpoint_info.sort()
        expire_list = checkpoint_info[:-self.train_args.checkpoint.keep_last_n_checkpoints]

        best_ckpt = save_dir / CHECKPOINT_BEST_DIR_NAME
        if best_ckpt.exists():
            for p in best_ckpt.glob('*.ckpt'):
                if p.is_symlink():
                    ckpt_name = p.resolve().parent.name
                    if ckpt_name in expire_list:
                        expire_list.remove(ckpt_name)
                        logger.info('Keep old checkpoint `%s` because it is the best.', ckpt_name)
                break # just check the first file is enough

        for _, ckpt_name in expire_list:
            logger.info('Removing old checkpoint: %s', ckpt_name)
            shutil.rmtree(save_dir / ckpt_name)

    def _global_batch_iterator(self, num_skip_first = 0, stage='train'):
        samples = []
        for idx, sample in enumerate(self.dataloader[stage]):
            if idx < num_skip_first * self.train_args.update_freq:
                continue
            sample = self._fix_input(sample)
            samples.append(sample)
            if len(samples) == self.train_args.update_freq:
                yield samples
                samples = []
        if samples:
            yield samples

    def aggregate_outputs(self, loss_outputs, sync_group) -> AggregatedOutputs:
        # loss is the first element of the output (or the only element)
        losses = [
            loss if isinstance(loss, torch.Tensor)
            else loss[0]
            for loss in loss_outputs
        ]
        loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
        torch.distributed.all_reduce(loss_sum, group=sync_group)
        num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
        torch.distributed.all_reduce(num_batches, group=sync_group)

        return AggregatedOutputs(
            loss_sum = loss_sum.item(),
            num_batches=num_batches.item(),
        )

    def _fix_batches(self, batches):
        num_batches = len(batches)
        is_dummy_batch = [False] * num_batches
        if num_batches < self.train_args.update_freq:
            gap = self.train_args.update_freq - num_batches
            is_dummy_batch += [True] * gap
            batches += [self.dummy_input] * gap
        return batches, is_dummy_batch

    def train(self):
        assert self.train_status.next_batch_index <= self.total_train_steps_per_epoch, \
            f"next_batch_index({self.train_status.next_batch_index}) " \
            f"should not be larger than total_train_steps_per_epoch ({self.total_train_steps_per_epoch})"

        if self.train_status.next_batch_index == self.total_train_steps_per_epoch:
            self.train_status.epoch += 1
            self.train_status.next_batch_index = 0

        next_batch_index = self.train_status.next_batch_index
        self.hook.on_train_start(self)

        for epoch in range(self.train_status.epoch, self.train_args.max_epochs or sys.maxsize):
            self.dataloader['train'].sampler.set_epoch(epoch)

            torch.distributed.barrier()
            self.train_status.epoch = epoch
            self.train_status.next_batch_index = next_batch_index

            self.hook.on_epoch_start(self, epoch)
            self.train_epoch(epoch)
            self.hook.on_epoch_end(self, epoch)

            if self.lr_scheduler and self.train_args.lr_scheduler.interval == 'epoch':
                self.lr_scheduler.step()

            if self.train_args.max_train_steps and self.num_train_steps >= self.train_args.max_train_steps:
                logger.info(f"Reached max train steps({self.train_args.max_train_steps}): Training is done.")
                break

            next_batch_index = 0
        else:  # not break from for loop, which means not finished with max_train_steps
            # finished with max_epochs
            logger.info(f"Reached max_epochs({self.train_args.max_epochs}): Training is done.")

        self._log_finalize()
        self.hook.on_train_end(self)
        torch.distributed.barrier()

    def _validate_and_save(self, step_stat: _StepStat):
        if self.dataloader['val'] is None:
            self._save_checkpoint(step_stat.train_loss)
            return

        if step_stat.val_loss is None:
            self._validate(step_stat)  # will update step_stat.val_loss internally

        loss = step_stat.val_loss
        self._save_checkpoint(loss)
        if self.train_status.best_loss > loss:
            self.train_status.best_loss = loss

    def _validate(self, step_stat: _StepStat):
        if self.dataloader['val'] is None:
            logger.info('No val dataset specified. Use train_loss as val_loss.')
            step_stat.val_loss = step_stat.train_loss
            return step_stat.val_loss

        data_iter = enumerate(self._global_batch_iterator(stage='val'))
        if self.rank == 0:
            total_val_steps_per_epoch = len(self.dataloader['val']) // self.train_args.update_freq
            if len(self.dataloader['val']) % self.train_args.update_freq != 0:
                total_val_steps_per_epoch += 1  # will add extra dummy batches
            data_iter = tqdm(
                data_iter,
                total=total_val_steps_per_epoch,
                initial=0,
                desc=f'Validating',
                disable=not self.train_args.enable_progress_bar
            )

        loss_sum = 0.0
        batches_count = 0

        self.hook.on_val_start(self)
        for idx, batches in data_iter:
            if self.train_args.max_val_steps and idx >= self.train_args.max_val_steps:
                break

            num_batches = len(batches)
            batches, _ = self._fix_batches(batches)

            self.model.eval()
            with torch.inference_mode():
                self.hook.on_val_step_start(self, batches[:num_batches], idx)
                losses = self.model.infer_step(batches)
                self.hook.on_val_step_end(self, losses[:num_batches], batches[:num_batches], idx)

            aggregate_outputs = self.train_args.resolved_aggregate_outputs_fn or self.aggregate_outputs
            aggregated_outputs = aggregate_outputs(losses[:num_batches], self.sync_group)
            self.hook.after_aggregate_val_step_outputs(
                self, aggregated_outputs,
                aggregated_outputs.loss_sum/aggregated_outputs.num_batches,
                idx
            )
            loss_sum += aggregated_outputs.loss_sum
            batches_count += aggregated_outputs.num_batches

        # update train status
        loss = loss_sum / batches_count
        self.hook.on_val_end(self, loss)

        step_stat.val_loss = loss
        self._log_metrics(asdict(step_stat), self.num_train_steps)
        return step_stat.val_loss

    def train_epoch(self, epoch):
        VAL_STATUS_NO = 0     # not validated or saved
        VAL_STATUS_VAL = 1    # validated but not saved
        VAL_STATUS_SAVE = 2   # validated and saved
        has_validated = VAL_STATUS_NO   # 3 states
        resume_from_idx = self.train_status.next_batch_index
        data_iter = enumerate(self._global_batch_iterator(num_skip_first=resume_from_idx))
        if self.rank == 0:
            data_iter = tqdm(
                data_iter,
                total=self.total_train_steps_per_epoch,
                initial=resume_from_idx,
                desc=f'Epoch {epoch:04d}',
                disable=not self.train_args.enable_progress_bar
            )

        step_stat: Optional[_StepStat] = None
        for idx, batches in data_iter:
            step_stat = _StepStat()
            has_validated = VAL_STATUS_NO
            # the current batch is idx + resume_from_idx
            # `+1` because the next_batch_index is the index of the next batch
            # all save_checkpoint will be done at the end of the loop with correct next_batch_index
            self.train_status.next_batch_index = idx + resume_from_idx + 1
            num_batches = len(batches)
            batches, is_dummy_batch = self._fix_batches(batches)

            self.model.train()

            self.hook.before_zero_grad(self)
            self.optimizer.zero_grad()
            self.hook.after_zero_grad(self)

            self.hook.on_train_step_start(self, batches[:num_batches], idx)
            losses = self.model.train_step(batches, is_dummy_batch)
            self.hook.on_train_step_end(self, losses[:num_batches], batches[:num_batches], idx)

            aggregate_outputs = self.train_args.resolved_aggregate_outputs_fn or self.aggregate_outputs
            aggregated_outputs = aggregate_outputs(losses[:num_batches], self.sync_group)
            if self.train_args.optimizer.loss_reduction == 'mean':
                loss = aggregated_outputs.loss_sum / aggregated_outputs.num_batches
            else:
                loss = aggregated_outputs.loss_sum
            step_stat.train_loss = loss
            self.hook.after_aggregate_train_step_outputs(self, aggregated_outputs, loss, idx)
            if self.rank == 0:
                data_iter.set_postfix({'loss': loss})

            self.hook.before_sync_grad(self)
            # actually `sync_shard_grad` is no-op here
            # because trainer only supports end2end model
            # and syncing grad in end2end model is done in `_train_step`.
            self.optimizer.sync_shard_grad()
            self.hook.after_sync_grad(self)

            # scale gradients
            if self.train_args.optimizer.grad_reduction == 'sum':
                # do nothing. Already done in reducers
                pass
            elif self.train_args.optimizer.grad_reduction == 'mean':
                if not aggregated_outputs.num_batches:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_batches` field")
                self.optimizer.scale_grads(1.0 / aggregated_outputs.num_batches)
            else:
                assert self.train_args.optimizer.grad_reduction == 'per-token-mean'
                if not aggregated_outputs.num_tokens:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_tokens` field")
                self.optimizer.scale_grads(1.0 / aggregated_outputs.num_tokens)

            # clip gradients
            self.hook.before_gnorm_clip(self)
            if self.train_args.optimizer.clip_gnorm:
                step_stat.gnorm = self.optimizer.clip_gnorm(self.train_args.optimizer.clip_gnorm)
            else:
                step_stat.gnorm = self.optimizer.clip_gnorm()
            self.hook.after_gnorm_clip(self, step_stat.gnorm)
            step_stat.gnorm = step_stat.gnorm.item()

            # update parameters
            step_stat.lr = self.optimizer.param_groups[0]['lr']
            self.hook.before_optimizer_step(self)
            self.optimizer.step()
            self.hook.after_optimizer_step(self)
            if self.lr_scheduler and self.train_args.lr_scheduler.interval == 'step':
                self.lr_scheduler.step()

            # validate and save checkpoint
            if self.train_args.checkpoint.every_n_train_steps and \
                self.num_train_steps % self.train_args.checkpoint.every_n_train_steps == 0:
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE

            if self.train_args.max_train_steps and self.num_train_steps >= self.train_args.max_train_steps:
                if not has_validated:
                    self._validate_and_save(step_stat)
                    has_validated = VAL_STATUS_SAVE
                break

            if not has_validated and self.train_args.val_every_n_train_steps and \
                self.num_train_steps % self.train_args.val_every_n_train_steps == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL
            # import time
            # time.sleep(0.2)
        else:  # not finished with max_train_steps
            if step_stat is None:
                return  # no train step runs. No need to save checkpoint
            if has_validated < VAL_STATUS_SAVE and \
                self.train_args.max_epochs == self.train_status.epoch + 1 \
                or (self.train_args.checkpoint.every_n_epochs and \
                (self.train_status.epoch + 1) % self.train_args.checkpoint.every_n_epochs == 0):
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE
            if not has_validated and self.train_args.val_every_n_epochs and \
                (self.train_status.epoch + 1) % self.train_args.val_every_n_epochs == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL
