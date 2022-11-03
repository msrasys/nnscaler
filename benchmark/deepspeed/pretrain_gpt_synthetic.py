# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os
import subprocess

from torch import nn
import torch.nn.functional as F

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    vocab_size = 50257
    after = vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while after % multiple != 0:
        after += 1
    args.padded_vocab_size = after

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            print_rank_0('building GPT model using DeepSpeed ...')
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=torch.cuda.current_device())).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                return_moe_loss=False
            )

    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    vocab_size = 50257
    tokens = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    labels = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=torch.cuda.current_device())
    attention_mask = torch.tril(torch.ones(
        (args.micro_batch_size, args.seq_length, args.seq_length), device=torch.cuda.current_device()
    )).view(args.micro_batch_size, 1, args.seq_length, args.seq_length)
    attention_mask = (attention_mask < 0.5)
    position_ids = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    # args = get_args()
    # tokenizer = get_tokenizer()

    # # Items and their type.
    # keys = ['text']
    # datatype = torch.int64
    # 
    # # Broadcast data.
    # data_b = mpu.broadcast_data(keys, data, datatype)
    # 
    # # Unpack.
    # tokens_ = data_b['text'].long()
    # labels = tokens_[:, 1:].contiguous()
    # tokens = tokens_[:, :-1].contiguous()
    # 
    # # Get the masks and postition ids.
    # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
    #     tokens,
    #     tokenizer.eod,
    #     args.reset_position_ids,
    #     args.reset_attention_mask,
    #     args.eod_mask_loss)
    # if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
    #     # seqlen-based curriculum learning
    #     # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
    #     tokens = tokens[:, :args.curriculum_seqlen].contiguous()
    #     position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
    #     if labels is not None:
    #         labels = labels[:, :args.curriculum_seqlen].contiguous()
    #     loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()\
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(None)

    return (tokens, position_ids, attention_mask), (labels, loss_mask)



def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                                        labels=labels)

    moe_loss = 0
    mos_loss = 0
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    return [1]*10000, None, None


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    mem = torch.cuda.max_memory_allocated()
    for rank in range(torch.distributed.get_world_size()):
        if rank == torch.distributed.get_rank():
            print(f'rank[{rank}]: memory consumption: {round(mem / 1024 / 1024 / 1024 * 100) / 100} GBs')
        torch.distributed.barrier()