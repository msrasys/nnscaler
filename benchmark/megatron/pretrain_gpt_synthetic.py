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
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    vocab_size = 50257
    after = vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while after % multiple != 0:
        after += 1
    args.padded_vocab_size = after

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    vocab_size = 50257
    tokens = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    labels = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((args.micro_batch_size, 1, args.seq_length, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)    
    position_ids = torch.rand((args.micro_batch_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
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

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    return None, None, None


if __name__ == "__main__":

    

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    mem = torch.cuda.max_memory_allocated()
    for rank in range(torch.distributed.get_world_size()):
        if rank == torch.distributed.get_rank():
            print(f'rank[{rank}]: memory consumption: {round(mem / 1024 / 1024 / 1024 * 100) / 100} GBs')
        torch.distributed.barrier()