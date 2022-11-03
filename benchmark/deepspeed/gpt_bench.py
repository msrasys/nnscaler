"""
Following

https://github.com/microsoft/DeepSpeedExamples/blob/master/HelloDeepSpeed/train_bert_ds.py

Config file:
https://www.deepspeed.ai/docs/config-json/

deepspeed --num_nodes 1 --num_gpus 8 \
    benchmark/deepspeed/gpt_bench.py \
        --fp16 --mbs 1 --gbs 4 \
        --zero 2 \
        --layers 24 --heads 32 --hidden 2048 --seqlen 2048

"""

from typing import List, Tuple
import torch
import time
import numpy as np
import os
import logging

from examples.nlp.gpt.model import GPT, Config
from examples.nlp.gpt.model import GPTDataLoader

import argparse
import deepspeed

logging.getLogger().setLevel(logging.WARN)


parser = argparse.ArgumentParser(description='GPT Train')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--mbs', type=int, default=1,
                    help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256,
                    help='global batch size')
parser.add_argument('--zero', type=int, required=True,
                    help='zero stage, 2 or 3')
parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--hidden', type=int, required=True)

parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

print(args)
torch.cuda.set_device(args.local_rank)

ds_zero3_config = {
    "train_micro_batch_size_per_gpu": args.mbs,
    "gradient_accumulation_steps": args.gbs // args.mbs,
    "zero_optimization": {
        "stage": 3,
        "offload_param": {  # Zero-3
            "device": "cpu"
        },
        "offload_optimizer": {  # Zero-2
            "device": "cpu"
        },
        "contiguous_gradients": True,
        "overlap_comm": True,
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
            "betas": [0.9, 0.95]
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "wall_clock_breakdown": True,
    "steps_per_print": 1,
}


ds_zero2_config = {
    "train_micro_batch_size_per_gpu": args.mbs,
    "gradient_accumulation_steps": args.gbs // args.mbs,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {  # Zero-2
            "device": "cpu"
        },
        "contiguous_gradients": True,
        "overlap_comm": True,
    },
    "mp_size": 2,
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
            "betas": [0.9, 0.95]
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "wall_clock_breakdown": True,
    "steps_per_print": 1,
}

assert args.zero in [2, 3], f"Zero stage can only be 2 or 3"
zero_config = ds_zero2_config if args.zero == 2 else ds_zero3_config

def log_dist(message: str, ranks: List[int] = None) -> None:
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        print(f"rank [{my_rank}] {message}")


def train():

    batch_size = args.mbs
    Config.seqlen = args.seqlen
    Config.layers = args.layers
    Config.embed_dim = args.hidden
    Config.attention_heads = args.heads

    model = GPT()
    model = model if not args.fp16 else model.half()

    nparams = 0
    param: torch.Tensor
    for param in model.parameters():
        nparams += param.nelement()
    log_dist(f'parameter before zero: {nparams}', [0])
  
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=zero_config)
    model.train()
    log_dist("DeepSpeed engine created", ranks=[0])

    nparams = 0
    param: torch.Tensor
    for param in model.parameters():
        nparams += param.nelement()
    log_dist(f'parameter after zero: {nparams}', [0])

    dataloader = GPTDataLoader(batch_size)


    iter_num = 3
    warmup = 1
    for step in range(iter_num):
        if step == warmup:
            torch.cuda.synchronize()
            tic = time.time()

        data = next(dataloader)
        loss = model(*data)
        model.backward(loss)
        model.step()

        if step == 0:
            log_dist('passed first iteration', ranks=[0])
        if (step + 1) % 2 == 0:
            log_dist(f'iter [{step + 1}/{iter_num}]', ranks=[0])
    torch.cuda.synchronize()
    toc = time.time()
    log_dist(f"iteration time: {(toc-tic) / (iter_num - warmup) * 1000} ms", ranks=[0])
    log_dist(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB", [0])

if __name__ == '__main__':
    train()