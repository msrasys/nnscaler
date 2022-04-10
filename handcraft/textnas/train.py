"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/textnas/train.py \
        --bs 128 --models 12 --schedule pipe
"""

import numpy as np
import torch
import torch.nn as nn
import argparse

import cube
from cube.runtime.device import DeviceGroup
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary

from handcraft.textnas.ops import ConvBN, LinearCombine, AvgPool, MaxPool, RNN, Attention, BatchNorm, GlobalMaxPool, GlobalAvgPool
from handcraft.textnas.dataloader import SharedDataLoader
from handcraft.module.stage import layer_division


cube.init()

parser = argparse.ArgumentParser(description='textnas')
parser.add_argument('--schedule', type=str, default='replicate', choices=['replicate', 'pipe'],
                    help='scheduling algorithm. model: train model with replicated dataloader. pipe: train model with shared dataloader')
parser.add_argument('--models', type=int, default=1,
                    help='number of models to be trained in total')
parser.add_argument('--bs', type=int, default=128,
                    help='number of micro batch (default: paper setting)')
parser.add_argument('--non-uniform', action='store_true', default=False,
                    help='use non-uniform partition that Bert-allocated GPU can also have models')
args = parser.parse_args()
print(args)


_model_divisions = []
if args.schedule == 'replicate':
    num_trainers = DeviceGroup().world_size
    num_model_per_device = args.models // num_trainers
    _model_divisions = [num_model_per_device] * num_trainers
    for idx in range(args.models % num_trainers):
        _model_divisions[-1-idx] += 1
if args.schedule == 'pipe':
    num_trainers = DeviceGroup().world_size - 1
    if args.non_uniform:
        times = [160.65] + [78.79] * args.models
        _model_divisions = layer_division(times, DeviceGroup().world_size)
        _model_divisions = [end-start for start, end in _model_divisions]
        _model_divisions[0] -= 1
    else:
        num_model_per_device = args.models // num_trainers
        _model_divisions = [0] + [num_model_per_device] * num_trainers
        for idx in range(args.models % num_trainers):
            _model_divisions[-1-idx] += 1
print_each_rank(f'model number placements: {_model_divisions}')


class WrapperOp(nn.Module):
    def __init__(self, op_choice, input_args):
        super(WrapperOp, self).__init__()
        self.op_choice = op_choice
        self.input_args = input_args
        self.op = None

        def conv_shortcut(kernel_size, hidden_units, cnn_keep_prob):
            return ConvBN(kernel_size, hidden_units, hidden_units,
                          cnn_keep_prob, False, True)
        
        if op_choice == 'conv_shortcut1':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut3':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut5':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut7':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'AvgPool':
            self.op = AvgPool(3, False, True)
        elif op_choice == 'MaxPool':
            self.op = MaxPool(3, False, True)
        elif op_choice == 'RNN':
            self.op = RNN(*input_args)
        elif op_choice == 'Attention':
            self.op = Attention(*input_args)
        else:
            raise

    def forward(self, prec, mask):
        return self.op(prec, mask)


class Layer(nn.Module):
    def __init__(self, key, prev_keys, hidden_units, choose_from_k, cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask):
        super(Layer, self).__init__()

        self.n_candidates = len(prev_keys)
        if self.n_candidates:
            #===self.prec = mutables.InputChoice(choose_from=prev_keys[-choose_from_k:], n_chosen=1)
            self.prec = 1
        else:
            # first layer, skip input choice
            self.prec = None
        '''self.op = mutables.LayerChoice([
            conv_shortcut(1),
            conv_shortcut(3),
            conv_shortcut(5),
            conv_shortcut(7),
            AvgPool(3, False, True),
            MaxPool(3, False, True),
            RNN(hidden_units, lstm_keep_prob),
            Attention(hidden_units, 4, att_keep_prob, att_mask)
        ])'''
        #self.op = conv_shortcut(1)
        #self.op = Attention(hidden_units, 4, att_keep_prob, att_mask)
        #self.op = RNN(hidden_units, lstm_keep_prob)
        #self.op = WrapperOp('RNN', [hidden_units, lstm_keep_prob])
        #self.op = WrapperOp('Attention', [hidden_units, 4, att_keep_prob, att_mask])
        #self.op = WrapperOp('MaxPool', [3, False, True])
        #self.op = WrapperOp('AvgPool', [3, False, True])
        #self.op = WrapperOp('conv_shortcut7', [7, hidden_units, cnn_keep_prob])
        #self.op = WrapperOp('conv_shortcut5', [5, hidden_units, cnn_keep_prob])
        #self.op = WrapperOp('conv_shortcut3', [3, hidden_units, cnn_keep_prob])
        self.op = WrapperOp('conv_shortcut1', [1, hidden_units, cnn_keep_prob])
        if self.n_candidates:
            #===self.skipconnect = mutables.InputChoice(choose_from=prev_keys)
            self.skipconnect = 1
        else:
            self.skipconnect = None
        self.bn = BatchNorm(hidden_units, False, True)
        
        self.prec_n_candidates = choose_from_k
        self.skip_n_candidates = len(prev_keys)

    def forward(self, last_layer, prev_layers, mask):
        # pass an extra last_layer to deal with layer 0 (prev_layers is empty)
        if self.prec is None:
            prec = last_layer
        else:
            #===prec = self.prec(prev_layers[-self.prec.n_candidates:])  # skip first
            x = min(len(prev_layers), self.prec_n_candidates)
            prec = prev_layers[-x]  # skip first
        out = self.op(prec, mask)
        if self.skipconnect is not None:
            #===connection = self.skipconnect(prev_layers[-self.skipconnect.n_candidates:])
            connection = prev_layers[-self.skip_n_candidates]
            if connection is not None:
                out = out + connection
        out = self.bn(out, mask)
        return out


class Model(nn.Module):
    def __init__(self, embedding_dim=768, hidden_units=256, num_layers=24, num_classes=5, choose_from_k=5,
                 lstm_keep_prob=0.5, cnn_keep_prob=0.5, att_keep_prob=0.5, att_mask=True,
                 embed_keep_prob=0.5, final_output_keep_prob=1.0, global_pool="avg"):
        super(Model, self).__init__()

        # self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.init_conv = ConvBN(1, embedding_dim, hidden_units, cnn_keep_prob, False, True)

        self.layers = nn.ModuleList()
        candidate_keys_pool = []
        for layer_id in range(self.num_layers):
            k = "layer_{}".format(layer_id)
            self.layers.append(Layer(k, candidate_keys_pool, hidden_units, choose_from_k,
                                     cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask))
            candidate_keys_pool.append(k)

        self.linear_combine = LinearCombine(self.num_layers)
        self.linear_out = nn.Linear(self.hidden_units, self.num_classes)

        self.embed_dropout = nn.Dropout(p=1 - embed_keep_prob)
        self.output_dropout = nn.Dropout(p=1 - final_output_keep_prob)

        assert global_pool in ["max", "avg"]
        if global_pool == "max":
            self.global_pool = GlobalMaxPool()
        elif global_pool == "avg":
            self.global_pool = GlobalAvgPool()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, mask, labels):
        # sent_ids, mask = inputs
        # seq = self.embedding(sent_ids.long())
        seq = self.embed_dropout(inputs)

        seq = torch.transpose(seq, 1, 2)  # from (N, L, C) -> (N, C, L)

        x = self.init_conv(seq, mask)
        prev_layers = []

        for layer in self.layers:
            x = layer(x, prev_layers, mask)
            prev_layers.append(x)

        x = self.linear_combine(torch.stack(prev_layers))
        x = self.global_pool(x, mask)
        x = self.output_dropout(x)
        x = self.linear_out(x)
        loss = self.criterion(x, labels)
        return loss


if __name__ == '__main__':

    # initialize models
    num_model = _model_divisions[DeviceGroup().rank]
    print_each_rank(f'initializing {num_model} models...')
    models = [Model().cuda() for _ in range(num_model)]

    # initialize dataloaders
    if args.schedule == 'replicate':
        dataloader = SharedDataLoader(args.bs, replicate=True)
    elif args.schedule == 'pipe':
        dataloader = SharedDataLoader(args.bs, replicate=False)
    else:
        assert False
    dataloader = iter(dataloader)
    
    # initialize optimizer
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98)) for model in models
    ]

    CudaTimer(enable=False)
    torch.distributed.barrier()
    iter_num = 32
    for step in range(iter_num):
        if step >= 8:
            CudaTimer(enable=True).start('e2e')
        # if args.schedule == 'replicate':
        #     # retiarii baseline
        #     for _ in range(len(models)):
        #         text, masks, labels = next(dataloader)
        # else:
        #     text, masks, labels = next(dataloader)
        text, masks, labels = next(dataloader)
        for model, optimizer in zip(models, optimizers):
            CudaTimer().start('nas-model')
            loss = model(text, masks, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            CudaTimer().stop('nas-model')

        # CudaTimer().start('nas-model')
        # losses = []
        # for model in models:
        #     losses.append(model(text, masks, labels))
        # for loss in losses:
        #     loss.backward()
        # for optimizer in optimizers:
        #     optimizer.step()
        #     optimizer.zero_grad()
        # CudaTimer().stop('nas-model')
        
        if step >= 8:
            CudaTimer().stop('e2e')

        if step == 0:
            torch.distributed.barrier()
            print_each_rank('memory after optimizer:', rank_only=0)
            memory_summary()

        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-8, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-8)
    memory_summary()
