#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CausalSelfAttention is copied from https://github.com/karpathy/nanoGPT/blob/master/model.py
# with minor modifications.
# See the original license in the file https://github.com/karpathy/nanoGPT/blob/master/LICENSE

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict

from streaming import MDSWriter, StreamingDataset, StreamingDataLoader

import nnscaler
from nnscaler.cli.trainer_args import TrainerArgs
from tests.parallel_module.test_end2end import MLP
from tests.utils import init_random as init_random_fn



class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SimpleTransformerModel(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, nlayers: int, vocab_size: int):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        for _ in range(nlayers):
            self.layers.append(CausalSelfAttention(n_embd, n_head, dropout))

    def forward(self, data):
        x = data['input']
        target = data['target']
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        return loss


def csa_forward_args_gen_fn(trainer_args: TrainerArgs):
    seq_len = 128 # dynamicness is controlled by trainer_args.vars['dynamic_dims']

    return {
        'x': torch.randn(1, seq_len, trainer_args.model.args['n_embd']),
    }


def post_csa_forward_args_gen_fn(trainer_args: TrainerArgs, args):
    dynamic_dims = trainer_args.get_resolved_var('dynamic_dims', default=[])
    nnscaler.mark_dynamic(args['x'], dynamic_dims)
    return args


def transformer_dummy_sample_gen_fn(trainer_args: TrainerArgs):
    seq_len = 128 # dynamicness is controlled by trainer_args.vars['dynamic_dims']
    dynamic_dims = trainer_args.get_resolved_var('dynamic_dims', default=[])
    return {
        'input': nnscaler.mark_dynamic(torch.randn(1, seq_len, trainer_args.model.args['n_embd']), dynamic_dims),
        'target': nnscaler.mark_dynamic(torch.randint(0, trainer_args.model.args['vocab_size'], (1, seq_len)), dynamic_dims),
    }


class MixModuleMLP(nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        if init_random:
            init_random_fn()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


class MixModuleMLP2(MixModuleMLP):
    pass


class MixModuleMLP3(MixModuleMLP):
    pass


class MixModuleMLP4(MixModuleMLP):
    pass


class MixModuleMLPWithLoss(nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        self.mlp = MixModuleMLP(dim, nlayers, init_random=init_random)
        self.loss_fn = nn.BCELoss()

    def forward(self, input, target):
        x = self.mlp(input)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, target)
        return loss


class MixedModule(torch.nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        self.mlp0 = MixModuleMLP(dim, nlayers, init_random=init_random)
        self.mlp1 = MixModuleMLP2(dim, nlayers, init_random=init_random)
        self.mlp2 = MixModuleMLP3(dim, nlayers, init_random=init_random)
        self.mlploss = MixModuleMLPWithLoss(dim, nlayers, init_random=init_random)

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        target = data['target']
        x = self.mlp0(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return self.mlploss(x, target)


def forward_args_gen_fn(trainer_args: TrainerArgs):
    return {
        'input':
            torch.randn(trainer_args.dataset.train_args['size'], trainer_args.dataset.train_args['dim']),
        'target':
            torch.rand(trainer_args.dataset.train_args['size'], trainer_args.dataset.train_args['dim']),
    }


class SimpleDataset(Dataset):
    def __init__(self, dim: int, size: int = 100):
        torch.manual_seed(0)
        self.data = torch.randn(size, dim)
        self.target = torch.rand(size, dim)

    def __getitem__(self, idx: int):
        return {
            'data': self.data[idx],
            'target': self.target[idx]
        }

    def __len__(self):
        return len(self.data)


class SimpleIterDataset(StreamingDataset):
    def __init__(self, split, *args, **kwargs):
        name = Path(__file__).parent / f'streaming_data/simple_dataset_{split}'
        super().__init__(local=name, *args, **kwargs)
        # the data files are created using:
        # dataset = SimpleDataset(dim, size)
        # with MDSWriter(
        #     columns={'data' : 'ndarray', 'target': 'ndarray'},
        #     out=name, compression='zstd'
        # ) as out:
        #     for item in dataset:
        #         out.write({
        #             'data': item['data'].numpy(),
        #             'target': item['target'].numpy()
        #         })

    def __iter__(self):
        for item in super().__iter__():
            yield {
                'data': torch.tensor(item['data']),
                'target': torch.tensor(item['target'])
            }
