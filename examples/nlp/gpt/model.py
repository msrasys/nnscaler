import torch

from examples.nlp.blocks.encoder import EncoderLayer, EncoderLayerFineGrained, EncoderInferLayer
import cube
from dataclasses import dataclass

@dataclass
class Config:
    embed_dim: int = 1024
    layers: int = 8
    attention_heads: int = 16
    attn_hidden_dim: int = 1024
    ffn_hidden_dim: int = 4096
    num_embeddings: int = 50432
    seqlen: int = 1024
    dropout: float = 0.2
    attn_dropout: float = 0.2
    activation_dropout: float = 0.2


def build_gpt_config(name: str) -> Config:
    if name == '350M':
        embed_dim, layers, attention_heads = 1024, 24, 16
    elif name == '760M':
        embed_dim, layers, attention_heads = 1536, 24, 16
    elif name == '1.3B':
        embed_dim, layers, attention_heads = 2048, 24, 32
    elif name == '2.6B':
        embed_dim, layers, attention_heads = 2560, 32, 32
    elif name == '6.7B':
        embed_dim, layers, attention_heads = 4096, 32, 32
    elif name == '13B':
        embed_dim, layers, attention_heads = 5120, 48, 40 
    elif name == '39B':
        embed_dim, layers, attention_heads = 8192, 48, 64
    elif name == '175B':
        embed_dim, layers, attention_heads = 12288, 96, 96
    else:
        assert False, f'unrecognized name: {name}'
    return Config(embed_dim, layers, attention_heads, embed_dim, 4 * embed_dim)

class GPTFineGrained(torch.nn.Module):

    def __init__(self, cfg=Config()):
        super().__init__()

        self.embedw = torch.nn.Parameter(torch.empty(cfg.num_embeddings, cfg.embed_dim))
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [EncoderLayerFineGrained(
                cfg.embed_dim, cfg.attention_heads,
                cfg.attn_hidden_dim, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            cube.runtime.function.anchor('transformer start')
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        logits = torch.nn.functional.linear(enc, self.embedw)
        # simplified
        loss = torch.sum(logits)
        return loss

class GPT(torch.nn.Module):

    def __init__(self, cfg=Config()):
        super().__init__()

        # self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.embed_dim)
        self.embedw = torch.nn.Parameter(torch.empty(cfg.num_embeddings, cfg.embed_dim))
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.embed_dim, cfg.attention_heads,
                cfg.attn_hidden_dim, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):

        # embed = self.embed(input_ids)
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            cube.runtime.function.anchor('transformer start')
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        # logits = torch.nn.functional.linear(enc, self.embed.weight)
        logits = torch.nn.functional.linear(enc, self.embedw)
        # simplified
        loss = torch.sum(logits)
        return loss


class GPTInfer(torch.nn.Module):

    def __init__(self, batch_size: int = 1, cfg: Config = Config()):
        super().__init__()
        # self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.embed_dim)
        self.embedw = torch.nn.Parameter(torch.rand(cfg.num_embeddings, cfg.embed_dim) / 128)
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [EncoderInferLayer(
                cfg.embed_dim, cfg.attention_heads,
                cfg.attn_hidden_dim, cfg.ffn_hidden_dim, cfg.seqlen,
                batch_size,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)


    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):

        # embed = self.embed(input_ids)
        cube.runtime.function.anchor('first_embed')
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            cube.runtime.function.anchor('transformer start')
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        # logits = torch.nn.functional.linear(enc, self.embed.weight)
        cube.runtime.function.anchor('last_embed')
        logits = torch.nn.functional.linear(enc, self.embedw)
        # simplified
        loss = torch.sum(logits)
        return loss


class GPTDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int, cfg: Config = Config()):

        self.bs = batch_size
        self.cfg = cfg
        super().__init__(
            shapes=([batch_size, self.cfg.seqlen],
                    [batch_size, self.cfg.seqlen],
            ),
            dtypes=(torch.int64, torch.int64),
            batch_dims=(0, 0)
        )
        self.samples = [self.random_sample()]

    def random_sample(self):
        input_ids = torch.randint(
            0, self.cfg.num_embeddings,
            size=(self.bs, self.cfg.seqlen),
            dtype=torch.int64, device=torch.cuda.current_device()
        )
        position_ids = torch.arange(
            0, self.cfg.seqlen, dtype=torch.int64, device=torch.cuda.current_device()
        ).repeat(self.bs).view(self.bs, -1)
        return (input_ids, position_ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]


class GPTInferDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int, cfg: Config = Config()):

        self.bs = batch_size
        self.cfg = cfg
        super().__init__(
            shapes=([batch_size, 1],
                    [batch_size, 1],
            ),
            dtypes=(torch.int64, torch.int64),
            batch_dims=(0, 0)
        )
        self.samples = [self.random_sample()]

    def random_sample(self):
        input_ids = torch.randint(
            0, self.cfg.num_embeddings,
            size=(self.bs, 1),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        position_ids = torch.arange(
            0, 1, dtype=torch.int64,
            device=torch.cuda.current_device()
        ).repeat(self.bs).view(self.bs, -1)
        return (input_ids, position_ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]