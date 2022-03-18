import torch
import math

from examples.nlp.blocks.encoder import EncoderLayer

import cube


class Config:

    num_embeddings = 50304
    seqlen = 512

    # 1.7B model
    embed_dim = 2304
    layers = 24
    attention_heads = 24

    # 3.6B model
    # embed_dim = 3072
    # layers = 32
    # attention_heads = 32

    # 7.5B model
    # embed_dim = 4096
    # layers = 32
    # attention_heads = 36

    ffn_embed_dim = embed_dim * 4
    dropout = 0.0
    attn_dropout = 0.0
    activation_dropout = 0.0


class GPT(torch.nn.Module):

    def __init__(self):
        super().__init__()
        cfg = Config()

        self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.embed_dim)
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.embed_dim, cfg.attention_heads, cfg.ffn_embed_dim,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):

        embed = self.embed(input_ids)
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        logits = torch.nn.functional.linear(enc, self.embed.weight)
        # simplified
        loss = torch.sum(logits)
        return loss


class GPTDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.cfg = Config()
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
        ).repeat(self.bs)
        return (input_ids, position_ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]