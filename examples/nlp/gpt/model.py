
import torch

from examples.nlp.blocks.encoder import EncoderLayer, EncoderInferLayer
import cube


class Config:

    num_embeddings = 50432
    seqlen = 1024

    # toy model
    embed_dim = 1024
    layers = 4 # 96
    attention_heads = 16

    # # 1 layer of 175B model
    # embed_dim = 12288
    # layers = 1 # 96
    # attention_heads = 96
    #
    # # 350 M model (Medium)*
    # embed_dim = 1024
    # layers = 24
    # attention_heads = 16

    # 1.3 B model
    # embed_dim = 2048
    # layers = 24
    # attention_heads = 32

    # 2.6 B model
    # embed_dim = 2560
    # layers = 32
    # attention_heads = 32

    # 6.7 B model
    # embed_dim = 4096
    # layers = 1
    # attention_heads = 32

    # 15 B model
    # embed_dim = 5120
    # layers = 48
    # attention_heads = 36

    # 39 B model
    # embed_dim = 8192
    # layers = 48
    # attention_heads = 64

    # 175 B model*
    # embed_dim = 12288
    # layers = 96
    # attention_heads = 96

    attn_hidden_dim = embed_dim
    ffn_hidden_dim  = embed_dim * 4
    dropout = 0.2
    attn_dropout = 0.2
    activation_dropout = 0.2


class GPT(torch.nn.Module):

    def __init__(self):
        super().__init__()
        cfg = Config()

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

    def __init__(self, batch_size: int = 1):
        super().__init__()
        cfg = Config()

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
        ).repeat(self.bs).view(self.bs, -1)
        return (input_ids, position_ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]

class GPTInferDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.cfg = Config()
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