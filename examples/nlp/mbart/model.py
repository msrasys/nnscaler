import torch
import math

from examples.nlp.blocks.encoder import EncoderLayer
from examples.nlp.blocks.decoder import DecoderLayer

import cube


class Config:

    # source and target
    max_source_positions = 1024
    max_target_positions = 1024

    num_embeddings = 250027

    encoder_embed_dim = 1024
    encoder_ffn_embed_dim = 4 * 1024
    encoder_layers = 12
    encoder_attention_heads = 16

    decoder_embed_dim = 1024
    decoder_ffn_embed_dim = 4 * 1024
    decoder_layers = 12
    decoder_attention_heads = 16

    attention_dropout = 0.0
    dropout = 0.1
    activation_dropout = 0.0

    pad_token_id = 1
    eos_token_id = 2

    # classification task
    num_classes = 3


class PositionalEmbedding(torch.nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, seq_len: int):
        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=torch.cuda.current_device()
        )
        return super().forward(positions + self.offset)


class MBartClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
    
        self.num_classes = num_classes
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim, num_classes)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, dec: torch.Tensor, labels):
        # sentence_represent = dec[eos_mask,:].view(dec.size(0), -1, hidden_states.size(-1))[:,-1,:]
        dec = dec[:,-1,:]
        sentence_represent = dec
        hidden_states = self.dropout(sentence_represent)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return loss


class MBartForSentenceClassification(torch.nn.Module):

    def __init__(self):
        super().__init__()
        cfg = Config()
        # embedding
        self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.encoder_embed_dim)
        
        # encoder embedding
        self.encoder_position = PositionalEmbedding(cfg.max_source_positions, cfg.encoder_embed_dim)
        self.embed_scale_encoder = math.sqrt(cfg.encoder_embed_dim)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        # encoder layers
        self.encoders = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.decoder_embed_dim, cfg.decoder_attention_heads, cfg.decoder_ffn_embed_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout
            ) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        # decoder embedding
        self.decoder_position = PositionalEmbedding(cfg.max_target_positions, cfg.decoder_embed_dim)
        self.embed_scale_decoder = math.sqrt(cfg.decoder_embed_dim)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.decoder_embed_dim)

        # decoder layers
        self.decoders = torch.nn.ModuleList(
            [DecoderLayer(
                cfg.decoder_embed_dim, cfg.decoder_attention_heads, cfg.decoder_ffn_embed_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout
            ) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm_decoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        self.head = MBartClassificationHead(cfg.decoder_embed_dim, 1024, cfg.num_classes, 0.0)
    
    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor, labels: torch.Tensor):

        # encoder embedding
        enc_emb = self.embed(input_ids)
        enc_emb = enc_emb * self.embed_scale_encoder
        enc_emb = enc_emb + self.encoder_position(input_ids.size(1))
        enc_emb = self.layernorm_embedding_encoder(enc_emb)
        enc_emb = torch.nn.functional.dropout(enc_emb, p=0.0)
        enc = enc_emb.transpose(0, 1)

        # encoder layers
        for layer in self.encoders:
            enc = layer(enc)
        enc = self.layer_norm_encoder(enc)
        
        # decoder embedding
        dec_emb = self.embed(decoder_input_ids)
        dec_emb = dec_emb * self.embed_scale_decoder
        dec_emb = dec_emb + self.decoder_position(decoder_input_ids.size(1))
        dec_emb = self.layernorm_embedding_decoder(dec_emb)
        dec_emb = torch.nn.functional.dropout(dec_emb, p=0.0)
        dec = dec_emb.transpose(0, 1)

        # decoder layers
        for layer in self.decoders:
            dec = layer(dec, enc)
        dec = self.layer_norm_decoder(dec)
        dec = dec.transpose(0, 1)
        
        # head
        loss = self.head(dec, labels)
        return loss


class MBartDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.cfg = Config()
        super().__init__(
            shapes=([batch_size, self.cfg.max_source_positions,],
                    [batch_size, self.cfg.max_target_positions],
                    [batch_size]
            ),
            dtypes=(torch.int64, torch.int64, torch.int64),
            batch_dims=(0, 0, 0)
        )
        self.samples = [self.random_sample()]
        
    def random_sample(self):
        input_ids = torch.randint(
            0, self.cfg.num_embeddings,
            size=(self.bs, self.cfg.max_source_positions),
            dtype=torch.int64, device=torch.cuda.current_device()
        )
        decoder_input_ids = MBartDataLoader.shift_tokens_right(input_ids, self.cfg.pad_token_id)
        labels = torch.randint(
            0, self.cfg.num_classes,
            size=(self.bs,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        return (input_ids, decoder_input_ids, labels)
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
        prev_output_tokens = input_ids.clone()
        prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
        index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
        prev_output_tokens[:, 0] = decoder_start_tokens
        return prev_output_tokens
    
