import torch
import math
from dataclasses import dataclass

from examples.nlp.blocks.transformer import TransformerLayer

import cube
from cube.runtime.utils import create_dummy_dataloader


@dataclass
class Config:

    hidden: int = 1024
    heads: int = 16
    layers: int = 4  # for encoder and decoder layers separately
    seqlen: int = 2048
    ffn_hidden_dim: int = 4096
    vocab: int = 2500

    attention_dropout: float = 0.2
    dropout: float = 0.2
    activation_dropout: float = 0.2

    pad_token_id: int = 1
    eos_token_id: int = 1
    num_classes: int = 3


class PositionalEmbedding(torch.nn.Embedding):

    def __init__(self, vocab: int, embedding_dim: int):
        self.offset = 2
        super().__init__(vocab + self.offset, embedding_dim)

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

    # def forward(self, dec: torch.Tensor, labels):
    def forward(self, dec: torch.Tensor):
        # sentence_represent = dec[eos_mask,:].view(dec.size(0), -1, hidden_states.size(-1))[:,-1,:]
        dec = torch.select(dec, dim=1, index=-1)
        # dec = dec[:,-1,:]
        sentence_represent = dec
        hidden_states = self.dropout(sentence_represent)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        # loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        loss = logits.sum()
        return loss


class MBartForSentenceClassification(torch.nn.Module):

    def __init__(self, batch_size: int, cfg: Config):
        super().__init__()
        self.vocab_size = cfg.vocab
        # embedding
        self.vocab = torch.nn.Parameter(torch.empty(
            cfg.vocab, cfg.hidden))
        # encoder embedding
        self.embed_offset = 2
        self.encoder_position = torch.nn.Parameter(torch.empty(
            cfg.seqlen, cfg.hidden))
        self.embed_scale_encoder = math.sqrt(cfg.hidden)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.hidden)

        # encoder layers
        self.encoders = torch.nn.ModuleList(
            [TransformerLayer(
                cfg.hidden, cfg.heads,
                cfg.hidden, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout,
                use_cross_attention=False,
            ) for _ in range(cfg.layers)]
        )
        self.layer_norm_encoder = torch.nn.LayerNorm(cfg.hidden)

        # decoder embedding
        self.decoder_position = torch.nn.Parameter(torch.empty(
            cfg.seqlen, cfg.hidden))
        self.embed_scale_decoder = math.sqrt(cfg.hidden)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.hidden)

        # decoder layers
        self.decoders = torch.nn.ModuleList(
            [TransformerLayer(
                cfg.hidden, cfg.heads,
                cfg.hidden, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout,
                use_cross_attention=True,
            ) for _ in range(cfg.layers)]
        )
        self.layer_norm_decoder = torch.nn.LayerNorm(cfg.hidden)
        self.head = MBartClassificationHead(cfg.hidden, 1024, cfg.num_classes, 0.0)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor):
        """
        The forward is only for benchmark performance,
        the original input of input_ids, decoder_input_ids and labels are
        simplied by using only ine input_ids.

        The loss computation is also simplified by using sum.
        """
        # encoder embedding
        cube.runtime.function.anchor('encoder embedding')
        enc_emb = torch.nn.functional.embedding(input_ids, self.vocab)
        enc_emb = enc_emb * self.embed_scale_encoder
        enc_emb = enc_emb + self.encoder_position
        enc_emb = self.layernorm_embedding_encoder(enc_emb)
        enc_emb = torch.nn.functional.dropout(enc_emb, p=0.1)
        enc = enc_emb.transpose(0, 1)

        # encoder layers
        for layer in self.encoders:
            cube.runtime.function.anchor('encoder layer')
            enc = layer(enc)
        enc = self.layer_norm_encoder(enc)
        
        # decoder embedding
        cube.runtime.function.anchor('decoder embedding')
        dec_emb = torch.nn.functional.embedding(decoder_input_ids, self.vocab)
        dec_emb = dec_emb * self.embed_scale_decoder
        dec_emb = dec_emb + self.decoder_position
        dec_emb = self.layernorm_embedding_decoder(dec_emb)
        dec_emb = torch.nn.functional.dropout(dec_emb, p=0.1)
        dec = dec_emb.transpose(0, 1)

        # decoder layers
        for layer in self.decoders:
            cube.runtime.function.anchor('decoder layer')            
            dec = layer(dec, enc)
        
        dec = self.layer_norm_decoder(dec)
        dec = dec.transpose(0, 1)
        
        # head
        loss = self.head(dec)
        return loss


def get_mbart_dummy_dataloader(batch_size: int, config: Config):

    input_ids = torch.randint(
        0, config.vocab,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    decoder_input_ids = torch.randint(
        0, config.vocab,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    labels = torch.randint(
        0, config.num_classes,
        size=(), # scalar
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    return create_dummy_dataloader((input_ids, decoder_input_ids,), batch_size)
