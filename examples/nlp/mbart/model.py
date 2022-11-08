import torch
import math

from examples.nlp.blocks.encoder import EncoderLayer
from examples.nlp.blocks.decoder import DecoderLayer

import cube


@cube.graph.parser.register('* -> *, *', name='multi2ref')
def multi2ref(tensor: torch.Tensor):
    return tensor, tensor


class Config:

    TBD = None # to be decided
    # source and target
    num_embeddings = 2500
    hidden = 1024
    heads = 16
    layers = 4
    seqlen = 2048

    max_source_positions = None
    max_target_positions = None

    encoder_embed_dim = TBD
    encoder_ffn_embed_dim = TBD
    encoder_layers = TBD
    encoder_attention_heads = TBD

    decoder_embed_dim = TBD
    decoder_ffn_embed_dim = TBD
    decoder_layers = TBD
    decoder_attention_heads = TBD

    attention_dropout = TBD
    dropout = TBD
    activation_dropout = TBD

    pad_token_id = TBD
    eos_token_id = TBD

    # classification task
    num_classes = TBD

    def __init__(self) -> None:

        Config.max_source_positions = Config.seqlen
        Config.max_target_positions = Config.seqlen

        Config.encoder_embed_dim = Config.hidden
        Config.encoder_ffn_embed_dim = 4 * Config.hidden
        Config.encoder_layers = Config.layers
        Config.encoder_attention_heads = Config.heads

        Config.decoder_embed_dim = Config.hidden
        Config.decoder_ffn_embed_dim = 4 * Config.hidden
        Config.decoder_layers = Config.layers
        Config.decoder_attention_heads = Config.heads

        Config.attention_dropout = 0.1
        Config.dropout = 0.1
        Config.activation_dropout = 0.1

        Config.pad_token_id = 1
        Config.eos_token_id = 2

        Config.num_classes = 3

    def __repr__(self) -> str:
        return f'Config(num_embeddings={Config.num_embeddings}, hidden={Config.hidden}, heads={Config.heads}, layers={Config.layers}, seqlen={Config.seqlen})'


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

    def __init__(self, batch_size: int):
        super().__init__()
        cfg = Config()
        self.vocab_size = cfg.num_embeddings
        print("Model Arch:", cfg)
        # embedding
        self.vocab = torch.nn.Parameter(torch.empty(
            cfg.num_embeddings, cfg.encoder_embed_dim))
        # encoder embedding
        self.embed_offset = 2
        self.encoder_position = torch.nn.Parameter(torch.empty(
            cfg.max_source_positions, cfg.encoder_embed_dim))
        self.embed_scale_encoder = math.sqrt(cfg.encoder_embed_dim)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        # encoder layers
        self.encoders = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.encoder_embed_dim, cfg.encoder_attention_heads,
                cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout
            ) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        # decoder embedding
        self.decoder_position = torch.nn.Parameter(torch.empty(
            cfg.max_target_positions, cfg.decoder_embed_dim))
        self.embed_scale_decoder = math.sqrt(cfg.decoder_embed_dim)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.decoder_embed_dim)

        # decoder layers
        self.decoders = torch.nn.ModuleList(
            [DecoderLayer(
                cfg.decoder_embed_dim, cfg.decoder_attention_heads,
                cfg.decoder_embed_dim, cfg.decoder_ffn_embed_dim,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout
            ) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm_decoder = torch.nn.LayerNorm(cfg.decoder_embed_dim)
        self.head = MBartClassificationHead(cfg.decoder_embed_dim, 1024, cfg.num_classes, 0.0)

        # FIXME: cube now is not safe for multiple
        # tensor transmissions between stages.
        decoder_input_ids = torch.randint(
            0, self.vocab_size, (batch_size, cfg.seqlen), dtype=torch.int64, device=torch.device('cpu'),
        )
        self.register_buffer('decoder_input_ids', decoder_input_ids)


    def forward(self, input_ids: torch.Tensor):
        """
        The forward is only for benchmark performance,
        the original input of input_ids, decoder_input_ids and labels are
        simplied by using only ine input_ids.

        The loss computation is also simplified by using sum.
        """
        # decoder_input_ids = torch.clone(input_ids)
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
        dec_emb = torch.nn.functional.embedding(self.decoder_input_ids, self.vocab)
        dec_emb = dec_emb * self.embed_scale_decoder
        dec_emb = dec_emb + self.decoder_position
        dec_emb = self.layernorm_embedding_decoder(dec_emb)
        dec_emb = torch.nn.functional.dropout(dec_emb, p=0.1)
        dec = dec_emb.transpose(0, 1)

        # FIXME: need to cat and chunk because cube now is not safe
        # for multiple tensor transformation between stages.
        encdec = torch.cat((enc, dec), dim=-1)

        # decoder layers
        for layer in self.decoders:
            cube.runtime.function.anchor('decoder layer')
            enc, dec = torch.chunk(encdec, 2, dim=-1)
            
            enc, next_enc = multi2ref(enc)
            
            dec = layer(dec, enc)
            encdec = torch.cat((next_enc, dec), dim=-1)
        
        enc, dec = torch.chunk(encdec, 2, dim=-1)
        dec = self.layer_norm_decoder(dec)
        dec = dec.transpose(0, 1)
        
        # head
        # loss = self.head(dec, labels)
        loss = self.head(dec)
        return loss


class MBartSyntheticDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.cfg = Config()
        super().__init__(
            shapes=([batch_size, self.cfg.max_source_positions,],),
            dtypes=(torch.int64,),
            batch_dims=(0,)
        )
        self.samples = [self.random_sample()]
        
    def random_sample(self):
        input_ids = torch.randint(
            0, self.cfg.num_embeddings,
            size=(self.bs, self.cfg.max_source_positions),
            dtype=torch.int64, device=torch.cuda.current_device()
        )
        return input_ids
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]


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
    
