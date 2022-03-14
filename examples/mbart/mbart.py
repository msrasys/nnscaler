"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/mbart/mbart.py
"""

from typing import Optional
import argparse
import math
import torch

import cube
from cube.runtime.syndata import SynTextDataLoader
from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary, model_summary
from cube.profiler.timer import print_each_rank


# fairseq task
# translation_from_pretrained_bart

# fairseq criterion
# label_smoothed_cross_entropy, --label_smoothing = 0.2

class Config:

    num_embeddings = 250027

    encoder_embed_path = None
    encoder_embed_dim = 1024
    encoder_ffn_embed_dim = 4 * 1024
    encoder_layers = 12
    encoder_attention_heads = 16
    encoder_normalize_before = True
    encoder_learned_pos = True

    decoder_embed_path = None
    decoder_embed_dim = 1024
    decoder_ffn_embed_dim = 4 * 1024
    decoder_layers = 12
    decoder_attention_heads = 16
    decoder_normalize_before = True
    decoder_learned_pos = True
    cross_self_attention = False
    no_cross_attention = False

    attention_dropout = 0.0
    activation_dropout = 0.0
    dropout = 0.1

    max_target_positions = 1024
    max_source_positions = 1024
    adaptive_softmax_cutoff = None
    adaptive_softmax_dropout = 0

    share_decoder_input_output_embed = True
    share_all_embeddings = True

    decoder_output_dim = 1024  # same with decorder_embed_dim
    decoder_input_dim = 1024   # same with decorder_embed_dim

    no_scale_embedding = False # True in bart large
    layernorm_embedding = True
    activation_fn = 'gelu'
    pooler_activation_fn = 'tanh'
    pooler_dropout = 0.0


def attn_fn(query: torch.Tensor, key: torch.Tensor,
            wq: torch.Tensor, wq_bias: Optional[torch.Tensor],
            wk: torch.Tensor, wk_bias: Optional[torch.Tensor],
            wv: torch.Tensor, wv_bias: Optional[torch.Tensor],
            wout: torch.Tensor, wout_bias: Optional[torch.Tensor],
            h: int, scale: float, dropout: float, mask=True):
    """
    query, key: (L, N, E) = (seqlen, batch size, embed_dim)
    wq, wk, wv weight: [(num_head * dim_head), E]
    dropout: float
    h: int: number of heads
    """
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = wq.size(0) // num_head

    q = torch.nn.functional.linear(query, wq, wq_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(key, wk, wk_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(key, wv, wv_bias)   # L N E, (h d) E -> L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    q = q.transpose(0, 1)  # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q * scale          # (N h) L d, 1 -> (N h) L d
    k = k.transpose(1, 2)  # (N h) L d -> (N h) d L
    attn = torch.bmm(q, k) # (N h) L d, (N h) d L -> (N h) L L

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        mask = torch.tril(ones)
        mask = mask.view(N, 1, L, L)
        mask = (mask < 0.5)
        attn = attn.masked_fill_(mask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout, True, False) # (N h) L L -> (N h) L L
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, wout, wout_bias) # L N (h d), E E  -> L N E
    return output


class MultiheadAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0, bias=True):
        super().__init__()
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(embed_dim, self.kdim))
        if bias:
            self.k_bias = torch.nn.Parameter(torch.empty(embed_dim))
        else:
            self.k_bias = None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(embed_dim, self.vdim))
        if bias:
            self.v_bias = torch.nn.Parameter(torch.empty(embed_dim))
        else:
            self.v_bias = None
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(embed_dim, embed_dim))
        if bias:
            self.q_bias = torch.nn.Parameter(torch.empty(embed_dim))
        else:
            self.q_bias = None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, embed_dim))
        if bias:
            self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))
        else:
            self.out_bias = None

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        return attn_fn(query, key, 
               self.q_proj, self.q_bias,
               self.k_proj, self.k_bias,
               self.v_proj, self.v_bias,
               self.out_proj, self.out_bias,
               self.num_heads, self.scaling, self.dropout_p)

    def forward_encoder_decoder_attn(self, query: torch.Tensor, key: torch.Tensor):
        # tgt_len, bsz, embed_dim = query.size()
        # q = torch.nn.functional.linear(query, self.q_proj, self.q_bias)
        # k = torch.nn.functional.linear(key, self.k_proj, self.k_bias)
        # v = torch.nn.functional.linear(key, self.v_proj, self.v_bias)
        # q = q * self.scaling
        # q = q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = torch.bmm(q, k.transpose(1, 2))
        # # TODO: here needs a mask
        # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        # attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
        # attn = torch.bmm(attn_probs, v)
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        # attn = torch.nn.functional.linear(attn, self.out_proj, self.out_bias)
        return attn_fn(query, key, 
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)

    def forward_self_attn(self, query):
        return attn_fn(query, query,
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)


class EncoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()
        self.self_attn = MultiheadAttention(cfg.encoder_embed_dim, cfg.encoder_attention_heads, cfg.attention_dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)
        self.fc1 = torch.nn.Linear(cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim)
        self.fc2 = torch.nn.Linear(cfg.encoder_ffn_embed_dim, cfg.encoder_embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)

    def forward(self, x):  # , encoder_padding_mask: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = x + residual
        return x


class Encoder(torch.nn.Module):

    def __init__(self, cfg: Config, embed_tokens: torch.nn.Module):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.max_source_positions = cfg.max_source_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        self.embed_positions = torch.nn.Embedding(cfg.max_source_positions, cfg.encoder_embed_dim)
        self.layernorm_embedding = torch.nn.LayerNorm(cfg.encoder_embed_dim)
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [EncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        # normalize before
        self.layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)

    def forward(self, src_tokens: torch.Tensor):
        token_embedding = self.embed_tokens(src_tokens)
        embed = self.embed_scale * token_embedding

        x = embed + self.embed_positions.weight # self.embed_positions(src_tokens)
        x = self.layernorm_embedding(x)
        x = self.dropout(x)
        
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x) # encoder_padding_mask if has_pads else None)
        x = self.layer_norm(x)
        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.self_attn = MultiheadAttention(cfg.decoder_embed_dim, cfg.decoder_attention_heads, cfg.attention_dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)
        # encoder atten
        self.encoder_attn = MultiheadAttention(cfg.encoder_embed_dim, cfg.decoder_attention_heads, cfg.attention_dropout)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)
        # self.encoder_layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        self.fc1 = torch.nn.Linear(cfg.decoder_embed_dim, cfg.decoder_ffn_embed_dim)
        self.fc2 = torch.nn.Linear(cfg.decoder_ffn_embed_dim, cfg.decoder_embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)

    def forward(self, x, encoder_out):  # encoder_padding_mask):
        residual = x
        # normalize before
        x = self.self_attn_layer_norm(x)

        # self attention
        x = self.self_attn(x, x)
        x = self.dropout(x)
        x = residual + x
    
        # encoder attn
        residual = x
        # normalize before
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_out)
        x = self.dropout(x)
        x = x + residual

        residual = x
        # normalize before
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        return x


class Decoder(torch.nn.Module):

    def __init__(self, cfg: Config, embed_tokens: torch.nn.Module):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(cfg.decoder_embed_dim)
        self.embed_positions = torch.nn.Embedding(cfg.max_source_positions, cfg.decoder_embed_dim)
        self.layernorm_embedding = torch.nn.LayerNorm(cfg.decoder_embed_dim)
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [DecoderLayer(cfg) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)

    def forward(self, prev_output_tokens: torch.Tensor, enc: torch.Tensor):
        positions = self.embed_positions.weight  # self.embed_positions(prev_output_tokens)
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + positions
        x = self.layernorm_embedding(x)
        x = self.dropout(x)
        # B T C -> T B C
        x = x.transpose(0, 1)
        # decoder layers
        for layer in self.layers:
            x = layer(x, enc)
        x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # B T C, N, C -> B T N
        x = torch.nn.functional.linear(x, self.embed_tokens.weight)
        return x

# label_smoothed_cross_entropy
def criterion(output: torch.Tensor, prev_output_tokens: torch.Tensor, label_smoothing: float = 0.2):
    target = prev_output_tokens[:, 1:]
    # fairseq.criterions.label_smoothed_cross_entory
    # model.get_normalized_probs
    lprobs = torch.nn.functional.softmax(output, dim=-1)
    # fairseq.criterions.label_smoothed_nll_loss
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = label_smoothing / (lprobs.size(-1) - 1)
    loss = (1.0 - label_smoothing - eps_i) * nll_loss + eps_i * smooth_loss
    return loss


class mBART(torch.nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        emb = torch.nn.Embedding(cfg.num_embeddings, cfg.encoder_embed_dim)
        self.encoder: Encoder = Encoder(cfg, emb)
        self.decoder: Decoder = Decoder(cfg, emb)
    
    def forward(self, src_tokens, prev_output_tokens):
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        loss = criterion(decoder_out, prev_output_tokens)
        return loss


def train():
    bs = 1

    cfg = Config()
    model = mBART(cfg).cuda()

    print_each_rank('model weight consumpition:')
    memory_summary()

    dataloader = SynTextDataLoader(
        shapes=(
            [bs, cfg.max_source_positions],
            [bs, cfg.max_target_positions]
        ),
        dtypes=(torch.int64, torch.int64),
        batch_dims=(0,0,)
    )

    def train_iter(model, dataloader):
        # model.eval()
        src_tokens, prev_output_tokens = next(dataloader)
        model_summary(model, (src_tokens, prev_output_tokens))
        # loss = model(src_tokens, prev_output_tokens)
        # loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    CudaTimer(enable=False).warmup()
    iter_num = 1
    for step in range(iter_num):
        if step >= 40:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        break
        optimizer.step()
        optimizer.zero_grad()
        if step >= 40:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    # print_each_rank('e2e time (ms) per iteration: {} ms'.format(
    #       CudaTimer().duration(iter_num-40, field_name='e2e')))
    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()