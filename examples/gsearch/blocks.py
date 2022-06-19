import torch
import cube
import warnings


@cube.graph.parser.register('L N E+, (h d) E+, (h d), (h d) E+, (h d), (h d) E+, (h d) -> N h L d, N h L d, N h L d', name='attn_qkv')
def attn_qkv(query: torch.Tensor,
             q_proj: torch.Tensor, q_bias: torch.Tensor,
             k_proj: torch.Tensor, k_bias: torch.Tensor,
             v_proj: torch.Tensor, v_bias: torch.Tensor,
             h: int, scale: float):
    L, N = query.size(0), query.size(1)
    d = q_proj.size(0) // h

    q = torch.nn.functional.linear(query, q_proj, q_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(query, k_proj, k_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(query, v_proj, v_bias)   # L N E, (h d) E -> L N (h d)
    q = q.contiguous().view(L, (N * h), d) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * h), d) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * h), d) # L N (h d) -> L (N h) d
    q = q.transpose(0, 1)  # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q * scale          # (N h) L d, 1 -> (N h) L d
    q = q.view(N, h, L, d)
    k = k.view(N, h, L, d)
    v = v.view(N, h, L, d)
    return q, k, v


@cube.graph.parser.register('N h L d+, N h L d+ -> N h L L', name='attn_score')
def attn_score(q: torch.Tensor, k: torch.Tensor, h: int, mask: bool = True):
    N, num_head, L, d = q.size()
    assert num_head == h
    q = q.view(-1, L, d)
    k = k.view(-1, L, d)
    k = k.transpose(1, 2)
    attn = torch.bmm(q, k)
    attn = attn.view(N, h, L, L)
    # attention mask
    if mask:
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
    return attn


@cube.graph.parser.register('N h L K^ -> N h L K^', name='attn_softmax')
def attn_softmax(attn: torch.Tensor):
    N, h, L, L = attn.size()
    attn = attn.view((N * h), L, L)
    attn = torch.nn.functional.softmax(attn, dim=-1)
    return attn.view(N, h, L, L)


@cube.graph.parser.register('N h L L -> N h L L', name='attn_dropout')
def attn_dropout(attn: torch.Tensor, dropout_p: float):
    return torch.nn.functional.dropout(attn, dropout_p, True, False)


@cube.graph.parser.register('N h L K+, N h K+ d -> L N (h d)', name='attn_context')
def attn_context(attn: torch.Tensor, v: torch.Tensor):
    N, h, L, d = v.size()
    attn = attn.view((N * h), L, L)
    v = v.view((N * h), L, d)
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, h * d)  # (N h) L d -> L N (h d)
    return output


@cube.graph.parser.register('L N hd+, E hd+, E -> L N E', name='attn_dense_out')
def attn_dense_out(context: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return torch.nn.functional.linear(context, weight, bias)


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0, bias=True):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(inner_dim)) if bias else None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim)) if bias else None

    def forward(self, query: torch.Tensor):
        # QKV
        q, k, v = attn_qkv(
            query,
            self.q_proj, self.q_bias,
            self.k_proj, self.k_bias,
            self.v_proj, self.v_bias,
            self.num_heads, self.scaling
        )
        # AttentionScore
        attn = attn_score(q, k, self.num_heads, mask=True)
        # softmax
        attn = attn_softmax(attn)
        # dropout
        attn = attn_dropout(attn, self.dropout_p) # N h L L -> N h L L
        # attn = torch.nn.functional.dropout(attn, self.dropout_p, True, False) # N h L L -> N h L L
        # context
        context = attn_context(attn, v)
        # DenseOutput
        # output = torch.nn.functional.linear(context, self.out_proj, self.out_bias) # L N (h d), E E  -> L N E
        output = attn_dense_out(context, self.out_proj, self.out_bias) # L N (h d), E E  -> L N E
        return output


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.linear(x, self.proj1, self.proj1_bias)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.linear(x, self.proj2, self.proj2_bias)
        x = torch.nn.functional.dropout(x, self.dropout, True, False)
        return x


class EncoderLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
        warnings.warn('residual is disabled in encoder block')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        # x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        # x = x + residual
        return x
