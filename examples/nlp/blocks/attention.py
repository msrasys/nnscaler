import torch
import cube


@cube.graph.parser.register('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
def self_attention(query: torch.Tensor, 
                   qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                   out_proj: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = qkv_proj.size(0) // num_head // 3

    qkv = torch.nn.functional.linear(query, qkv_proj, qkv_bias) # L N E, (h d 3) E -> L N (h d 3)
    qkv = qkv.view(L, N, num_head * dim_head, 3) # L N (h d 3) -> L N (h d) 3
    q, k, v = qkv.chunk(3, dim=-1)  # L N (3 h d) -> L N (h d), L N (h d), L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    
    # ======== replace the semantic into more efficient implementation ============
    # q = q.transpose(0, 1)  # L (N h) d -> (N h) L d
    # k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    # q = q * scale          # (N h) L d, 1 -> (N h) L d
    # k = k.transpose(1, 2)  # (N h) L d -> (N h) d L
    # attn = torch.bmm(q, k) # (N h) L d, (N h) d L -> (N h) L L
    
    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, L, L], dtype=query.dtype, device=query.device)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) L d
        k.transpose(0, 1).transpose(1, 2), # (N h) d L
        beta=0.0, alpha=scale
    )
    # ======== replace the semantic into more efficient implementation ============

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj) # L N (h d), E E  -> L N E
    return output

@cube.graph.parser.register('L^ N E^, (h d^ 3) E^, (h d^ 3) -> L^ N (h d^) 3', name='qkv_combined')
def qvk_combined(query: torch.Tensor, 
                   qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                   #out_proj: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = qkv_proj.size(0) // num_head // 3

    qkv = torch.nn.functional.linear(query, qkv_proj, qkv_bias) # L N E, (h d 3) E -> L N (h d 3)
    output = qkv.view(L, N, num_head * dim_head, 3) # L N (h d 3) -> L N (h d) 3

    return output

@cube.graph.parser.register('L^ N (h d^) 3 -> L^ N (h d^)', name='attention_mask')
def attention_mask(qkv: torch.Tensor, 
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    
    L, N = qkv.size(0), qkv.size(1)
    num_head = h
    dim_head = qkv.size(2) // num_head
    
    q, k, v = qkv.chunk(3, dim=-1)  # L N (h d) 3 -> L N (h d), L N (h d), L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d

    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, L, L], dtype=q.dtype, device=q.device)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) L d
        k.transpose(0, 1).transpose(1, 2), # (N h) d L
        beta=0.0, alpha=scale
    )
    # ======== replace the semantic into more efficient implementation ============

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    return output 
    
@cube.graph.parser.register('L^ N (h+ d^),  E^ (h+ d^) -> L^ N E^', name='attention_mask')
def lin(lin_input: torch.Tensor, 
                #    qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
        out_proj: torch.Tensor,
        h: int, scale: float, dropout_p: float, mask: bool = False):
        
    output = torch.nn.functional.linear(lin_input, out_proj) # L N (h d), E E  -> L N E
    return output


@cube.graph.parser.register('L^ N E^, L^ N E^, (h+ d) E^, (h+ d), (h+ d) E^, (h+ d), (h+ d) E^, (h+ d), E^ (h+ d) -> L^ N E^', name='cross_attention')
def cross_attention(query: torch.Tensor, key: torch.Tensor,
                    q_proj: torch.Tensor, q_bias: torch.Tensor,
                    k_proj: torch.Tensor, k_bias: torch.Tensor,
                    v_proj: torch.Tensor, v_bias: torch.Tensor,
                    out_proj: torch.Tensor,
                    h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = q_proj.size(0) // num_head

    q = torch.nn.functional.linear(query, q_proj, q_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(key, k_proj, k_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(key, v_proj, v_bias)   # L N E, (h d) E -> L N (h d)
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
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj, None) # L N (h d), E E  -> L N E
    return output


# @cube.graph.parser.register('l N E^, L^ N (h+ d), L^ N (h+ d), (h+ d) E^, (h+ d), (h+ d) E^, (h+ d), (h+ d) E^, (h+ d), E^ (h+ d) -> l N E^', name='one_attention')
@cube.graph.parser.register('l N E^, L^ N (h+ d), L^ N (h+ d), (h+ d 3) E^, (h+ d 3), E^ (h+ d) -> l N E^', name='one_attention')
def one_attention(hidden_states: torch.Tensor,
                  past_embed_key: torch.Tensor,
                  past_embed_value: torch.Tensor,
                  # q_proj: torch.Tensor, q_bias: torch.Tensor,
                  # k_proj: torch.Tensor, k_bias: torch.Tensor,
                  # v_proj: torch.Tensor, v_bias: torch.Tensor,
                  qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                  out_proj: torch.Tensor, #out_bias: torch.Tensor,
                  h: int, scale: float, dropout_p: float, is_training: bool = True, mask: bool = False):
    num_head = h
    l, N = hidden_states.size(0), hidden_states.size(1)
    # dim_head = q_proj.size(0) // num_head
    dim_head = qkv_proj.size(0) // num_head // 3

    # q = torch.nn.functional.linear(hidden_states, q_proj, q_bias)   # l N E, (h d) E -> l N (h d)
    # k = torch.nn.functional.linear(hidden_states, k_proj, k_bias)   # l N E, (h d) E -> l N (h d)
    # v = torch.nn.functional.linear(hidden_states, v_proj, v_bias)   # l N E, (h d) E -> l N (h d)
    qkv = torch.nn.functional.linear(hidden_states, qkv_proj, qkv_bias) # l N E, (h d 3) E -> l N (h d) 3
    q, k, v = qkv.chunk(3, dim=-1)

    if past_embed_key is not None and past_embed_value is not None:
        k = torch.cat((past_embed_key, k), dim=-3)
        v = torch.cat((past_embed_value, v), dim=-3)

    q_N = hidden_states.size(1)

    k_L = k.size(0)
    v_L = v.size(0)

    q = q.contiguous().view(l, (N * num_head), dim_head)    # l N (h d) -> l (N h) d
    k = k.contiguous().view(k_L, (N * num_head), dim_head)  # (L+l) N (h d) -> (L+l) (N h) d
    v = v.contiguous().view(v_L, (N * num_head), dim_head)  # (L+l) N (h d) -> (L+l) (N h) d

    # q = q.transpose(0, 1)  # l (N h) d -> (N h) l d
    # k = k.transpose(0, 1)  # (L+l) (N h) d -> (N h) (L+l) d
    # v = v.transpose(0, 1)  # (L+l) (N h) d -> (N h) (L+l) d
    # q = q * scale          # (N h) L d, 1 -> (N h) L d
    # k = k.transpose(1, 2)  # (N h) (L+l) d -> (N h) d (L+l)
    # attn = torch.bmm(q, k) # (N h) l d, (N h) d (L+l) -> (N h) l (L+l)

    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, l, k_L], dtype=hidden_states.dtype, device=hidden_states.device)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) l d
        k.transpose(0, 1).transpose(1, 2),  # (N h) d (L+l)
        beta=0.0, alpha=scale
    )

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) l (L+l) -> (N h) l (L+l)
    #no dropout in inference attn
    attn = torch.nn.functional.dropout(attn, dropout_p, is_training, False)    # (N h) l (L+l) -> (N h) l (L+l)
    v = v.transpose(0, 1)
    output = torch.bmm(attn, v) # (N h) l (L+l), (N h) (L+l) d -> (N h) l d
    output = output.transpose(0, 1).contiguous()     # (N h) l d -> l (N h) d
    output = output.view(l, N, num_head * dim_head)  # l (N h) d -> l N (h d)
    output = torch.nn.functional.linear(output, out_proj, None) # l N (h d), E E  -> l N E
    return output

class MultiHeadSelfAttentionLrw(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):

        qkv = qvk_combined(
              query, self.qkv_proj, self.qkv_bias,
              #.out_proj,
              self.num_heads, self.scaling, self.dropout_p, mask=False              
        )
        lin_input = attention_mask(
              qkv,
              self.num_heads, self.scaling, self.dropout_p, mask=False   
        )
        attn = lin(
              lin_input,
              self.out_proj,
              self.num_heads, self.scaling, self.dropout_p, mask=False   
        )
        attn = attn + self.out_bias
        return attn

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):
        attn = self_attention(
            query, self.qkv_proj, self.qkv_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=False
        )
        attn = attn + self.out_bias
        return attn


class MultiHeadCrossAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(inner_dim))
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(inner_dim))
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        attn = cross_attention(
            query, key,
            self.q_proj, self.q_bias,
            self.k_proj, self.k_bias,
            self.v_proj, self.v_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=True
        )
        attn = attn + self.out_bias
        return attn


class MultiHeadOneAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # # Q
        # self.q_proj = torch.nn.Parameter(torch.rand(inner_dim, embed_dim))
        # self.q_bias = torch.nn.Parameter(torch.rand(inner_dim))
        # # K
        # self.k_proj = torch.nn.Parameter(torch.rand(inner_dim, embed_dim))
        # self.k_bias = torch.nn.Parameter(torch.rand(inner_dim))
        # # V
        # self.v_proj = torch.nn.Parameter(torch.rand(inner_dim, embed_dim))
        # self.v_bias = torch.nn.Parameter(torch.rand(inner_dim))
        # QKV
        self.qkv_proj = torch.nn.Parameter(torch.rand(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.rand(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.rand(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.rand(embed_dim))


    def forward(self, query: torch.Tensor, past_embed_key: torch.Tensor, past_embed_value: torch.Tensor):
        attn = one_attention(
            query, past_embed_key, past_embed_value,
            # self.q_proj, self.q_bias,
            # self.k_proj, self.k_bias,
            # self.v_proj, self.v_bias,
            self.qkv_proj, self.qkv_bias,
            self.out_proj, #self.out_bias,
            self.num_heads, self.scaling, self.dropout_p, self.training, mask=True
        )
        attn = attn + self.out_bias
        return attn