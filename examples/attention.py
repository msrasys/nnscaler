from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, seq_len, embed_dim, heads, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.dim_head = embed_dim // heads
        self.scale = self.dim_head ** -0.5

        self.weight_qkv = torch.nn.Parameter(torch.empty(
            3 * embed_dim, embed_dim
        ))
        self.weight_out = torch.nn.Parameter(torch.empty(
            embed_dim, embed_dim
        ))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_qkv)
        torch.nn.init.xavier_uniform_(self.weight_out)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: [L, N, E]: seq_len, batch_size, embedding dimension
        output: [L, N, E]
        """
        bs = x.shape[1]

        qkv = F.linear(x, self.weight_qkv, None).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.contiguous().view(self.seq_len, (bs * self.num_heads), self.dim_head)
        q = q.transpose(0, 1)
        # => q: (batch size, seq_len, embed_dim)
        k = k.contiguous().view(self.seq_len, (bs * self.num_heads), self.dim_head)
        k = k.transpose(0, 1)
        v = v.contiguous().view(self.seq_len, (bs * self.num_heads), self.dim_head)
        v = v.transpose(0, 1)

        q = q * self.scale
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        output = output.transpose(0, 1).contiguous()
        output = output.view(self.seq_len, bs, self.embed_dim)
        output = F.linear(output, self.weight_out)
        return output

    def _ref_forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        X: [L, N, E]: seq_len, batch_size, embedding dimension
        """
        output, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.weight_qkv,
            in_proj_bias=None,
            bias_k = None,
            bias_v = None,
            add_zero_attn=False,
            dropout_p=self.dropout.p,
            out_proj_weight=self.weight_out,
            out_proj_bias=None,
            training=self.training,
            need_weights=False
        )
        return output


class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = rearrange(qkv[0], 'b n (h d) -> b h n d', h = h)
        k = rearrange(qkv[0], 'b n (h d) -> b h n d', h = h)
        v = rearrange(qkv[0], 'b n (h d) -> b h n d', h = h)


        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if mask:
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = torch.softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


if __name__ == '__main__':

    L = 64
    N = 16
    E = 128
    n_heads = 8

    model = MultiHeadSelfAttention(L, E, n_heads, dropout=0.0)

    x = torch.rand((L, N, E))

    out_ref = model._ref_forward(x)
    out = model(x)
    # print(out)
    # print(out_ref)
    assert torch.allclose(out, out_ref) is True
    print('Test passed')
    module = torch.jit.script(model)
    print(module.graph)
    print(module.code)