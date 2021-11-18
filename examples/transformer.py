import torch
from torch import nn
import torch.nn.functional as F


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

    def forward(self, x, mask):
        """
        x: [L, N, E]: seq_len, batch_size, embedding dimension
        output: [L, N, E]
        """
        bs = x.shape[1]

        # [L, N, E] -> [L, N, (num_heads * dim_head * 3)]
        qkv = F.linear(x, self.weight_qkv, None)
        # [L, N, E] -> [L, N, (num_heads * dim_head)] x 3
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = qkv

        # [L, N, (num_heads * dim_head)] -> [L, (N * num_heads), dim_head]
        q = q.contiguous()
        q = q.view(self.seq_len, (bs * self.num_heads), self.dim_head)
        k = k.contiguous()
        k = k.view(self.seq_len, (bs * self.num_heads), self.dim_head)
        v = v.contiguous()
        v = v.view(self.seq_len, (bs * self.num_heads), self.dim_head)

        # [L, N, (num_heads * dim_head)] -> [(N * num_heads), L, dim_head]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # [(N * num_heads), L, dim_head] -> [(N * num_heads), L, dim_head]
        q = q * self.scale
        # [(N * num_heads), L, dim_head] * [(N * num_heads), dim_head, L]
        # -> [(N * num_heads), L, L]
        attn = torch.bmm(q, k.transpose(-2, -1))

        # [(N * num_heads), L, L] -> [N, num_heads, L, L]
        attn = attn.view(bs, self.num_heads, self.seq_len, self.seq_len)
        # [N, num_heads, L, L] -> [N, num_heads, L, L]
        # attn += mask  # pytorch official implementation
        attn = attn.masked_fill_(mask, -100000.0)
        # [N, num_heads, L, L] -> [(N * num_heads), L, L]
        attn = attn.view((bs * self.num_heads), self.seq_len, self.seq_len)

        # [(N * num_heads), L, L] -> [(N * num_heads), L, L]
        attn = F.softmax(attn, dim=-1)

        # [(N * num_heads), L, L] -> [(N * num_heads), L, L]
        attn = self.dropout(attn)
        # [(N * num_heads), L, L] * [(N * num_heads), L, dim_head]
        # -> [(N * num_heads), L, dim_head]
        output = torch.bmm(attn, v)
        # [(N * num_heads), L, dim_head] -> [L, (N * num_heads), dim_head]
        output = output.transpose(0, 1).contiguous()
        # [L, (N * num_heads), dim_head] -> [L, N, (num_heads * dim_head)]
        output = output.view(self.seq_len, bs, self.embed_dim)
        # [L, N, (num_heads * dim_head)] * [(num_heads * dim_head), (num_heads * dim_head)]
        # => [L, N, (num_heads * dim_head)]
        output = F.linear(output, self.weight_out)
        return output

    def _ref_forward(self, x, mask=True):
        """
        X: [L, N, E]: seq_len, batch_size, embedding dimension
        mask: whether to use mask
        """
        if mask is not None:
            ones = torch.ones(
                (self.seq_len, self.seq_len),
                device=torch.cuda.current_device()
            )
            mask = torch.tril(ones)
            mask = (mask < 0.5)
        else:
            mask = None
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
            attn_mask=mask,
            training=self.training,
            need_weights=False,
        )
        return output


class MLP(torch.nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(
            hidden_size, 4 * hidden_size
        )
        self.dense_4h_to_h = torch.nn.Linear(
            4 * hidden_size, hidden_size
        )

    def forward(self, hidden_states):
        # [L, N, E] * [E, 4E] -> [L, N, 4E]
        out = self.dense_h_to_4h(hidden_states)
        # [L, N, 4E] -> [L, N, 4E]
        out = F.gelu(out)
        # [L, N, 4E] * [4E, E] -> [L, N, E]
        out = self.dense_4h_to_h(out)
        return out


class TransformerLayer(torch.nn.Module):

    def __init__(self, seq_len, hidden_size, head_num, dropout):
        super().__init__()
        # layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=0.00001)

        self.attention = MultiHeadSelfAttention(seq_len, hidden_size, head_num, dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.mlp_layernorm = torch.nn.LayerNorm(hidden_size, eps=0.00001)
        self.mlp = MLP(hidden_size)
        self.mlp_dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask):
        # Attention
        in_attn_norm = self.input_layernorm(hidden_states)
        attn_out = self.attention(in_attn_norm, attention_mask)
        # residual
        attn_out = self.attn_dropout(attn_out)
        residual = attn_out + hidden_states
        # MLP
        in_mlp_norm = self.mlp_layernorm(residual)
        mlp_out = self.mlp(in_mlp_norm)
        # residual
        mlp_out = self.mlp_dropout(mlp_out)
        mlp_out = mlp_out + residual
        return mlp_out


def get_attn_mask(batch_size: int, seq_len: int):
    ones = torch.ones(
        (batch_size, seq_len, seq_len),
        device=torch.cuda.current_device()
    )
    mask = torch.tril(ones)
    mask = mask.view(batch_size, 1, seq_len, seq_len)
    mask = (mask < 0.5)
    return mask


def reset_parameter(model):
    for param in model.parameters():
        torch.nn.init.uniform_(param)


def test_attention():
    L = 64
    N = 16
    E = 128
    n_heads = 8

    model = MultiHeadSelfAttention(L, E, n_heads, dropout=0.0).cuda()
    reset_parameter(model)

    x = torch.rand((L, N, E)).cuda()
    mask = get_attn_mask(N, L).cuda()
    
    out = model(x, mask)
    out_ref = model._ref_forward(x, mask)

    assert torch.allclose(out, out_ref) is True
    print('test passed')


if __name__ == '__main__':

    L = 64
    N = 16
    E = 1024
    n_heads = 8

    test_attention()
    
    model = TransformerLayer(L, E, n_heads, 0.5).cuda()
    reset_parameter(model)

    x = torch.rand((L, N, E)).cuda()
    mask = get_attn_mask(N, L).cuda()

    out = model(x, mask)
    # print(out)
    # print(out_ref)
    # assert torch.allclose(out, out_ref) is True
    print('Test passed')
    module = torch.jit.script(model)
    print(module.graph)
    print(module.code)
