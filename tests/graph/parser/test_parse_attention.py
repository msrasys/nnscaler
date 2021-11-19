from torch import nn
import torch
import torch.nn.functional as F

from cube.graph import parser
import cube


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, bs, seq_len, embed_dim, heads, dropout):
        super().__init__()

        self.batch_size = bs
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

    def forward(self, x): #, mask):
        """
        x: [L, N, E]: seq_len, batch_size, embedding dimension
        output: [L, N, E]
        """
        bs = self.batch_size

        # # [L, N, E] -> [L, N, (num_heads * dim_head * 3)]
        qkv = F.linear(x, self.weight_qkv, None)
    
        # # [L, N, E] -> [L, N, (num_heads * dim_head)] x 3
        # qkv = qkv.chunk(3, dim=-1)
        # q, k, v = qkv
        # 
        # # [L, N, (num_heads * dim_head)] -> [L, (N * num_heads), dim_head]
        # q = q.contiguous()
        # q = q.view(self.seq_len, (bs * self.num_heads), self.dim_head)
        # k = k.contiguous()
        # k = k.view(self.seq_len, (bs * self.num_heads), self.dim_head)
        # v = v.contiguous()
        # v = v.view(self.seq_len, (bs * self.num_heads), self.dim_head)

        # [L, N, E] -> 3 x [L, (N * num_heads), dim_head]
        q, k, v = cube.runtime.function.toqkv(
            qkv, bs, self.seq_len, self.num_heads, self.dim_head
        )

        # [L, (N * num_heads), dim_head] -> [(N * num_heads), L, dim_head]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # [(N * num_heads), L, dim_head] -> [(N * num_heads), L, dim_head]
        q = q * self.scale
        # [(N * num_heads), L, dim_head] * [(N * num_heads), dim_head, L]
        # -> [(N * num_heads), L, L]
        k = k.transpose(-2, -1)
        attn = torch.bmm(q, k)

        # # [(N * num_heads), L, L] -> [N, num_heads, L, L]
        # attn = attn.view(bs, self.num_heads, self.seq_len, self.seq_len)
        # # [N, num_heads, L, L] -> [N, num_heads, L, L]
        # attn = attn.masked_fill_(mask, -100000.0)
        # # [N, num_heads, L, L] -> [(N * num_heads), L, L]
        # attn = attn.view((bs * self.num_heads), self.seq_len, self.seq_len)
        attn = cube.runtime.function.tril_mask(attn, bs)

        # [(N * num_heads), L, L] -> [(N * num_heads), L, L]
        attn = F.softmax(attn, dim=-1)

        # [(N * num_heads), L, L] -> [(N * num_heads), L, L]
        attn = self.dropout(attn)
        # [(N * num_heads), L, L] * [(N * num_heads), L, dim_head]
        # -> [(N * num_heads), L, dim_head]
        output = torch.bmm(attn, v)

        # # [(N * num_heads), L, dim_head] -> [L, (N * num_heads), dim_head]
        # output = output.transpose(0, 1)
        # output = output.contiguous()
        # # [L, (N * num_heads), dim_head] -> [L, N, (num_heads * dim_head)]
        # output = output.view(self.seq_len, bs, self.embed_dim)
        output = cube.runtime.function.attn_view(output, self.num_heads)

        # [L, N, (num_heads * dim_head)] * [(num_heads * dim_head), (num_heads * dim_head)]
        # => [L, N, (num_heads * dim_head)]
        output = F.linear(output, self.weight_out)
        return output


def test_parse_attention():

    L = 64      # seq len
    N = 16      # batch
    E = 1024    # hiddend size = dim_head * num_head
    n_heads = 8

    model = MultiHeadSelfAttention(N, L, E, n_heads, dropout=0.5)
    module = torch.jit.script(model)
    print(module.graph)
    # print(module.code)

    graph = parser.convert(model, input_shapes=([L, N, E],))
    print(graph)

    assert False
