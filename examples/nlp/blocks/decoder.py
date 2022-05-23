import torch
from examples.nlp.blocks.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from examples.nlp.blocks.mlp import MLP


class DecoderLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )

        self.cross_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)

        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x, encoder_output)

        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)

        x = self.dropout(x)
        x = x + residual
        return x
