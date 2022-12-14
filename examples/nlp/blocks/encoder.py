import torch
from examples.nlp.blocks.attention import MultiHeadSelfAttention, MultiHeadOneAttention, MultiHeadSelfAttentionLrw
from examples.nlp.blocks.mlp import MLP

class EncoderLayerLrw(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionLrw(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual
         
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x


class EncoderInferLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int, seqlen: int = -1,
                 batch_size: int = 1,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()
        self.self_attn_partial = MultiHeadOneAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

        # id-embed + pos-embed
        tmp_batch_size = batch_size
        self.past_embed_key = torch.nn.Parameter(torch.rand(seqlen, tmp_batch_size, embed_dim))
        self.past_embed_value = torch.nn.Parameter(torch.rand(seqlen, tmp_batch_size, embed_dim))

    # def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn_partial(x, self.past_embed_key, self.past_embed_value)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x