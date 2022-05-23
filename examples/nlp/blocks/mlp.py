import torch
import cube


@cube.graph.parser.register('L^ N E^, H+ E^, H+, E H+, E -> L^ N E')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor, proj2_bias: torch.Tensor,
                dropout: float) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, True, False)
    x = torch.nn.functional.linear(x, proj2, proj2_bias)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float, bias=True):
        super().__init__()
        print((hidden_dim, embed_dim))
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = feedforward(x,
                        self.proj1, self.proj1_bias,
                        self.proj2, self.proj2_bias,
                        self.dropout)
        return x
