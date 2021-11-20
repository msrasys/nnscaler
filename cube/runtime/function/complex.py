import torch
import torch.nn.functional as F


def toqkv(hidden_state: torch.Tensor, weight: torch.Tensor,
          num_heads: int):
    """
    Inputs:
        hidden_state: [L, N, E] (seqlen, batch size, num_heads * dim_head)
        weight: [E * 3, E]

    Returns:
        Q: [L, N * num_heads, dim_head]
        K: [L, N * num_heads, dim_head]
        V: [L, N * num_heads, dim_head]
    """
    seqlen = hidden_state.shape[0]
    bs = hidden_state.shape[1]
    dim_head = hidden_state.shape[2] // num_heads
    qkv = F.linear(hidden_state, weight, None)
    qkv = qkv.chunk(3, dim=-1)
    q, k, v = qkv
    q = q.contiguous()
    q = q.view(seqlen, (bs * num_heads), dim_head)
    k = k.contiguous()
    k = k.view(seqlen, (bs * num_heads), dim_head)
    v = v.contiguous()
    v = v.view(seqlen, (bs * num_heads), dim_head)
    return q, k, v


def tril_mask(input: torch.Tensor, num_heads: int):
    """
    Inputs:
        input: [N * num_heads, L, L]
        num_head: int
    
    Returns:
        output: [N * num_heads, L, L]
    """
    bs: int = input.shape[0] // num_heads
    seqlen: int = input.shape[2]
    input = input.view(bs, num_heads, seqlen, seqlen)
    # set up mask
    ones = torch.ones(
        (bs, seqlen, seqlen),
        device=input.device,
    )
    mask = torch.tril(ones)
    mask = mask.view(bs, 1, seqlen, seqlen)
    mask = (mask < 0.5)
    # mask
    masked_input = input.masked_fill_(mask, -100000.0)
    masked_input = masked_input.view((bs * num_heads), seqlen, seqlen)
    return masked_input


def attn_view(input: torch.Tensor, num_heads: int):
    """
    Inputs:
        [N * num_heads, L, dim_head]

    Outputs:
        [L, N, num_heads * dim_head]
    """
    bs: int = input.shape[0] // num_heads
    seqlen: int = input.shape[1]
    dim_head = input.shape[2]
    # [(N * num_heads), L, dim_head] -> [L, (N * num_heads), dim_head]
    input = input.transpose(0, 1).contiguous()
    # [L, (N * num_heads), dim_head] -> [L, N, (num_heads * dim_head)]
    input = input.view(seqlen, bs, num_heads * dim_head)
    return input
