import torch
import torch.nn.functional as F


def toqkv(hidden_state: torch.Tensor, weight: torch.Tensor,
          num_head: int):
    """
    Inputs:
        hidden_state: [L, N, E]
        weight: [3 * (num_head * dim_head), E]
        num_head: int

    where L = sequence length, N = batch size, E = num_head * dim_head

    Returns:
        Q: [L, N * num_head, dim_head]
        K: [L, N * num_head, dim_head]
        V: [L, N * num_head, dim_head]
    """
    seqlen = hidden_state.shape[0]
    bs = hidden_state.shape[1]
    dim_head = weight.shape[0] // 3 // num_head
    qkv = F.linear(hidden_state, weight, None)
    qkv = qkv.chunk(3, dim=-1)
    q, k, v = qkv
    q = q.contiguous()
    q = q.view(seqlen, (bs * num_head), dim_head)
    k = k.contiguous()
    k = k.view(seqlen, (bs * num_head), dim_head)
    v = v.contiguous()
    v = v.view(seqlen, (bs * num_head), dim_head)
    return q, k, v


def tril_mask(input: torch.Tensor, num_head: int):
    """
    Inputs:
        input: [N * num_head, L, L]
        num_head: int
    
    Returns:
        output: [N * num_head, L, L]
    """
    bs: int = input.shape[0] // num_head
    seqlen: int = input.shape[2]
    input = input.view(bs, num_head, seqlen, seqlen)
    # set up mask
    ones = torch.ones(
        (bs, seqlen, seqlen),
        device=input.device,
    )
    mask = torch.tril(ones)
    mask = mask.view(bs, 1, seqlen, seqlen)
    mask = (mask < 0.5)
    # mask
    masked_input = input.masked_fill_(mask, -10000.0)
    masked_input = masked_input.view((bs * num_head), seqlen, seqlen)
    return masked_input


def attn_view(input: torch.Tensor, num_head: int):
    """
    Inputs:
        [N * num_head, L, dim_head]

    Outputs:
        [L, N, num_head * dim_head]
    """
    bs: int = input.shape[0] // num_head
    seqlen: int = input.shape[1]
    dim_head = input.shape[2]
    # [(N * num_head), L, dim_head] -> [L, (N * num_head), dim_head]
    input = input.transpose(0, 1).contiguous()
    # [L, (N * num_head), dim_head] -> [L, N, (num_head * dim_head)]
    input = input.view(seqlen, bs, num_head * dim_head)
    return input
