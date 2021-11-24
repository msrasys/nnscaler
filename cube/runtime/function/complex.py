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


def self_attn(hidden_state, w_qkv, w_out,
              num_head: int, dim_head: int,
              dropout_p: float):
    """
    Multi-Head Self-Attention.

    L: sequence length
    N: batch size
    E: embedding size
    
    Inputs:
        hidden_state: [L, N, E]
        w_qkv       : [3 * num_head * dim_head, E]
        w_out       : [E, E]

    Outputs:
        hidden_state: [L, N, E]
    """
    scale = dim_head ** -0.5
    seqlen = hidden_state.shape[0]
    bs = hidden_state.shape[1]

    qkv = F.linear(hidden_state, w_qkv, None)
    qkv = qkv.chunk(3, dim=-1)
    q, k, v = qkv
    q = q.contiguous()
    q = q.view(seqlen, (bs * num_head), dim_head)
    k = k.contiguous()
    k = k.view(seqlen, (bs * num_head), dim_head)
    v = v.contiguous()
    v = v.view(seqlen, (bs * num_head), dim_head)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    q = q * scale
    k = k.transpose(-2, -1)
    attn = torch.bmm(q, k)

    attn = tril_mask(attn, num_head)
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, dropout_p, True, False)
    output = torch.bmm(attn, v)
    output = attn_view(output, num_head)
    
    output = F.linear(output, w_out, None)
    return output


def feedforward(hidden_state, w_proj1, w_bias1, w_proj2, w_bias2):
    """
    FeedForward

    Inputs:
        hidden_state: [L, N, E]
        w_proj1: [4 * E, E]
        w_bias1: [4 * E,]
        w_porj2: [E, 4 * E]
        w_bias2: [E,]
    """
    out = F.linear(hidden_state, w_proj1, w_bias1)
    out = F.gelu(out)
    out = F.linear(out, w_proj2, w_bias2)
    return out


def embedding(input: torch.Tensor, weight: torch.Tensor, start: int, stop: int):
    """
    Embedding

    Inputs:
        input: torch.Tensor [*]
        weight: [vocab size, embed size]
        start: int
        stop: int

    Outputs:
        output: [*, embed_size]
    """
    input_mask = (input < start) | (input >= stop)
    masked_input = input.clone() - start
    masked_input[input_mask] = 0
    output = F.embedding(
        masked_input, weight,
        None, None, 2.0, False, False
    )
    output[input_mask, :] = 0.0
    return output
