"""
Attention Module for MSA Attention and Pair Attention in Evoformer
"""

import cube
import torch
import torch.utils.checkpoint as ckpt


@cube.graph.parser.register('N S R^ M^, M^ (head+ dim), M^ (head+ dim 3), (head+ dim) M^ -> N S R^ M^', name='msa_attn')
@torch.jit.ignore
def msa_attn(x: torch.Tensor, gate_proj: torch.Tensor,
                 qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
                 c: int, scale: float, chunk_size: int, is_train: bool):
    # cube.profiler.CudaTimer().start('msa_attn')
    bs, s, r, cm = x.size()

    if chunk_size == -1:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)
        gate = gate.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c)
        q = q.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c)
        k = k.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c).transpose(1, 2)
        v = v.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c)
        sim = torch.bmm(q, k) * scale
        sim = torch.nn.functional.softmax(sim, dim=-1)
        attend = torch.bmm(sim, v) * gate
        out = attend.reshape(bs, s, head, r, c).transpose(2, 3).reshape(bs, s, r, head * c)
        out = torch.matmul(out, out_proj)
    else:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)
        gate = gate.reshape(bs, s, r, head, c).transpose(2, 3)
        q = q.reshape(bs, s, r, head, c).transpose(2, 3)
        k = k.reshape(bs, s, r, head, c).transpose(2, 3)
        v = v.reshape(bs, s, r, head, c).transpose(2, 3)
        assert s % chunk_size == 0
        out_chunks = []

        def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      gate: torch.Tensor, start: int):
            cur_q = q[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)
            cur_k = k[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c).transpose(1, 2)
            cur_v = v[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)
            cur_gate = gate[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)
            sim = torch.bmm(cur_q, cur_k) * 0.125
            sim = torch.nn.functional.softmax(sim, dim=-1)
            attend = torch.bmm(sim, cur_v) * cur_gate
            attend = attend.reshape(bs, chunk_size, head, r, c).transpose(
                2, 3).reshape(bs, chunk_size, r, head * c)
            return attend

        for start in range(0, s, chunk_size):
            attend = ckpt.checkpoint(attention, q, k, v, gate, start)
            # attend = attention(q, k, v, gate, start)
            out_chunks.append(attend)

        out = torch.matmul(torch.cat(out_chunks, dim=1), out_proj)
    # cube.profiler.CudaTimer().stop('msa_attn')
    return out


@cube.graph.parser.register('N S R^ M^, M^ (head+ dim), M^ (head+ dim 3), (head+ dim) M^, N 1 head+ R^ R^ -> N S R^ M^', name='msa_attn_bias')
@torch.jit.ignore
def msa_attn_bias(x: torch.Tensor, gate_proj: torch.Tensor,
                  qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                  bias: torch.Tensor, head: int, c: int, scale: float,
                  chunk_size: int, is_train: bool):
    # cube.profiler.CudaTimer().start('msa_attn_bias')
    bs, s, r, cm = x.size()
    assert gate_proj.size(1) % head == 0
    c = gate_proj.size(1) // head

    if chunk_size == -1:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)  # N S R (head dim)
        gate = gate.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c) # (N S head) r dim
        q = q.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c) # (N S head) r dim
        k = k.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c).transpose(1, 2) # (N S head) dim r
        v = v.reshape(bs, s, r, head, c).transpose(2, 3).reshape(bs * s * head, r, c)
        sim = torch.bmm(q, k) * scale  # (N S head) r r
        sim = torch.nn.functional.softmax(sim, dim=-1)  # (N S head) r r
        sim = sim.reshape(bs, s, head, r, r) + bias  # N S head r r, N S 1 r r
        sim = sim.reshape(bs * s * head, r, r)  # (N S head) r r
        attend = torch.bmm(sim, v) * gate # (N S head) r dim
        out = attend.reshape(bs, s, head, r, c).transpose(2, 3).reshape(bs, s, r, head * c)
        out = torch.matmul(out, out_proj)
    else:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)
        gate = gate.reshape(bs, s, r, head, c).transpose(2, 3)
        q = q.reshape(bs, s, r, head, c).transpose(2, 3)
        k = k.reshape(bs, s, r, head, c).transpose(2, 3)
        v = v.reshape(bs, s, r, head, c).transpose(2, 3)
        assert s % chunk_size == 0
        out_chunks = []

        def attention_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           gate: torch.Tensor, bias: torch.Tensor, start: int):
            cur_q = q[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)
            cur_k = k[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c).transpose(1, 2)
            cur_v = v[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)
            cur_gate = gate[:, start:start + chunk_size, :, :, :].reshape(
                bs * chunk_size * head, r, c)

            sim = torch.bmm(cur_q, cur_k) * scale
            sim = torch.nn.functional.softmax(sim, dim=-1)
            sim = sim.reshape(bs, chunk_size, head, r, r) + bias
            sim = sim.reshape(bs * chunk_size * head, r, r)

            attend = torch.bmm(sim, cur_v) * cur_gate
            attend = attend.reshape(bs, chunk_size, head, r, c).transpose(
                2, 3).reshape(bs, chunk_size, r, cm)
            return attend

        for start in range(0, s, chunk_size):
            if is_train:
                attend = ckpt.checkpoint(attention_bias, q, k, v, gate, bias, start)
            else:
                attend = attention_bias(q, k, v, gate, bias, start)
            out_chunks.append(attend)
        out = torch.matmul(torch.cat(out_chunks, dim=1), out_proj)
    # cube.profiler.CudaTimer().stop('msa_attn_bias')
    return out


# note: code not reused constrained by cube's interface
@cube.graph.parser.register('N S R^ M^, N R^ R^ Z^, M^ (head+ dim), M^ (head+ dim 3), (head+ dim) M^, Z^ head+ -> N S R^ M^', name='row_attn')
def row_attn(msa_repr: torch.Tensor, pair_repr: torch.Tensor,
             gate_proj: torch.Tensor,
             qkv_proj: torch.Tensor, out_proj: torch.Tensor,
             bias_proj: torch.Tensor, head: int, c: int,
             scale: float, chunk_size: int, is_train: bool):
    # call: MSAAttentionWithBias
    bs, s, r, cm = msa_repr.size()
    # N R R Z, Z h -> N R R h -> N h S R -> N 1 h S R
    bias = torch.matmul(pair_repr, bias_proj).permute(0, 3, 1, 2).reshape(bs, 1, head, r, r)

    return msa_attn_bias(msa_repr, gate_proj, qkv_proj, out_proj, bias,
                        head, c, scale, chunk_size, is_train)


@cube.graph.parser.register('N S^ R M^, M^ (head+ dim), M^ (head+ dim 3), (head+ dim) M^ -> N S^ R M^', name='col_attn')
def col_attn(msa_repr: torch.Tensor, gate_proj: torch.Tensor,
             qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
             c: int, scale: float, chunk_size: int, is_train: bool):
    # call: MSAAttention
    msa_repr = msa_repr.permute(0, 2, 1, 3)
    out = msa_attn(
        msa_repr, gate_proj, qkv_proj, out_proj,
        head, c, scale, chunk_size, is_train)
    out = out.permute(0, 2, 1, 3)
    return out


# @cube.graph.parser.register('N S^ R^ M^, M^ (head+ dim^), M^ E^, M^ E^, M^ (head+ dim^), (head+ dim) M^ -> N S^ R^ M^', name='MSAColGlobalAttention')
def global_attn(msa_repr: torch.Tensor, q_proj: torch.Tensor,
                          k_proj: torch.Tensor, v_proj: torch.Tensor,
                          gate_proj: torch.Tensor, out_proj: torch.Tensor,
                          head: int, c: int, scale: float):
    # [N R S M]
    msa_repr = msa_repr.transpose(-2, -3)

    # [N R M]
    q = torch.sum(msa_repr, dim=-2)
    # [N R M]
    q = torch.matmul(q, q_proj) * scale
    # [N R H E]
    q = q.view(q.shape[:-1] + (head, -1))

    # [N R S E]
    k, v = torch.matmul(msa_repr, k_proj), torch.matmul(msa_repr, v_proj)

    # N R H E, N R E S -> N R H S
    a = torch.matmul(q, k.transpose(-1, -2))
    a = torch.nn.functional.softmax(a, dim=-1)
    # [N R H E]
    o = torch.matmul(a, v)

    # [N R S M]
    g = torch.sigmoid(torch.matmul(msa_repr, gate_proj))
    # [N R S H E]
    g = g.view(g.shape[:-1] + (head, -1))

    # [N R 1 H E]
    o = o.unsqueeze(-3) * g
    # [N R S M]
    o = o.reshape(o.shape[:-2] + (-1, ))

    return torch.matmul(o, out_proj).transpose(-2, -3)


@cube.graph.parser.register('N S R M^, M^ E+, E+ M^ -> N S R M^', name='feedforward')
@torch.jit.ignore
def feedforward(msa_repr: torch.Tensor, proj1: torch.Tensor, proj2: torch.Tensor):
    """
    MSA transition
    """
    # cube.profiler.CudaTimer().start('ffn')
    x = torch.matmul(msa_repr, proj1)
    x = torch.nn.functional.relu(x)
    x = torch.matmul(x, proj2)
    # cube.profiler.CudaTimer().stop('ffn')
    return x


@cube.graph.parser.register('N S R^ Z^, Z^ (head+ dim), Z^ (head+ dim 3), (head+ dim) Z^, N R^ R^ head+ -> N S R^ Z^', name='tri_attn_start')
def tri_attn_start(pair_repr: torch.Tensor,
                   gate: torch.Tensor, qkv: torch.Tensor,
                   out: torch.Tensor, bias: torch.Tensor,
                   head: int, c: int, scale: float,
                   chunk_size: int, is_train: bool):
    # bias = torch.matmul(pair_repr, bias).permute(0, 3, 1, 2).unsqueeze(1)
    bias = bias.permute(0, 3, 1, 2).unsqueeze(1)
    out = msa_attn_bias(pair_repr, gate, qkv, out, bias,
                        head, c, scale, chunk_size, is_train)
    return out


@cube.graph.parser.register('N S^ R Z^, Z^ (head+ dim), Z^ (head+ dim 3), (head+ dim) Z^, N S^ S^ head+ -> N S^ R Z^', name='tri_attn_end')
def tri_attn_end(pair_repr: torch.Tensor,
                 gate: torch.Tensor, qkv: torch.Tensor,
                 out: torch.Tensor, bias: torch.Tensor,
                 head: int, c: int, scale: float, chunk_size: int, is_train: bool):
    # bias = torch.matmul(pair_repr, bias).permute(0, 3, 2, 1).unsqueeze(1)
    bias = bias.permute(0, 3, 2, 1).unsqueeze(1)
    pair_repr = pair_repr.permute(0, 2, 1, 3)
    out = msa_attn_bias(pair_repr, gate, qkv, out, bias,
                        head, c, scale, chunk_size, is_train)
    return out.permute(0, 2, 1, 3)


class MSARowAttention(torch.nn.Module):
    """
    MSA Row Attention with Pair Bias
    """

    def __init__(self, hidden: int, heads: int, z: int, scale: float, chunk_size: int = -1):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.dhead = hidden // heads
        self.chunk_size = chunk_size
        self.scale = scale
        self.bias = torch.nn.Parameter(torch.empty(z, heads))
        self.gate = torch.nn.Parameter(torch.empty(hidden, hidden))
        self.qkv = torch.nn.Parameter(torch.empty(hidden, hidden * 3))
        self.out = torch.nn.Parameter(torch.empty(hidden, hidden))

    def forward(self, msa_repr: torch.Tensor, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        msa_repr: [N S R M]
        pair_repr: [N R R Z]
        """
        out = row_attn(
            msa_repr, pair_repr, self.gate, self.qkv, self.out, self.bias,
            self.heads, self.dhead, self.scale, self.chunk_size, self.training
        )
        return out


class MSAColAttention(torch.nn.Module):
    """
    MSA Coloumn Attention (no bias)
    """
    def __init__(self, hidden: int, heads: int, scale: float, chunk_size: int = -1) -> None:
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.dhead = hidden // heads
        self.chunk_size = chunk_size
        self.scale = scale
        self.gate = torch.nn.Parameter(torch.empty(hidden, hidden))
        self.qkv = torch.nn.Parameter(torch.empty(hidden, hidden * 3))
        self.out = torch.nn.Parameter(torch.empty(hidden, hidden))

    def forward(self, msa_repr: torch.Tensor) -> torch.Tensor:
        """
        msa_repr: [N S R M]
        """
        out = col_attn(
            msa_repr, self.gate, self.qkv, self.out, 
            self.heads, self.dhead, self.scale,self.chunk_size, self.training
        )
        return out


class Transition(torch.nn.Module):
    """
    Feedforward for msa_repr and pair_repr
    """
    def __init__(self, hidden: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty(hidden, ff_mult * hidden))
        self.proj2 = torch.nn.Parameter(torch.empty(ff_mult * hidden, hidden))

    def forward(self, msa_repr: torch.Tensor) -> torch.Tensor:
        """
        msa_repr: [N S R M]
        """
        return feedforward(msa_repr, self.proj1, self.proj2)


class TriangleAttentionNodeStart(torch.nn.Module):

    def __init__(self, cz: int, pair_head: int, c: int, scale: float, chunk_size=-1) -> None:
        super().__init__()
        self.heads = pair_head
        self.c = c
        self.scale = scale
        self.chunk_size = chunk_size
        self.layer_norm = torch.nn.LayerNorm(cz)
        self.gate = torch.nn.Parameter(torch.empty(cz, pair_head * c))
        self.qkv = torch.nn.Parameter(torch.empty(cz, 3 * pair_head * c))
        self.out = torch.nn.Parameter(torch.empty(pair_head * c, cz))
        self.bias = torch.nn.Parameter(torch.empty(cz, pair_head))

    def forward(self, pair_repr: torch.Tensor):
        """
        pair_repr: N R R cz
        """
        pair_repr = self.layer_norm(pair_repr)
        bias = torch.matmul(pair_repr, self.bias)
        pair_repr = tri_attn_start(
            pair_repr, self.gate, self.qkv, self.out, bias,
            self.heads, self.c, self.scale, self.chunk_size, self.training
        )
        return pair_repr


class TriangleAttentionNodeEnd(torch.nn.Module):

    def __init__(self, cz: int, pair_head: int, c: int, scale: float, chunk_size=-1) -> None:
        super().__init__()
        self.heads = pair_head
        self.c = c
        self.scale = scale
        self.chunk_size = chunk_size
        self.layer_norm = torch.nn.LayerNorm(cz)
        self.gate = torch.nn.Parameter(torch.empty(cz, pair_head * c))
        self.qkv = torch.nn.Parameter(torch.empty(cz, 3 * pair_head * c))
        self.out = torch.nn.Parameter(torch.empty(pair_head * c, cz))
        self.bias = torch.nn.Parameter(torch.empty(cz, pair_head))

    def forward(self, pair_repr: torch.Tensor):
        pair_repr = self.layer_norm(pair_repr)
        bias = torch.matmul(pair_repr, self.bias)
        pair_repr = tri_attn_end(
            pair_repr, self.gate, self.qkv, self.out, bias,
            self.heads, self.c, self.scale, self.chunk_size, self.training
        )
        return pair_repr

