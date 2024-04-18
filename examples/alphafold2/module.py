import nnscaler
import torch
import torch.utils.checkpoint as ckpt


@nnscaler.graph.parser.register('*, *, * -> *, *, *, *', name='calc_qkvg')
def calc_qkvg(x: torch.Tensor, qkv_proj: torch.Tensor, gate_proj: torch.Tensor,
              bs: int, s: int, r: int, head: int, c: int):
    gate = torch.sigmoid(torch.matmul(x, gate_proj))
    q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)

    gate = gate.reshape(bs, s, r, head, c).transpose(2, 3)
    q = q.reshape(bs, s, r, head, c).transpose(2, 3)
    k = k.reshape(bs, s, r, head, c).transpose(2, 3)
    v = v.reshape(bs, s, r, head, c).transpose(2, 3)
    return q, k, v, gate


"""
[bs, s, r, cm] -> [bs, s, r, cm]

used as column-wise gated self-attention
"""


@nnscaler.graph.parser.register('N S R^ M^, M^ E^, M^ F^, E^ M^ -> N S R^ M^',
                            name='MSAAttention')
@torch.jit.ignore
def MSAAttention(x: torch.Tensor, gate_proj: torch.Tensor,
                 qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
                 c: int, scale: float, chunk_size: int, is_train: bool):
    bs, s, r, cm = x.size()

    if chunk_size == -1:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)

        gate = gate.reshape(bs, s, r, head,
                            c).transpose(2, 3).reshape(bs * s * head, r, c)
        q = q.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r, c)
        k = k.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r,
                                                 c).transpose(1, 2)
        v = v.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r, c)

        sim = torch.bmm(q, k) * scale
        sim = torch.nn.functional.softmax(sim, dim=-1)

        attend = torch.bmm(sim, v) * gate

        out = attend.reshape(bs, s, head, r,
                             c).transpose(2, 3).reshape(bs, s, r, cm)
        out = torch.matmul(out, out_proj)
    else:
        if is_train:
            q, k, v, gate = ckpt.checkpoint(calc_qkvg, x, qkv_proj, gate_proj,
                                            bs, s, r, head, c)
        else:
            q, k, v, gate = calc_qkvg(x, qkv_proj, gate_proj, bs, s, r, head,
                                      c)
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
                2, 3).reshape(bs, chunk_size, r, cm)
            return attend

        for start in range(0, s, chunk_size):
            if is_train:
                attend = ckpt.checkpoint(attention, q, k, v, gate, start)
            else:
                attend = attention(q, k, v, gate, start)
            out_chunks.append(attend)

        out = torch.matmul(torch.cat(out_chunks, dim=1), out_proj)
    return out


@nnscaler.graph.parser.register('N S R^ M^, M^ E^, M^ F^, E^ M^, N 1^ 8^ R^ R^ -> N S R^ M^',
                            name='MSAAttentionWithBias')
@torch.jit.ignore
def MSAAttentionWithBias(x: torch.Tensor, gate_proj: torch.Tensor,
                         qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                         bias: torch.Tensor, head: int, c: int, scale: float,
                         chunk_size: int, is_train: bool):
    bs, s, r, cm = x.size()
    assert cm % head == 0
    c = cm // head

    if chunk_size == -1:
        gate = torch.sigmoid(torch.matmul(x, gate_proj))
        q, k, v = torch.matmul(x, qkv_proj).chunk(3, dim=-1)

        gate = gate.reshape(bs, s, r, head,
                            c).transpose(2, 3).reshape(bs * s * head, r, c)
        q = q.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r, c)
        k = k.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r,
                                                 c).transpose(1, 2)
        v = v.reshape(bs, s, r, head,
                      c).transpose(2, 3).reshape(bs * s * head, r, c)

        sim = torch.bmm(q, k) * scale
        sim = torch.nn.functional.softmax(sim, dim=-1)

        sim = sim.reshape(bs, s, head, r, r) + bias
        sim = sim.reshape(bs * s * head, r, r)

        attend = torch.bmm(sim, v) * gate

        out = attend.reshape(bs, s, head, r,
                             c).transpose(2, 3).reshape(bs, s, r, cm)
        out = torch.matmul(out, out_proj)
    else:
        if is_train:
            q, k, v, gate = ckpt.checkpoint(calc_qkvg, x, qkv_proj, gate_proj,
                                            bs, s, r, head, c)
        else:
            q, k, v, gate = calc_qkvg(x, qkv_proj, gate_proj, bs, s, r, head,
                                      c)

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
                attend = ckpt.checkpoint(attention_bias, q, k, v, gate, bias,
                                         start)
            else:
                attend = attention_bias(q, k, v, gate, bias, start)

            out_chunks.append(attend)
        out = torch.matmul(torch.cat(out_chunks, dim=1), out_proj)
    return out


"""
([bs, s, r, cm], [bs, r, r, cz]) -> [bs, s, r, cm]
"""


# note: code not reused constrained by cube's interface
@nnscaler.graph.parser.register('N S R^ M^, N R^ R^ Z^, M^ E^, M^ F^, E^ M^, Z^ H^ -> N S R^ M^',
                            name='MSARowAttentionWithPairBias')
def MSARowAttentionWithPairBias(msa_repr: torch.Tensor,
                                pair_repr: torch.Tensor,
                                gate_proj: torch.Tensor,
                                qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                                bias_proj: torch.Tensor, head: int, c: int,
                                scale: float, chunk_size: int, is_train: bool):
    # call: MSAAttentionWithBias
    bs, s, r, cm = msa_repr.size()

    bias = torch.matmul(pair_repr,
                        bias_proj).permute(0, 3, 1,
                                           2).reshape(bs, 1, head, r, r)

    return MSAAttentionWithBias(msa_repr, gate_proj, qkv_proj, out_proj, bias,
                                head, c, scale, chunk_size, is_train)


@nnscaler.graph.parser.register('N S^ R M^, M^ E^, M^ F^, E^ M^ -> N S^ R M^',
                            name='MSAColAttention')
def MSAColAttention(msa_repr: torch.Tensor, gate_proj: torch.Tensor,
                    qkv_proj: torch.Tensor, out_proj: torch.Tensor, head: int,
                    c: int, scale: float, chunk_size: int, is_train: bool):
    # call: MSAAttention
    return MSAAttention(msa_repr.permute(0, 2, 1, 3), gate_proj, qkv_proj,
                        out_proj, head, c, scale, chunk_size,
                        is_train).permute(0, 2, 1, 3)


@nnscaler.graph.parser.register('N S^ R^ M^, M^ M^, M^ E^, M^ E^, M^ M^, M^ M^ -> N S^ R^ M^',
                            name='MSAColGlobalAttention')
def MSAColGlobalAttention(msa_repr: torch.Tensor, q_proj: torch.Tensor,
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

    # [N R H S]
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


"""
[bs, s, r, cm] -> [bs, s, r, cm]
"""


@nnscaler.graph.parser.register('N S R M^, M^ E^, E^ M^ -> N S R M^',
                            name='MSATransition')
def MSATransition(msa_repr: torch.Tensor, proj1: torch.Tensor,
                  proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(msa_repr, proj1)), proj2)


@nnscaler.graph.parser.register('N S R M^, M^ C^ -> N S R C^', name='OPMLeftProj')
def OPMLeftProj(msa_repr: torch.Tensor, proj: torch.Tensor):
    return torch.matmul(msa_repr, proj)


@nnscaler.graph.parser.register('N S R M^, M^ C^ -> N S R C^', name='OPMRightProj')
def OPMRightProj(msa_repr: torch.Tensor, proj: torch.Tensor):
    return torch.matmul(msa_repr, proj)


"""
[bs, s, r, cm] -> [bs, r, r, cz]
"""


@nnscaler.graph.parser.register('N S^ R M^, N S^ T^ M^, F^ Z^ -> N R^ T Z^',
                            name='OuterProductMean')
@torch.jit.ignore
def OuterProductMean(left_act: torch.Tensor, right_act: torch.Tensor,
                     out_proj: torch.Tensor, chunk_size: int, is_train: bool):
    bs, s, r, c = left_act.size()
    t = right_act.size(2)

    a = left_act.transpose(-2, -3)
    b = right_act.transpose(-2, -3)

    if chunk_size == -1:
        outer = torch.einsum('...bac,...dae->...bdce', a,
                             b).reshape(bs, r, t, c * c)
        outer = torch.matmul(outer, out_proj)
    else:
        out_chunks = []

        def opm(lhs: torch.Tensor, rhs: torch.Tensor, start: int):
            lhs_slice = lhs[:, start:start + chunk_size, :, :]
            out = torch.einsum('...bac,...dae->...bdce', lhs_slice,
                               rhs).reshape(bs, chunk_size, t, c * c)
            out = torch.matmul(out, out_proj)
            return out

        for start in range(0, r, chunk_size):
            if is_train:
                ret = ckpt.checkpoint(opm, a, b, start)
            else:
                ret = opm(a, b, start)
            out_chunks.append(ret)
        outer = torch.cat(out_chunks, dim=1)
    return outer


@nnscaler.graph.parser.register('N S R^ Z^, Z^ E^, Z^ E^ -> N S R^ E^', name='TMOLeftProj')
def TMOLeftProj(pair_repr: torch.Tensor, proj1: torch.Tensor,
                proj2: torch.Tensor):
    a = torch.sigmoid(torch.matmul(pair_repr, proj1))
    b = a * torch.matmul(pair_repr, proj2)
    return b


@nnscaler.graph.parser.register('N S R^ Z^, Z^ E^, Z^ E^ -> N S R^ E^',
                            name='TMORightProj')
def TMORightProj(pair_repr: torch.Tensor, proj1: torch.Tensor,
                 proj2: torch.Tensor):
    a = torch.sigmoid(torch.matmul(pair_repr, proj1))
    b = a * torch.matmul(pair_repr, proj2)
    return b


@nnscaler.graph.parser.register('N S T^ Z^, Z^ Z^ -> N S T^ Z^', name='TMOGate')
def TMOGate(pair_repr: torch.Tensor, proj: torch.Tensor):
    return torch.sigmoid(torch.matmul(pair_repr, proj))


@nnscaler.graph.parser.register('N S R^ E^, N T^ R^ E^, N S T^ Z^, E^, E^, E^ Z^ -> N S T^ Z^',
                            name='TriangleMultiplicationOut')
def TriangleMultiplicationOut(a: torch.Tensor, b: torch.Tensor,
                              g: torch.Tensor,
                              tri_mul_norm2_weight: torch.Tensor,
                              tri_mul_norm2_bias: torch.Tensor,
                              tri_mul_proj5: torch.Tensor, cz: int):
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 2, 1)

    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), tri_mul_norm2_weight,
                                       tri_mul_norm2_bias)
    p = torch.matmul(p, tri_mul_proj5)
    return p * g


@nnscaler.graph.parser.register('N R^ S Z^, Z^ E^, Z^ E^ -> N R^ S E^', name='TMILeftProj')
def TMILeftProj(pair_repr: torch.Tensor, proj1: torch.Tensor,
                proj2: torch.Tensor):
    a = torch.sigmoid(torch.matmul(pair_repr, proj1))
    a = a * torch.matmul(pair_repr, proj2)
    return a


@nnscaler.graph.parser.register('N R^ T Z^, Z^ E^, Z^ E^ -> N R^ T E^',
                            name='TMIRightProj')
def TMIRightProj(pair_repr: torch.Tensor, proj1: torch.Tensor,
                 proj2: torch.Tensor):
    a = torch.sigmoid(torch.matmul(pair_repr, proj1))
    a = a * torch.matmul(pair_repr, proj2)
    return a


@nnscaler.graph.parser.register('N S^ T Z^, Z^ Z^ -> N S^ T Z^', name='TMIGate')
def TMIGate(pair_repr: torch.Tensor, proj: torch.Tensor):
    return torch.sigmoid(torch.matmul(pair_repr, proj))


@nnscaler.graph.parser.register('N R^ S E^, N R^ T^ E^, N T^ S Z^, E^, E^, E^ Z^ -> N T^ S Z^',
                            name='TriangleMultiplicationIn')
def TriangleMultiplicationIn(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
                             tri_mul_norm2_weight: torch.Tensor,
                             tri_mul_norm2_bias: torch.Tensor,
                             tri_mul_proj5: torch.Tensor, cz: int):
    a = a.permute(0, 3, 2, 1)
    b = b.permute(0, 3, 1, 2)

    p = torch.matmul(a, b).permute(0, 2, 3, 1)
    p = torch.nn.functional.layer_norm(p, (128, ), tri_mul_norm2_weight,
                                       tri_mul_norm2_bias)
    p = torch.matmul(p, tri_mul_proj5)
    return p.permute(0, 2, 1, 3) * g


@nnscaler.graph.parser.register('N S R^ C^, C^ D^ -> N S R^ D^', name='TANSBias')
def TANSBias(pair_repr: torch.Tensor, bias_proj: torch.Tensor):
    return torch.matmul(pair_repr, bias_proj)


@nnscaler.graph.parser.register('N S R^ Z^, Z^ E^, Z^ F^, E^ Z^, N T^ R^ G^ -> N S R^ Z^',
                            name='TriangleAttentionNodeStart')
def TriangleAttentionNodeStart(pair_repr: torch.Tensor,
                               gate_proj: torch.Tensor, qkv_proj: torch.Tensor,
                               out_proj: torch.Tensor, bias: torch.Tensor,
                               head: int, c: int, scale: float,
                               chunk_size: int, is_train: bool):
    # call: MSAAttentionWithBias
    bias = bias.permute(0, 3, 1, 2).unsqueeze(1)

    return MSAAttentionWithBias(pair_repr, gate_proj, qkv_proj, out_proj, bias,
                                head, c, scale, chunk_size, is_train)


@nnscaler.graph.parser.register('N S^ R C^, C^ D^ -> N S^ R D^', name='TANEBias')
def TANEBias(pair_repr: torch.Tensor, bias_proj: torch.Tensor):
    return torch.matmul(pair_repr, bias_proj)


@nnscaler.graph.parser.register('N R^ S Z^, Z^ E^, Z^ F^, E^ Z^, N R^ T^ G^ -> N R^ S Z^',
                            name='TriangleAttentionNodeEnd')
def TriangleAttentionNodeEnd(pair_repr: torch.Tensor, gate_proj: torch.Tensor,
                             qkv_proj: torch.Tensor, out_proj: torch.Tensor,
                             bias: torch.Tensor, head: int, c: int,
                             scale: float, chunk_size: int, is_train: bool):
    # call: TriangleAttentionNodeStart
    pair_repr = pair_repr.permute(0, 2, 1, 3)
    bias = bias.permute(0, 2, 1, 3)
    out = TriangleAttentionNodeStart(pair_repr, gate_proj, qkv_proj, out_proj,
                                     bias, head, c, scale, chunk_size,
                                     is_train)
    return out.permute(0, 2, 1, 3)


@nnscaler.graph.parser.register('N R T^ Z^, Z^ E^, E^ Z^ -> N R T^ Z^',
                            name='PairTransition')
def PairTransition(pair_repr: torch.Tensor, proj1: torch.Tensor,
                   proj2: torch.Tensor):
    return torch.matmul(
        torch.nn.functional.relu(torch.matmul(pair_repr, proj1)), proj2)


@nnscaler.graph.parser.register('* -> *, *', name='multi2ref')
def multi2ref(x: torch.Tensor):
    return (x, x)
