import torch
from cube.profiler import CudaTimer

bs, n, dim, heads, dim_head = 10, 2048, 4096, 16, 256
scale = 0.125

dev = torch.device('cuda:0')

def multi_head_attention(x: torch.Tensor, qkv_proj: torch.Tensor,
                         out_proj: torch.Tensor):

    q, kv = torch.matmul(x, qkv_proj).split((dim, dim_head), dim=-1)
    q = q.view(bs, n, heads, dim_head).transpose(1, 2)
    q = q.reshape(bs, heads * n, dim_head)
    trans_kv = kv.transpose(1, 2)
    sim = torch.bmm(q, trans_kv).view(bs, heads, n, n)
    attn = torch.nn.functional.softmax(sim, dim=-1)
    attn = attn.view(bs, heads * n, n)
    out = torch.bmm(attn, kv).view(bs, heads, n, dim_head)
    out = torch.transpose(out, 1, 2).reshape(bs, n, dim)
    out = torch.matmul(out, out_proj)
    return out

def ffn(x: torch.Tensor, xx: torch.Tensor, y: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
    return torch.matmul(x, w1), torch.matmul(xx * y, w2)

x = torch.randn(bs, n, dim).to(dev)
xx = torch.randn(bs, n, dim).to(dev)
y = torch.randn(bs, n, dim).to(dev)
qkv_proj = torch.randn(dim, dim+dim_head).to(dev)
q_proj = torch.randn(dim, dim).to(dev)
kv_proj = torch.randn(dim, dim_head).to(dev)
out_proj = torch.randn(dim, dim).to(dev)
w1 = torch.randn(dim, 2 * dim).to(dev)
w2 = torch.randn(dim, dim).to(dev)
score = torch.randn([bs * heads * n, n], requires_grad=True).to(dev)

CudaTimer(enable=False).warmup()

iter_num = 64
warmup = 20

for step in range(iter_num):
    softmax_score = torch.nn.functional.softmax(score, dim=-1)
    if step >= warmup:
        CudaTimer(enable=True).start('e2e')
    # out = multi_head_attention(x, qkv_proj, out_proj)
    # out = ffn(x, xx, y, w1, w2)
    out = torch.autograd.grad(outputs=softmax_score, inputs=score, grad_outputs=softmax_score)
    if step >= warmup:
        CudaTimer().stop('e2e')
    if (step + 1) % 20 == 0:
        print(f'iter [{step + 1}/{iter_num}]')

print('e2e time (ms) per iteration: {} ms'.format(
    CudaTimer().duration(iter_num - warmup, field_name='e2e')))