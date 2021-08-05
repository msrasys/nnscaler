import torch


if __name__ == '__main__':

    global_bs = 128
    bs = 32
    feats = 1024

    inputs = [torch.randn((bs, feats)).cuda() for _ in range(16)]
    weight = torch.randn((feats, feats)).cuda().requires_grad_()
    bias = torch.randn((feats,)).cuda().requires_grad_()

    update_interval = int(global_bs / bs)
    tic = 0
    for input_data in inputs:
        tic += 1
        # forward
        print('forward')
        out = torch._C._nn.linear(input_data, weight, bias)
        loss = torch.sum(out)
        # backward - calculate grad:
        # note pytorch in default accumulates gradients
        print('backward')
        loss.backward()
        
        # weight update
        if tic % update_interval == 0:
            print('weight update')
            weight.data += weight.grad
            weight.grad = None
            bias.data += bias.grad
            bias.grad = None
