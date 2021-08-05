import torch
from torch.nn.parameter import Parameter
torch.manual_seed(121)


def linear(input, weight, bias=None):
    output = torch._C._nn.linear(input, weight, bias)
    return output


if __name__ == '__main__':

    torch.cuda.set_device(0)

    # tensor definition
    batch_size = 128
    out_features = 1024
    in_features = 1024
    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    bias = torch.rand(out_features).cuda().requires_grad_()
    input = torch.rand((batch_size, in_features)).cuda()
    # print('weight: ', weight)
    # print('bias: ', bias)
    # print('input: ', input)
    
    # iterations
    for _ in range(4):
        # forward
        output = linear(input, weight, bias)
        loss = torch.mean(output)
        print(loss)
        # backward
        loss.backward()
        # print('weight grad: ', weight.grad.t())
        # weight update
        weight.data += weight.grad
        weight.grad = None
        bias.data += bias.grad
        bias.grad = None
