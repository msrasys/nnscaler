import torch
from torch.nn.parameter import Parameter
torch.manual_seed(121)


def linear(input, weight, bias=None):
    output = torch._C._nn.linear(input, weight, bias)
    return output


if __name__ == '__main__':

    torch.cuda.set_device(0)

    # tensor definition
    batch_size = 32
    out_features = 10240
    in_features = 10240
    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    # print('weight: ', weight)
    bias = torch.rand(out_features).cuda().requires_grad_()
    # print('bias: ', bias)
    input = torch.rand((batch_size, in_features)).cuda()
    # print('input: ', input)
    
    # op compute
    print('======== Naive Single Device =======')
    output = linear(input, weight, bias)
    loss = torch.mean(output) * 100
    print(loss)
    loss.backward()
    print('weight grad: ', weight.grad.t())
    print('======== Naive Single Device =======')
