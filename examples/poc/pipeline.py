"""
This is to check whether backward can be stopped in the middle

Verified by using `detach()`, `requires_grad_()` and `retain_grad()`
"""


import torch
from torch import nn

torch.manual_seed(100)


class LinearModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)

    def forward(self, x):
        x2_ = None

        x1 = self.linear1(x)

        x2 = self.linear2(x1)
        
        x2_ = x2.detach()
        x2_.requires_grad_()
        x2_.retain_grad()
        x3 = self.linear3(x2_)

        x4 = self.linear4(x3)

        return x4, x2, x2_


if __name__ == '__main__':

    bs = 32
    dim = 1024

    model = LinearModel(dim)
    model = model.cuda()

    inputs = torch.randn((bs, dim), device=torch.device('cuda:0'))

    output, x2, x2_ = model(inputs)
    loss = torch.sum(output)

    # check before backward grads
    # print('before linear1 weight grad:\n{}'.format(model.linear1.weight.grad))
    # print('before linear2 weight grad:\n{}'.format(model.linear3.weight.grad))
    # print('before x2 tensor:\n{}'.format(x2.grad))
    # print('===============================')
    assert model.linear1.weight.grad is None
    assert model.linear2.weight.grad is None

    loss.backward()
    assert model.linear1.weight.grad is None
    assert torch.is_tensor(model.linear3.weight.grad) is True
    # print('after linear1 weight grad :\n{}'.format(model.linear1.weight.grad))
    # print('after linear2 weight grad :\n{}'.format(model.linear3.weight.grad))
    # print('after x2 tensor:\n{}'.format(x2.grad))

    torch.autograd.backward(x2, grad_tensors=x2_.grad)
    assert torch.is_tensor(model.linear1.weight.grad) is True
    # print('===============================')
    # print('after autograd linear1 weight grad :\n{}'.format(model.linear1.weight.grad))
