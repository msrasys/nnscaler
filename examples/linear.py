import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.functional as F

import combo.physical.operator as combo_op

import math
import argparse


class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return combo_op.linear(input, self.weight, self.bias)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4., classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(dim * mult, dim)
        )

        self.classifier = Linear(dim, classes)

    def forward(self, x, labels):
        output = self.net(x)
        output = self.classifier(output)
        loss = F.cross_entory(output, labels)
        return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--classes', type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    model = FeedForward(args.dim)
    model = model.cuda()

    inputs = torch.rand((args.bs, args.dim)).cuda()
    labels = torch.randint((args.bs, args.classes)).cuda()
    for _ in range(100):
        loss = model(inputs, labels)
        loss.backward()
    print('Done.')