import torch
from torch import nn
import torch.nn.functional as F

import argparse


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


def data_iter(bs, dim, classes, length=64):
    for _ in range(length):
        data = torch.randn((bs, dim))
        label = torch.randint(0, classes, (bs,))
        yield data, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--classes', type=int, default=10)
    args = parser.parse_args()

    model = torch.jit.script(FeedForward(args.dim).cuda())
    print(model.code)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.99),
        weight_decay=0
    )

    for (data, label) in data_iter(args.bs, args.dim, args.classes):
        data, label = data.cuda(), label.cuda()
        # forward
        output = model(data)
        loss = F.cross_entropy(output, label)
        # backward
        loss.backward()
        # weight update
        optimizer.step()
        optimizer.zero_grad()
