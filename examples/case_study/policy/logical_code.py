import torch
from torch import nn
import torch.nn.functional as F

import argparse

def sschedule(partial_dag, resources): pass
def tschedule(train_fn): pass
resources = None # available hardware resources


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

        self.classifier = nn.Linear(dim, classes)

    def forward(self, data, label):
        output = self.net(data)
        output = self.classifier(output)
        loss = F.cross_entropy(output, label)
        return loss


def data_iter(gbs, dim, classes, length=1024, mbs=None):
    mbs = mbs if mbs is not None else gbs
    num_mb = gbs // mbs
    for _ in range(length):
        gbs_data = list()
        gbs_label = list()
        for _ in range(num_mb):
            mbs_data = torch.randn((mbs, dim))
            mbs_label = torch.randint(0, classes, (mbs,))
            gbs_data.append(mbs_data)
            gbs_label.append(mbs_label)
        yield gbs_data, gbs_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--gbs', type=int, default=64)
    parser.add_argument('--mbs', type=int, default=4)
    parser.add_argument('--classes', type=int, default=10)
    args = parser.parse_args()

    model = FeedForward(args.dim, mult=args.heads, classes=args.classes)
    # model = model.cuda()

    # spatial schedule
    model = sschedule(model, resources)
    # temporal schedule
    @tschedule
    def train_iter(data, label):
        # forward
        loss = model(data, label)
        # backward
        loss.backward()
        # update
        optimizer.step()
        optimizer.zero_grad()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.99),
        weight_decay=0
    )

    for (data, label) in data_iter(args.gbs, args.dim, args.classes, mbs=args.mbs):
        train_iter(data, label)
