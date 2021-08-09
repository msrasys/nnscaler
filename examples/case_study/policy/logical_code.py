import torch
from torch import nn
import torch.nn.functional as F

import argparse


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

    def forward(self, x):
        with annotate(data_parallel):
            output = self.net(x)
            output = self.classifier(output)
        return output


def data_iter(bs, dim, classes, length=64):
    for _ in range(length):
        data = torch.randn((bs, dim))
        label = torch.randint(0, classes, (bs,))
        yield data, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--classes', type=int, default=10)
    args = parser.parse_args()

    model = FeedForward(args.dim, mult=args.heads, classes=args.classes)
    # model = model.cuda()

    ### ======= get DAG and modify by policy ======= ###
    dag = get_dag(model, data)
    new_dag = policy(dag, resources)[myrank]
    model = new_dag
    ### ======= get DAG and modify by policy ======= ###

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
