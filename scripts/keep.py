import torch
import time

interval = 10

a = torch.rand((8192, 8192)).cuda()
b = torch.rand((8192, 8192)).cuda()


while True:
    for _ in range(1000):
        c = a * b
    time.sleep(interval)
