import torch
import time

interval = 2

torch.cuda.set_device(3)
a = torch.rand((8192, 8192)).cuda()
b = torch.rand((8192, 8192)).cuda()


while True:
    tic = time.time()
    for _ in range(5000):
        c = a * b
    torch.cuda.synchronize()
    toc = time.time()
    print('time span: {}s'.format(toc - tic))
    time.sleep(interval)
