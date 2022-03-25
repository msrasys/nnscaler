"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 handcraft/mbart/test.py
"""

import torch
import cube
from cube.profiler.memory import memory_summary, model_summary
from handcraft.mbart.tp import AllReduceIdentity, IdentityAllreduce

scale = 7
embed_dim = 1024 + int(1024 * (scale * 0.25))
print(f'embed dim = {embed_dim}')



class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(500000, embed_dim)

    def forward(self, x):
        out = self.embed(x)
        loss = torch.sum(out)
        return loss

cube.init()
print('loading...')
model = Model().cuda()
input_ids = torch.randint(0, 25000, (1, 1024), dtype=torch.int).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

print('training...')
for _ in range(3):

    loss = model(input_ids)
    loss.backward()
    optimizer.step()

memory_summary()
