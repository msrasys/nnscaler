from typing import List

import torch
import torch.nn.functional as F
import time

torch.set_default_tensor_type(torch.DoubleTensor)

import cube
from examples.poisson.policy.naive import PAS

"""
OMP_NUM_THREADS=4 torchrun --standalone \
    --nproc_per_node=2 \
    --nnodes=1 \
    examples/poisson/sci.py
"""


class ScientificModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, r0: torch.Tensor, p: torch.Tensor, phi: torch.Tensor,
                filter: torch.Tensor):
        conv_out = F.conv2d(p, filter, padding=1)
        alpha = torch.mul(r0, r0).sum() / torch.mul(p, conv_out).sum()
        r1 = r0 - alpha * conv_out
        # update
        phi = phi + alpha * p
        r1_sum = torch.mul(r1, r1).sum()
        beta = r1_sum / torch.mul(r0, r0).sum()
        p = r1 + beta * p
        return r1, p, phi, r1_sum


class LoopVariables(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, variables: List[torch.Tensor], constants: List[torch.Tensor]):

        shapes = [list(var.size()) for var in variables + constants]
        dtypes = [var.dtype for var in variables + constants]
        batch_dims = [0] * (len(variables) + len(constants))
        super().__init__(shapes, dtypes, batch_dims)
        self.variables = list()
        self.constants = list()
        for var in variables:
            if torch.is_tensor(var) and var.device != torch.cuda.current_device():
                var = var.cuda()
            self.variables.append(var)
        for const in constants:
            if torch.is_tensor(const) and const.device != torch.cuda.current_device():
                const = const.cuda()
            self.constants.append(const)

    def __iter__(self):
        return self
    
    def update(self, variables: List[torch.Tensor] = None, constants: List[torch.Tensor] = None):
        if variables is not None:
            self.variables = variables
        if constants is not None:
            self.constants = constants

    def reset(self, batch_size):
        pass

    def __next__(self):
        if len(self.variables) + len(self.constants) == 1:
            return (self.variables + self.constants)[0]
        return tuple(self.variables + self.constants)


def train_loop():
    # initialize
    N = 1024 * 2
    filter = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]]
    ).view(1, 1, 3, 3)
    rho = F.conv2d(torch.ones((1, 1, N, N)), filter, padding=1)
    phi = torch.zeros((1, 1, N, N))
    r0 = rho - F.conv2d(phi, filter, padding=1)
    p = r0

    varloader = LoopVariables(variables=[r0, p, phi], constants=[filter])
    model = ScientificModel()
    model = cube.SemanticModel(model, input_shapes=tuple(varloader.shapes),)

    @cube.compile(model=model, dataloader=varloader, PAS=PAS)
    def train_iter(model, dataloader):
        r0, p, phi, filter = next(dataloader)
        r0, p, phi, r1_sum = model(r0, p, phi, filter)
        return r0, p, phi, r1_sum
    model = model.get_gen_module()

    start = time.time()

    counter = 0
    while True:
        counter += 1
        r0, p, phi, r1_sum = train_iter(model, varloader)
        varloader.update(variables=[r0, p, phi])
        if counter % 100 == 0:
            print('iters:\t', counter)
            print('rnorm:\t', torch.sqrt(r1_sum))
        if torch.sqrt(r1_sum) < 1e-10:
            print('**************** Converged ****************')
            print('iters:\t', counter)
            torch.cuda.synchronize()
            print('time:\t', time.time() - start)
            print('error:\t', torch.norm(phi - torch.ones((1, 1, N, N)).cuda()))
            break


if __name__ == '__main__':
    cube.init()
    train_loop()