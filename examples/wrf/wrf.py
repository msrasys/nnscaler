from typing import List

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch import nn
import torch.nn.functional as F
# from linalg import tridiagonal

import cube
from examples.poisson.policy.naive import PAS


device = 'cuda' #



def init(nx, ny, nz, dx, dy, dz, theta, PREF, Rd, g, p_t=None, p_s=None, u0=None, v0=None, w0=None, device='cuda'):
    # spatial discretization
    # dx, dy, dz, nx, ny, nz = dx, dy, 1. / (nz + 1), nx, ny, nz
    # agnostic variables
    P_t = p_t if p_t else torch.ones((1, ny, nx), device=device) * PREF * 0.0
    P_s = p_s if p_s else torch.ones((1, ny, nx), device=device) * PREF
    # pressure (nz, ny, nx)
    P = torch.linspace(dz, 1 - dz, nz, device=device).view(nz, 1, 1) * \
             (P_s - P_t).view(1, ny, nx) + P_t
    # Alpha (nz, ny, nx)
    Alpha = Rd / PREF * theta[1:-1] * (P / PREF) ** (-1 / 1.4)
    # prognostic variables
    # Mu (nz, ny, nx)
    Mu = torch.ones((nz, 1, 1), device=device) * (P_s - P_t).view(1, ny, nx)
    # Mu_t = (P_s - P_t).view(1, ny, nx)
    # Mu_s = (P_s - P_t).view(1, ny, nx)
    # Phi (nz - 1, ny, nx)
    Phi = torch.zeros((nz + 1, ny, nx), device=device)
    Phi[:-1] = Mu * Alpha * dz
    for i in range(nz - 1, -1, -1):
        Phi[i] += Phi[i + 1]
    Phi_t = Phi[0].view(1, ny, nx)
    Phi_s = Phi[-1].view(1, ny, nx)
    Phi = Phi[1:-1]
    # Theta (nz, ny, nx)
    theta_t = theta[0].view(1, ny, nx)
    theta_s = theta[-1].view(1, ny, nx)
    Theta = theta[1:-1] * Mu
    # U (nz, ny, nx + 1)
    U = u0 if u0 is not None else torch.zeros((nz, ny, nx + 1), device=device)
    # V (nz, ny + 1, nx)
    V = v0 if v0 is not None else torch.zeros((nz, ny + 1, nx), device=device)
    # W (nz - 1, ny, nx)
    W = w0 if w0 is not None else torch.zeros((nz - 1, ny, nx), device=device)

    return U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, theta_t, theta_s, P_t, P_s

class WRF(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.method_name()

    # def forward(self, dt):
    #     self.U, self.V, self.W, self.Theta, self.Mu, self.Phi = \
    #         self.RK3_step(self.U, self.V, self.W, self.Theta, self.Mu, self.Phi, dt)
    def forward(self,
                U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, theta_t, theta_s, P_t, P_s,
                dx, dy, dz,
                dt, PREF, Rd, g):

        U, V, W, Theta, Mu, Phi = self.RK3_step(U, V, W, Theta, Mu, Phi, dt, Phi_t, Phi_s, P_t, P_s, dx, dy, dz, PREF, Rd, g)
        return U, V, W, Theta, Mu, Phi

    def RHS(self,
            U, V, W, Theta, Mu, Phi,
            Phi_t, Phi_s,
            P_t, P_s, dx, dy, dz,
            PREF, Rd, g):

        # volecity
        u = U / self.bar_x(self.pad_x(Mu))
        v = V / self.bar_y(self.pad_y(Mu))
        w = W / self.bar_z(Mu)
        alpha = -self.delta_z(self.pad_z(Phi, Phi_t, Phi_s), dz) / Mu
        Alpha = alpha
        theta = Theta / Mu
        p = PREF * (Rd * theta / PREF / alpha)**1.4
        omega = -w * g / self.bar_z(alpha) / self.bar_z(Mu)
        Omega = omega * self.bar_z(Mu)
        #Omega = Omega

        # advection term
        R_U = - self.delta_x(self.bar_x(self.pad_x(U)) * self.bar_x(self.pad_x(u)), dx) \
              - self.delta_y(self.bar_x(self.pad_x(V)) * self.bar_y(self.pad_y(u)), dy) \
              - self.delta_z(self.bar_x(self.pad_x(self.pad_z(Omega))) * self.bar_z(self.pad_z(u)), dz)

        R_V = - self.delta_x(self.bar_y(self.pad_y(U)) * self.bar_x(self.pad_x(v)), dx) \
              - self.delta_y(self.bar_y(self.pad_y(V)) * self.bar_y(self.pad_y(v)), dy) \
              - self.delta_z(self.bar_y(self.pad_y(self.pad_z(Omega))) * self.bar_z(self.pad_z(v)), dz)

        R_W = - self.delta_x(self.bar_z(U) * self.bar_x(self.pad_x(w)), dx) \
              - self.delta_y(self.bar_z(V) * self.bar_y(self.pad_y(w)), dy) \
              - self.delta_z(self.bar_z(self.pad_z(Omega)) * self.bar_z(self.pad_z(w)), dz)

        R_Theta = - self.delta_x(U * self.bar_x(self.pad_x(theta)), dx) \
                  - self.delta_y(V * self.bar_y(self.pad_y(theta)), dy) \
                  - self.delta_z(self.pad_z(Omega) * self.bar_z(self.pad_z(theta)), dz)

        R_Phi = - self.bar_z(self.bar_x(u)) * self.delta_x(self.bar_x(self.pad_x(Phi)), dx) \
                - self.bar_z(self.bar_y(v)) * self.delta_y(self.bar_y(self.pad_y(Phi)), dy) \
                - Omega * self.delta_z(self.bar_z(self.pad_z(Phi, Phi_t, Phi_s)), dz)

        R_Mu = - self.delta_x(U, dx) - self.delta_y(V, dy) - self.delta_z(self.pad_z(Omega), dz)

        # pressure term
        R_U += - self.bar_x(self.pad_x(Mu)) * self.bar_x(self.pad_x(alpha)) * self.delta_x(self.pad_x(p), dx) \
               - (self.delta_z(self.bar_x(self.bar_z(self.pad_x(self.pad_z(p, P_t, P_s)))), dz) *
                  self.delta_x(self.pad_x(self.bar_z(self.pad_z(Phi, Phi_t, Phi_s))), dx))

        R_V += - self.bar_y(self.pad_y(Mu)) * self.bar_y(self.pad_y(alpha)) * self.delta_y(self.pad_y(p), dy) \
               - (self.delta_z(self.bar_y(self.bar_z(self.pad_y(self.pad_z(p, P_t, P_s)))), dz) *
                  self.delta_y(self.pad_y(self.bar_z(self.pad_z(Phi, Phi_t, Phi_s))), dy))

        R_W += g * (self.delta_z(p, dz) - self.bar_z(Mu))

        # gravity term
        R_Phi += g * w

        # Coriolis term
        # R_U += + 100 * self.bar_x(self.bar_y(self.pad_x(V))) \
        #        - 100 * self.bar_x(self.bar_z(self.pad_x(self.pad_z(W)))) \
        #        - u * self.bar_x(self.bar_z(self.pad_x(self.pad_z(W)))) / 6400. / 1000.
        # R_V += - 100 * self.bar_x(self.bar_y(self.pad_y(U))) \
        #        + 100 * self.bar_y(self.bar_z(self.pad_y(self.pad_z(W)))) \
        #        - v * self.bar_y(self.bar_z(self.pad_y(self.pad_z(W)))) / 6400. / 1000.

        return R_U, R_V, R_W, R_Theta, R_Mu, R_Phi #, Alpha, Omega

    # def RK3_step(self, U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, dt):
    def RK3_step(self, U, V, W, Theta, Mu, Phi, dt, Phi_t, Phi_s, P_t, P_s, dx, dy, dz, PREF, Rd, g):
        r"""One RK3 Step"""
        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, P_t, P_s, dx, dy, dz, PREF, Rd, g)
        U_ = U + dt * R_U / 3
        V_ = V + dt * R_V / 3
        W_ = W + dt * R_W / 3
        Theta_ = Theta + dt * R_Theta / 3
        Mu_ = Mu + dt * R_Mu / 3
        Phi_ = Phi + dt * R_Phi / 3

        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U_, V_, W_, Theta_, Mu_, Phi_, Phi_t, Phi_s, P_t, P_s, dx, dy, dz, PREF, Rd, g)
        U_ = U + dt * R_U / 2
        V_ = V + dt * R_V / 2
        W_ = W + dt * R_W / 2
        Theta_ = Theta + dt * R_Theta / 2
        Mu_ = Mu + dt * R_Mu / 2
        Phi_ = Phi + dt * R_Phi / 2

        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U_, V_, W_, Theta_, Mu_, Phi_, Phi_t, Phi_s, P_t, P_s, dx, dy, dz, PREF, Rd, g)
        U += dt * R_U
        V += dt * R_V
        W += dt * R_W
        Theta += dt * R_Theta
        Mu += dt * R_Mu
        Phi += dt * R_Phi

        return U, V, W, Theta, Mu, Phi

    def pad_x(self, X):
        r"""Periodic boundary condition in x axis"""
        return F.pad(X, (1, 1), "circular")

    def pad_y(self, X):
        r"""Periodic boundary condition in y axis"""
        Nz, Ny, Nx = X.shape
        return F.pad(X.view(1, Nz, Ny, Nx), (0, 0, 1, 1), "circular").view(Nz, Ny + 2, Nx)

    # TODO def pad_z(self, X, top=None, surface=None):
    def pad_z(self, X, top=torch.Tensor(), surface=torch.Tensor()):
        r"""Dirichlet boundary condition in z axis"""
        _, ny, nx = X.shape
        top = torch.zeros((1, ny, nx), device=X.device) #TODO top = top if top is not None else torch.zeros((1, ny, nx), device=X.device)
        surface = torch.zeros((1, ny, nx), device=X.device) #TODO surface = surface if surface is not None else torch.zeros((1, ny, nx), device=X.device)
        return torch.cat((top, X, surface), dim=0)

    def bar_x(self, X):
        r"""Numerical scheme for X\bar^x

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: X\bar^x with shape (Nz, Ny, Nx-1)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 1, 1, 2)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny, Nx - 1) / 2.

    def delta_x(self, X, dx):
        r"""Numerical scheme for \delta_x X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_x X with shape (Nz, Ny, Nx-1)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 1, 2)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny, Nx - 1) / dx

    def bar_y(self, X):
        r"""Numerical scheme for X\bar^y

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: X\bar^y with shape (Nz, Ny-1, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny - 1, Nx) / 2.

    def delta_y(self, X, dy):
        r"""Numerical scheme for \delta_y X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_y X with shape (Nz, Ny-1, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny - 1, Nx) / dy

    def bar_z(self, X):
        r"""Numerical scheme for X\bar^z

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: X\bar^z with shape (Nz-1, Ny, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz - 1, Ny, Nx) / 2.

    def delta_z(self, X, dz):
        r"""Numerical scheme for \delta_z X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_z X with shape (Nz-1, Ny, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz - 1, Ny, Nx) / dz

    def _acoustic_step(self, ):
        r"""One acustic step"""
        pass


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

if __name__ == "__main__":
    cube.init()

    # simulation settings
    nx = 201
    ny = 201
    nz = 201
    dx = 1e3  # m
    dy = 1e3  # m
    dz = 1. / (nz + 1)
    # constants
    PREF = torch.tensor(1e5)  # reference pressure, usually sea level pressure, Pa
    Rd = torch.tensor(287)  # gas constant for dry air, J/(kg*K)
    g = torch.tensor(9.81)  # the acceleration of gravity, m/s**2

    x0 = 100e3
    y0 = 100e3
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 200e3, 201), torch.linspace(0, 200e3, 201))
    # 100K
    theta = torch.linspace(0, 1, nz + 2).view(nz + 2, 1, 1) * torch.ones((1, ny, nx)) * 600. + 300
    theta += torch.linspace(1, 0, nz + 2).view(nz + 2, 1, 1) * \
        -100. * torch.exp(-0.5 * ((grid_x - x0)**2 + (grid_y - y0)**2) / 400e6).view(1, ny, nx)
    # u0 = torch.ones((nz, ny, nx + 1)).cuda()
    # wrf = WRF(dx, dy, nx, ny, nz, theta.cuda())
    theta = theta.cuda()

    dt = torch.tensor(0.1)
    # nx = torch.tensor(nx)
    # ny = torch.tensor(ny)
    # nz = torch.tensor(nz)
    dx = torch.tensor(dx)
    dy = torch.tensor(dy)
    dz = torch.tensor(dz)

    U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, theta_t, theta_s, P_t, P_s = init(nx, ny, nz, dx, dy, dz, theta, PREF, Rd, g)

    varloader = LoopVariables(variables=[U, V, W, Theta, Mu, Phi, dt], constants=[Phi_t, Phi_s, theta_t, theta_s, P_t, P_s, dx, dy, dz, PREF, Rd, g])
    model = WRF()
    model = cube.SemanticModel(model, input_shapes=tuple(varloader.shapes), )

    #TODO @cube.compile(model=model, dataloader=varloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        U, V, W, Theta, Mu, Phi, dt, Phi_t, Phi_s, theta_t, theta_s, P_t, P_s, dx, dy, dz, PREF, Rd, g = next(dataloader)
        U, V, W, Theta, Mu, Phi = model(U, V, W, Theta, Mu, Phi, Phi_t, Phi_s, theta_t, theta_s, P_t, P_s, dx, dy, dz, dt, PREF, Rd, g)
        return U, V, W, Theta, Mu, Phi
    #TODO model = model.get_gen_module()

    import matplotlib.pyplot as plt
    import numpy as np

    for iter in range (3): # while True:
        # plt.cla()
        # cf = plt.contourf(wrf.Theta[:, 100, :].cpu().numpy(), levels=50, cmap='jet')
        # cb = plt.colorbar(cf)
        # plt.savefig('res.jpeg', dpi=300)
        # plt.clf()
        # input('stop')

        # for i in range(1):
        print("iter-{}...".format(iter))
        U, V, W, Theta, Mu, Phi = train_iter(model, varloader) # Phi_t, Phi_s, theta_t, theta_s

