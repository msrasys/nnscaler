import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch import nn
import torch.nn.functional as F
# from linalg import tridiagonal


class WRF(torch.nn.Module):
    r"""WRF Model

    Args:
        theta (Tensor): inital potential temperature, (nz + 2, ny, nx), including boundary condition
        p_t (Tensor): pressure at model top, (ny, nx), if None all zeros
        p_s (Tensor): pressure at surface, (ny, nx), if None all sea level pressure
        u0 (Tensor): inital x flow, (nz, ny, nx + 1), if None all zeros, default to None.
        v0 (Tensor): inital y flow, (nz, ny + 1, nx), if None all zeros, detault to None.
        w0 (Tensor): inital z flow, (nz - 1, ny, nx), if None all zeros, default to None.
    """

    def __init__(self, dx, dy, nx, ny, nz, theta, p_t=None, p_s=None, u0=None, v0=None, w0=None, device='cuda'):
        super().__init__()
        self.device = device

        # constants
        self.PREF = 1e5  # reference pressure, usually sea level pressure, Pa
        self.Rd = 287  # gas constant for dry air, J/(kg*K)
        self.g = 9.81  # the acceleration of gravity, m/s**2

        # spatial discretization
        self.dx, self.dy, self.dz, self.nx, self.ny, self.nz = dx, dy, 1. / (nz + 1), nx, ny, nz

        # agnostic variables
        self.P_t = p_t if p_t else torch.ones((1, ny, nx), device=device) * self.PREF * 0.0
        self.P_s = p_s if p_s else torch.ones((1, ny, nx), device=device) * self.PREF
        # pressure (nz, ny, nx)
        self.P = torch.linspace(self.dz, 1 - self.dz, nz, device=device).view(nz, 1, 1) * \
            (self.P_s - self.P_t).view(1, ny, nx) + self.P_t
        # Alpha (nz, ny, nx)
        self.Alpha = self.Rd / self.PREF * theta[1:-1] * (self.P / self.PREF)**(-1/1.4)

        # prognostic variables
        # Mu (nz, ny, nx)
        self.Mu = torch.ones((nz, 1, 1), device=device) * (self.P_s - self.P_t).view(1, ny, nx)
        # self.Mu_t = (self.P_s - self.P_t).view(1, ny, nx)
        # self.Mu_s = (self.P_s - self.P_t).view(1, ny, nx)
        # Phi (nz - 1, ny, nx)
        Phi = torch.zeros((nz + 1, ny, nx), device=device)
        Phi[:-1] = self.Mu * self.Alpha * self.dz
        for i in range(nz - 1, -1, -1):
            Phi[i] += Phi[i + 1]
        self.Phi_t = Phi[0].view(1, ny, nx)
        self.Phi_s = Phi[-1].view(1, ny, nx)
        self.Phi = Phi[1:-1]
        # Theta (nz, ny, nx)
        self.theta_t = theta[0].view(1, ny, nx)
        self.theta_s = theta[-1].view(1, ny, nx)
        self.Theta = theta[1:-1] * self.Mu
        # U (nz, ny, nx + 1)
        self.U = u0 if u0 is not None else torch.zeros((nz, ny, nx + 1), device=device)
        # V (nz, ny + 1, nx)
        self.V = v0 if v0 is not None else torch.zeros((nz, ny + 1, nx), device=device)
        # W (nz - 1, ny, nx)
        self.W = w0 if w0 is not None else torch.zeros((nz - 1, ny, nx), device=device)

    def RHS(self, U, V, W, Theta, Mu, Phi):
        # volecity
        u = U / self.bar_x(self.pad_x(Mu))
        v = V / self.bar_y(self.pad_y(Mu))
        w = W / self.bar_z(Mu)
        alpha = -self.delta_z(self.pad_z(Phi, self.Phi_t, self.Phi_s)) / Mu
        self.Alpha = alpha
        theta = Theta / Mu
        p = self.PREF * (self.Rd * theta / self.PREF / alpha)**1.4
        omega = -w * self.g / self.bar_z(alpha) / self.bar_z(Mu)
        Omega = omega * self.bar_z(Mu)
        self.Omega = Omega

        # advection term
        R_U = - self.delta_x(self.bar_x(self.pad_x(U)) * self.bar_x(self.pad_x(u))) \
              - self.delta_y(self.bar_x(self.pad_x(V)) * self.bar_y(self.pad_y(u))) \
              - self.delta_z(self.bar_x(self.pad_x(self.pad_z(Omega))) * self.bar_z(self.pad_z(u)))

        R_V = - self.delta_x(self.bar_y(self.pad_y(U)) * self.bar_x(self.pad_x(v))) \
              - self.delta_y(self.bar_y(self.pad_y(V)) * self.bar_y(self.pad_y(v))) \
              - self.delta_z(self.bar_y(self.pad_y(self.pad_z(Omega))) * self.bar_z(self.pad_z(v)))

        R_W = - self.delta_x(self.bar_z(U) * self.bar_x(self.pad_x(w))) \
              - self.delta_y(self.bar_z(V) * self.bar_y(self.pad_y(w))) \
              - self.delta_z(self.bar_z(self.pad_z(Omega)) * self.bar_z(self.pad_z(w)))

        R_Theta = - self.delta_x(U * self.bar_x(self.pad_x(theta))) \
                  - self.delta_y(V * self.bar_y(self.pad_y(theta))) \
                  - self.delta_z(self.pad_z(Omega) * self.bar_z(self.pad_z(theta)))

        R_Phi = - self.bar_z(self.bar_x(u)) * self.delta_x(self.bar_x(self.pad_x(Phi))) \
                - self.bar_z(self.bar_y(v)) * self.delta_y(self.bar_y(self.pad_y(Phi))) \
                - omega * self.delta_z(self.bar_z(self.pad_z(Phi, self.Phi_t, self.Phi_s)))

        R_Mu = - self.delta_x(U) - self.delta_y(V) - self.delta_z(self.pad_z(Omega))

        # pressure term
        R_U += - self.bar_x(self.pad_x(Mu)) * self.bar_x(self.pad_x(alpha)) * self.delta_x(self.pad_x(p)) \
               - (self.delta_z(self.bar_x(self.bar_z(self.pad_x(self.pad_z(p, self.P_t, self.P_s))))) *
                  self.delta_x(self.pad_x(self.bar_z(self.pad_z(Phi, self.Phi_t, self.Phi_s)))))

        R_V += - self.bar_y(self.pad_y(Mu)) * self.bar_y(self.pad_y(alpha)) * self.delta_y(self.pad_y(p)) \
               - (self.delta_z(self.bar_y(self.bar_z(self.pad_y(self.pad_z(p, self.P_t, self.P_s))))) *
                  self.delta_y(self.pad_y(self.bar_z(self.pad_z(Phi, self.Phi_t, self.Phi_s)))))

        R_W += self.g * (self.delta_z(p) - self.bar_z(Mu))

        # gravity term
        R_Phi += self.g * w

        # Coriolis term
        # R_U += + 100 * self.bar_x(self.bar_y(self.pad_x(V))) \
        #        - 100 * self.bar_x(self.bar_z(self.pad_x(self.pad_z(W)))) \
        #        - u * self.bar_x(self.bar_z(self.pad_x(self.pad_z(W)))) / 6400. / 1000.
        # R_V += - 100 * self.bar_x(self.bar_y(self.pad_y(U))) \
        #        + 100 * self.bar_y(self.bar_z(self.pad_y(self.pad_z(W)))) \
        #        - v * self.bar_y(self.bar_z(self.pad_y(self.pad_z(W)))) / 6400. / 1000.

        return R_U, R_V, R_W, R_Theta, R_Mu, R_Phi,

    def RK3_step(self, U, V, W, Theta, Mu, Phi, dt):
        r"""One RK3 Step"""
        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U, V, W, Theta, Mu, Phi)
        U_ = U + dt * R_U / 3
        V_ = V + dt * R_V / 3
        W_ = W + dt * R_W / 3
        Theta_ = Theta + dt * R_Theta / 3
        Mu_ = Mu + dt * R_Mu / 3
        Phi_ = Phi + dt * R_Phi / 3

        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U_, V_, W_, Theta_, Mu_, Phi_)
        U_ = U + dt * R_U / 2
        V_ = V + dt * R_V / 2
        W_ = W + dt * R_W / 2
        Theta_ = Theta + dt * R_Theta / 2
        Mu_ = Mu + dt * R_Mu / 2
        Phi_ = Phi + dt * R_Phi / 2

        R_U, R_V, R_W, R_Theta, R_Mu, R_Phi = self.RHS(U_, V_, W_, Theta_, Mu_, Phi_)
        U += dt * R_U
        V += dt * R_V
        W += dt * R_W
        Theta += dt * R_Theta
        Mu += dt * R_Mu
        Phi += dt * R_Phi

        return U, V, W, Theta, Mu, Phi

    def forward(self, dt):
        self.U, self.V, self.W, self.Theta, self.Mu, self.Phi = \
            self.RK3_step(self.U, self.V, self.W, self.Theta, self.Mu, self.Phi, dt)

    def pad_x(self, X):
        r"""Periodic boundary condition in x axis"""
        return F.pad(X, (1, 1), "circular")

    def pad_y(self, X):
        r"""Periodic boundary condition in y axis"""
        Nz, Ny, Nx = X.shape
        return F.pad(X.view(1, Nz, Ny, Nx), (0, 0, 1, 1), "circular").view(Nz, Ny + 2, Nx)

    def pad_z(self, X, top=None, surface=None):
        r"""Dirichlet boundary condition in z axis"""
        _, ny, nx = X.shape
        top = top if top is not None else torch.zeros((1, ny, nx), device=X.device)
        surface = surface if surface is not None else torch.zeros((1, ny, nx), device=X.device)
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

    def delta_x(self, X):
        r"""Numerical scheme for \delta_x X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_x X with shape (Nz, Ny, Nx-1)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 1, 2)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny, Nx - 1) / self.dx

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

    def delta_y(self, X):
        r"""Numerical scheme for \delta_y X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_y X with shape (Nz, Ny-1, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz, Ny - 1, Nx) / self.dy

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

    def delta_z(self, X):
        r"""Numerical scheme for \delta_z X

        Args:
            X (Tensor): shape (Nz, Ny, Nx)

        Returns:
            Tensor: \delta_z X with shape (Nz-1, Ny, Nx)
        """
        Nz, Ny, Nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, Nz, Ny, Nx), filter).view(Nz - 1, Ny, Nx) / self.dz

    def _acoustic_step(self, ):
        r"""One acustic step"""
        pass


if __name__ == "__main__":
    # simulation settings
    nx = 201
    ny = 201
    nz = 201
    dx = 1e3   # m
    dy = 1e3  # m

    x0 = 100e3
    y0 = 100e3
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 200e3, 201), torch.linspace(0, 200e3, 201))
    # 100K
    theta = torch.linspace(0, 1, nz + 2).view(nz + 2, 1, 1) * torch.ones((1, ny, nx)) * 600. + 300
    theta += torch.linspace(1, 0, nz + 2).view(nz + 2, 1, 1) * \
        -100. * torch.exp(-0.5 * ((grid_x - x0)**2 + (grid_y - y0)**2) / 400e6).view(1, ny, nx)
    # u0 = torch.ones((nz, ny, nx + 1)).cuda()
    wrf = WRF(dx, dy, nx, ny, nz, theta.cuda())

    import matplotlib.pyplot as plt
    import numpy as np

    while True:
        plt.cla()
        cf = plt.contourf(wrf.Theta[:, 100, :].cpu().numpy(), levels=50, cmap='jet')
        cb = plt.colorbar(cf)
        plt.savefig('res.jpeg', dpi=300)
        plt.clf()
        input('stop')

        for i in range(1):
            wrf(0.1)
