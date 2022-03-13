import math
import torch
import numpy as np
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)


from typing import List
import cube
from cube.runtime.syndata import SciLoopVariables
from examples.atmosphere.policy.naive import PAS

from einops.layers.torch import Rearrange

class Atmoshpere(torch.nn.Module):
    def __init__(self,
                 nz, ny, nx, dy, dx, x0, y0, deltaA, Y, f, sigma, P_, P, phi, zs, w,
                 bar_x_filter, bar_y_filter, bar_z_filter,
                 bar_x2_filter, bar_y2_filter, bar_xy_filter,
                 delta_x_filter, delta_y_filter, delta_z_filter, delta_D_filter, delta_E_filter, laplas_filter,
                 device='cuda'):
        super().__init__()
        #self.device = torch.device(device)

        # physics constant
        self.g = 9.8  # acceleration of gravity, unit in m/s^2
        self.PSEA = 101325.  # sea level pressure, unit in Pa
        self.KAPPA = 0.286  # dimensionless
        self.RE = 6.4e6  # radius of earth, unit in m
        self.CPD = 1004.67  # specific heat of dry air at constant pressure J*kg^-1*K^-1
        # self.OMEGA = 7.292e-5  # angular speed of the Earth s^-1
        self.OMEGA = 1e-1  # angular speed of the Earth s^-1

        # simulation domain
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = 1. / nz
        self.x0 = x0
        self.y0 = y0

        self.deltaA = deltaA
        self.Y = Y
        self.f = f
        self.sigma = sigma
        self.P_ = P_
        self.P = P
        self.phi = phi
        self.zs = zs
        self.w = w

        self.bar_x_filter = bar_x_filter
        self.bar_y_filter = bar_y_filter
        self.delta_x_filter = delta_x_filter
        self.delta_y_filter = delta_y_filter
        self.bar_y2_filter = bar_y2_filter
        self.bar_x2_filter = bar_x2_filter
        self.bar_z_filter = bar_z_filter
        self.delta_z_filter = delta_z_filter
        self.delta_E_filter = delta_E_filter
        self.laplas_filter = laplas_filter
        self.bar_xy_filter = bar_xy_filter
        self.delta_D_filter = delta_D_filter

        self.pre_conv3d_reshape = Rearrange('(b0 b1 Nz) Ny Nx -> b0 b1 Nz Ny Nx', b0=1, b1=1) # x = X.view(1, 1, Nz, Ny, Nx)
        self.post_conv3d_reshape = Rearrange('b0 b1 Nz Ny Nx -> (b0 b1 Nz) Ny Nx')


    def step(self, dt, pi, theta, u, v, pi0, theta0, u0, v0):
        # flux
        F = self.bar_x(self.pad_x(pi)) * 0.5 * u * self.RE * self.dy  # (nz, ny, nx + 1)
        G = self.bar_y(self.pad_y(pi)) * 0.5 * v * self.RE * self.dx * torch.cos(self.Y)  # (nz, ny + 1, nx)
        B = self.bar_y2(self.bar_x(self.pad_y(F))) / 12.  # (nz, ny, nx)
        C = self.bar_y2(self.bar_x(self.pad_y(self.pad_x(G)))) / 12.  # (nz, ny + 1, nx + 1)
        D = self.bar_y2(self.pad_y(G)) + self.bar_y(self.bar_x(self.pad_y(F))) / 24.  # (nx, ny + 1, nx)
        E = self.bar_y2(self.pad_y(G)) - self.bar_y(self.bar_x(self.pad_y(F))) / 24.  # (nx, ny + 1, nx)
        Q = self.bar_x2(self.bar_y(self.pad_x(self.pad_y(F)))) / 12.  # (nz, ny + 1, nx + 1)
        R = self.bar_x2(self.bar_y(self.pad_x(G))) / 12.  # (nz, ny, nx)
        S = self.bar_y(self.bar_x(self.pad_x(G))) + self.bar_x2(self.pad_x(F)) / 24.  # (nz, ny, nx + 1)
        T = self.bar_y(self.bar_x(self.pad_x(G))) - self.bar_x2(self.pad_x(F)) / 24.  # (nz, ny, nx + 1)

        pi1 = pi0 - dt / self.deltaA * ((self.delta_x(F) + self.delta_y(G)) * self.dz).sum(dim=0) #sum(axis=0)  # (nz, ny, nx)
        # print('pi:', pi1.mean())

        # TODO a custom Op needed
        # # update diagnostic variable w (nz + 1, ny, nx)
        # for i in range(1, self.nz + 1):
        #     self.w[i] = - ((self.delta_x(F[:i]) + self.delta_y(G[:i])) * self.dz).sum(dim=0) / self.deltaA / pi1 \
        #         - self.sigma[i] * (pi1 - pi0) / dt / pi1

        # print('w:', self.w.mean())

        # update potential temperature theta (nz, ny, nx)
        # theta_ = self.pad_z(
        #     (self.bar_z(self.P * theta) - self.delta_z(theta) * self.P_[1:-1]) / self.delta_z(self.P)
        # )  # (nz + 1, ny, nx)
        theta_ = theta0 #TODO remove me
        theta1 = pi0 / pi1 * theta0 + dt / self.deltaA / pi1 * (
            (self.delta_x(F * self.bar_x(self.pad_x(theta))) + self.delta_y(G * self.bar_y(self.pad_y(theta)))) / 2. +
            pi * self.deltaA * self.delta_z(self.w * theta_) / self.dz +
            10e8 * self.laplas(self.pad_z(self.pad_y(self.pad_x(theta))))
        )

        # print('theta:', theta1.mean())

        # TODO a custom Op needed
        # # update geopotential
        # self.phi[-1] = self.g * self.zs - self.CPD * (self.P[-1] - self.P_[-1]) * theta[-1]
        # for i in range(1, self.nz):
        #     tmp = self.phi[-i] - self.CPD * (self.P_[-i - 1] - self.P[-i]) * theta[-i]
        #     self.phi[-1 - i] = tmp - self.CPD * (self.P[-1 - i] - self.P_[-1 - i]) * theta[-1 - i]

        # print('phi:', self.phi.mean())

        # update u (nz, ny, nx + 1)
        pi0_deltaA = self.bar_y2(self.bar_x(self.pad_y(self.pad_x(pi0 * self.deltaA)))) / 8.  # (nz, ny, nx + 1)
        pi1_deltaA = self.bar_y2(self.bar_x(self.pad_y(self.pad_x(pi1 * self.deltaA)))) / 8.  # (nz, ny, nx + 1)
        pi_w_deltaA = self.bar_y2(self.bar_x(self.pad_y(self.pad_x(pi * self.deltaA * self.w)))) / 8.  # (nz + 1, ny, nx + 1)
        advec = (
            - self.delta_x(self.pad_x(B * self.bar_x(u)))
            - self.delta_y(C * self.bar_y(self.pad_y(u)))
            + self.delta_D(self.pad_x(D * self.bar_xy(self.pad_y(u))))
            + self.delta_E(self.pad_x(E * self.bar_xy(self.pad_y(u))))
        ) / 2.
        trans = - self.delta_z(pi_w_deltaA * self.bar_z(self.pad_z(u)) * 0.5) / self.dz
        #TODO fixme press = - self.RE * self.dy * (
        press = self.dy * (
            self.delta_x(self.pad_x(self.phi)) * self.delta_x(self.pad_x(pi)) / 2. +
            self.delta_x(self.pad_x(pi)) * 0.5 * self.CPD * self.bar_x(self.pad_x(
                theta / self.dz * (self.delta_z(self.sigma * self.P_) - self.P * self.delta_z(self.sigma))
            ))
        )
        diff = 10e8 * self.laplas(self.pad_z(self.pad_y(self.pad_x(u))))
        #TODO fixme cori = self.RE * self.dx * self.dy * 0.25 * (
        cori = self.dy * (
            self.bar_x(self.pad_x(pi * self.bar_y(v) * (self.f + 0. * self.bar_x(u) * torch.sin(self.bar_y(self.Y)))))
        ) * 0.0
        u1 = (pi0_deltaA * u0 + dt * (advec + trans + press + diff + cori)) / pi1_deltaA
        # print('u1:', u1.mean())

        # # update v (nz, ny + 1, nx)
        pi0_deltaA = self.bar_x2(self.bar_y(self.pad_x(self.pad_y(pi0 * self.deltaA)))) / 8.  # (nz, ny + 1, nx)
        pi1_deltaA = self.bar_x2(self.bar_y(self.pad_x(self.pad_y(pi1 * self.deltaA)))) / 8.  # (nz, ny + 1, nx)
        pi_w_deltaA = self.bar_x2(self.bar_y(self.pad_x(self.pad_y(pi * self.deltaA * self.w)))) / 8.  # (nz + 1, ny + 1, nx)
        advec = (
            - self.delta_x(Q * self.bar_x(self.pad_x(v)))
            - self.delta_y(self.pad_y(R * self.bar_y(v)))
            + self.delta_D(self.pad_y(S * self.bar_xy(self.pad_x(v))))
            + self.delta_E(self.pad_y(T * self.bar_xy(self.pad_x(v))))
        ) / 2.
        trans = - self.delta_z(pi_w_deltaA * self.bar_z(self.pad_z(v)) * 0.5) / self.dz
        #TODO fixme press = - self.RE * self.dx * (
        press = self.dx * (
            self.delta_y(self.pad_y(self.phi)) * self.delta_y(self.pad_y(pi)) / 2. +
            self.delta_y(self.pad_y(pi)) * 0.5 * self.CPD * self.bar_y(self.pad_y(
                theta / self.dz * (self.delta_z(self.sigma * self.P_) - self.P * self.delta_z(self.sigma))
            ))
        )
        diff = 10e8 * self.laplas(self.pad_z(self.pad_y(self.pad_x(v))))
        #TODO fixme cori = - self.RE * self.dx * self.dy * 0.25 * (
        cori = self.dy * (
            self.bar_y(self.pad_y(pi * self.bar_x(u) * (self.f + self.bar_x(u) * torch.sin(self.bar_y(self.Y)))))
        ) * 0.0
        v1 = (pi0_deltaA * v0 + dt * (advec + trans + press + diff + cori)) / pi1_deltaA
        # # print('v1:', v1.mean())

        return pi1, theta1, u1, v1


    def forward(self, pi, theta, u, v, dt):
        pi_, theta_, u_, v_ = self.step(dt / 2., pi, theta, u, v, pi, theta, u, v)
        pi1, theta1, u1, v1 = self.step(dt, pi_, theta_, u_, v_, pi, theta, u, v)
        return pi1, theta1, u1, v1


    def pad_x(self, X):
        return F.pad(X, (1, 1), "circular")


    def bar_x(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_x_filter))


    def bar_x2(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_x2_filter))


    def delta_x(self, X):
        return  self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.delta_x_filter))


    def pad_y(self, X):
        #TODO check return F.pad(X.view(1, nz, ny, nx), (0, 0, 1, 1), "circular").view(nz, ny + 2, nx)
        return F.pad(X, (0, 0, 1, 1), "circular")


    def bar_y(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_y_filter))


    def bar_y2(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_y2_filter))


    def delta_y(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.delta_y_filter))


    def bar_z(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_z_filter))


    def pad_z(self, X):
        # return F.pad(X.view(1, 1, nz, ny, nx), (0, 0, 0, 0, 1, 1)).view(nz + 2, ny, nx)
        return F.pad(X, (0, 0, 0, 0, 1, 1))


    def delta_z(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.delta_z_filter))


    def delta_D(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.delta_D_filter))


    def delta_E(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.delta_E_filter))


    def bar_xy(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.bar_xy_filter))


    def laplas(self, X):
        return self.post_conv3d_reshape(F.conv3d(self.pre_conv3d_reshape(X), self.laplas_filter))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cube.init()

    nz = 15
    ny = 100
    nx = 100
    dy = 1e-4
    dx = 1e-4
    x0 = 0.0
    y0 = 0.2

    PSEA = 101325.  # sea level pressure, unit in Pa
    RE = 6.4e6  # radius of earth, unit in m

    xc = nx * dx / 2 + x0
    yc = ny * dy / 2 + y0
    X = torch.linspace(0, nx - 1, nx).view(1, 1, nx) * dx + x0
    Y = torch.linspace(0, ny - 1, ny).view(1, ny, 1) * dy + y0
    ps = torch.ones((1, ny, nx)) * PSEA - 300 * torch.exp(
        - 1e-6 * ((RE * torch.cos((Y + yc) / 2)) * (X - xc))**2
        - 1e-6 * (RE * (Y - yc))**2)
    pt = 250e2
    zs = torch.zeros((ny, nx)) + 10000 * torch.exp(
        - 1e-6 * (RE * (X - nx * dx / 3 - x0))**2
        - 1e-6 * (RE * (Y - yc))**2)

    u = torch.zeros((nz, ny, nx + 1))
    v = torch.zeros((nz, ny + 1, nx))

    dt = torch.tensor(1.)

    # physics constant
    g = 9.8  # acceleration of gravity, unit in m/s^2
    PSEA = 101325.  # sea level pressure, unit in Pa
    KAPPA = 0.286  # dimensionless
    RE = 6.4e6  # radius of earth, unit in m
    CPD = 1004.67  # specific heat of dry air at constant pressure J*kg^-1*K^-1
    # OMEGA = 7.292e-5  # angular speed of the Earth s^-1
    OMEGA = 1e-1  # angular speed of the Earth s^-1

    # atmoshpere verticle profile
    hight_profile = torch.tensor([
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
        8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
    ]) * 1e3

    pressure_profile = torch.tensor([
        1013.25, 1001.20, 989.45, 977.72, 966.11, 954.61, 943.22, 931.94, 920.77, 909.71, 898.80, 845.59, 795.0,
        746.9, 701.2, 657.8, 616.6, 577.5, 540.5, 505.4, 472.2, 440.7, 411.1, 383.0, 356.5, 331.5, 308.0, 285.8,
        265.0, 227.0, 194.0, 165.8, 141.7, 121.1, 103.5, 88.5, 75.7, 64.7, 55.3, 47.3, 40.5, 34.7, 29.7, 25.5, 21.9,
        18.8, 16.2, 13.9, 12.0, 10.3, 8.89, 7.67, 6.63, 5.75, 4.99, 4.33, 3.77, 3.29, 2.87, 2.51, 2.20, 1.93, 1.69,
        1.49, 1.31, 1.16, 1.02, 0.903, 0.903, 0.425, 0.220, 0.109, 0.0522, 0.0239, 0.0105, 0.0045, 0.0018, 0.00076,
        0.00032
    ]) * 1e2

    temperature_profile = torch.tensor([
        288.15, 287.50, 286.85, 286.20, 285.55, 284.90, 284.25, 283.60, 282.95, 282.30, 281.65, 278.40, 275.15,
        271.91, 268.66, 265.41, 262.17, 258.92, 255.68, 252.43, 249.19, 245.94, 242.70, 239.46, 236.22, 232.97,
        229.73, 226.49, 223.25, 216.78, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65,
        217.58, 218.57, 219.57, 220.56, 221.55, 222.54, 223.54, 224.53, 225.52, 226.51, 227.50, 228.49, 230.97,
        233.74, 236.51, 239.28, 242.05, 244.82, 247.58, 250.35, 253.11, 255.88, 258.64, 261.40, 264.16, 266.93,
        269.68, 270.65, 270.65, 270.65, 260.77, 247.02, 233.29, 219.59, 208.40, 198.64, 188.89, 186.87, 188.42,
        195.08
    ])

    density_profile = torch.tensor([
        1.225, 1.213, 1.202, 1.190, 1.179, 1.167, 1.156, 1.145, 1.134, 1.123, 1.112, 1.058, 1.007, 0.957, 0.909,
        0.863, 0.819, 0.777, 0.736, 0.697, 0.660, 0.624, 0.590, 0.557, 0.526, 0.496, 0.467, 0.440, 0.414, 0.365,
        0.312, 0.267, 0.228, 0.195, 0.166, 0.142, 0.122, 0.104, 0.0889, 0.0757, 0.0645, 0.0550, 0.0469, 0.0401,
        0.0343, 0.0293, 0.0251, 0.0215, 0.0184, 0.0158, 0.0136, 0.0116, 0.00989, 0.00846, 0.00726, 0.00624, 0.00537,
        0.00463, 0.00400, 0.00346, 0.00299, 0.00260, 0.00226, 0.00197, 0.00171, 0.0015, 0.00132, 0.00116, 0.00103,
        5.7e-4, 3.1e-4, 1.6e-4, 8.3e-4, 4.0e-5, 1.8e-5, 8.2e-6, 3.4e-6, 7.5e-7, 5.6e-7
    ])

    def hight_from_pressure(p):
        ind0 = torch.abs((p[None] - pressure_profile[(..., ) + (None, ) * len(p.shape)])).argmin(axis=0)
        ind1 = (p > pressure_profile[ind0]) * 2 - 1 + ind0
        hight = (hight_profile[ind1] - hight_profile[ind0]) * (p - pressure_profile[ind0]) / (
            pressure_profile[ind1] - pressure_profile[ind0]) + hight_profile[ind0]
        return hight

    def pressure_from_hight(z):
        ind0 = torch.abs((z[None] - hight_profile[(..., ) + (None, ) * len(z.shape)])).argmin(axis=0)
        ind1 = (hight_profile[ind0] > z) * 2 - 1 + ind0
        p = (pressure_profile[ind1] - pressure_profile[ind0]) * (z - hight_profile[ind0]) / \
            (hight_profile[ind1] - hight_profile[ind0]) + pressure_profile[ind0]
        return p

    def temperature_from_pressure(p):
        ind0 = torch.abs((p[None] - pressure_profile[(..., ) + (None, ) * len(p.shape)])).argmin(axis=0)
        ind1 = (p > pressure_profile[ind0]) * 2 - 1 + ind0
        T = (temperature_profile[ind1] - temperature_profile[ind0]) * (p - pressure_profile[ind0]) / (
            pressure_profile[ind1] - pressure_profile[ind0]) + temperature_profile[ind0]
        return T

    def bar_y(X):
        nz, ny, nx = X.shape
        filter = torch.tensor([1., 1.]).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz, ny - 1, nx)

    def delta_z(X):
        nz, ny, nx = X.shape
        filter = torch.tensor([-1., 1.]).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz - 1, ny, nx)

    def init(ps, pt, zs):
        Y = (
            torch.linspace(0, ny, ny + 1) * dy + y0
        ).view(1, ny + 1, 1)
        deltaA = RE**2 * torch.cos(bar_y(Y) * 0.5) * dx * dy  # (1, ny, 1)
        f = 2 * OMEGA * torch.sin(bar_y(Y)) * torch.cos(bar_y(Y)) * RE  # (nz, ny, nx)

        # vertical grids
        pt = torch.tensor([pt]).view(1, 1, 1)
        zt = hight_from_pressure(pt)
        z = torch.linspace(1, 0, nz + 1).view(-1, 1, 1) * zt
        p_ = pressure_from_hight(z)
        sigma = (p_ - pt) / (p_[-1] - pt)  # (nz + 1, 1, 1)

        # column pressure, with shape (1, ny, nx)
        pi = (ps - pt).view(1, ny, nx)

        # potential temperature factor
        p_ = pt + sigma * pi  # (nz + 1, ny, nx)
        P_ = (p_ / PSEA)**KAPPA  # (nz + 1, ny, nx)
        P = delta_z(p_ * P_) / delta_z(p_) / (1 + KAPPA)  # (nz, ny, nx)

        # potential temperature
        p = PSEA * P**(1 / KAPPA)
        T = temperature_from_pressure(p)
        theta = T / P

        # geopotential (nz, ny, nx)
        phi = torch.zeros((nz, ny, nx))
        zs = zs

        # vertical velocity
        w = torch.zeros((nz + 1, ny, nx))

        return pi, theta, deltaA, Y, f, sigma, P_, P, phi, zs, w

    pi, theta, deltaA, Y, f, sigma, P_, P, phi, zs, w = init(ps, pt, zs)
    print("[pi, theta, deltaA, Y, f, sigma, P_, P, phi, zs, w]")
    for var in [pi, theta, deltaA, Y, f, sigma, P_, P, phi, zs, w]:
        print(f'shape {var.shape}')

    bar_x_filter = torch.tensor([1., 1.]).view(1, 1, 1, 1, 2)
    bar_y_filter = torch.tensor([1., 1.]).view(1, 1, 1, 2, 1)
    delta_x_filter = torch.tensor([-1., 1.]).view(1, 1, 1, 1, 2)
    delta_y_filter = torch.tensor([-1., 1.]).view(1, 1, 1, 2, 1)
    bar_y2_filter = torch.tensor([1., 2., 1.]).view(1, 1, 1, 3, 1)
    bar_x2_filter = torch.tensor([1., 2., 1.]).view(1, 1, 1, 1, 3)
    bar_z_filter = torch.tensor([1., 1.]).view(1, 1, 2, 1, 1)
    delta_z_filter = torch.tensor([-1., 1.]).view(1, 1, 2, 1, 1)
    delta_E_filter = torch.tensor(
        [[0., 1.],
         [-1., 0.]]
    ).view(1, 1, 1, 2, 2)
    laplas_filter = torch.tensor(
        [[[0., 0., 0.],
          [0., 1., 0.],
          [0., 0., 0.]],
         [[0., 1., 0.],
          [1., -6, 1.],
          [0., 1., 0.]],
         [[0., 0., 0.],
          [0., 1., 0.],
          [0., 0., 0.]]],
    ).view(1, 1, 3, 3, 3)
    bar_xy_filter = torch.tensor(
        [[1., 0.],
         [0., 1.]]
    ).view(1, 1, 1, 2, 2)
    delta_D_filter = torch.tensor(
            [[1., 0.],
             [0., -1.]]
        ).view(1, 1, 1, 2, 2)

    model = Atmoshpere(nz, ny, nx, dy, dx, x0, y0, deltaA, Y, f, sigma, P_, P, phi, zs, w,
                       bar_x_filter, bar_y_filter, bar_z_filter,
                       bar_x2_filter, bar_y2_filter, bar_xy_filter,
                       delta_x_filter, delta_y_filter, delta_z_filter, delta_D_filter, delta_E_filter, laplas_filter)

    print("[pi, theta, u, v, dt]")
    for var in [pi, theta, u, v, dt]:
        print(f'shape {var.shape}')

    varloader = SciLoopVariables(variables=[pi, theta, u, v], constants=[dt])
    model = cube.SemanticModel(model, input_shapes=tuple(varloader.shapes))

    @cube.compile(model=model, dataloader=varloader, PAS=PAS, override=True)
    def train_iter(model, dataloader):
        pi, theta, u, v, dt = next(dataloader)
        pi, theta, u, v = model(pi, theta, u, v, dt)
        return pi, theta, u, v
    model = model.get_gen_module()

    for i in range(3):
        print("iter-{}...".format(i))
        pi, theta, u, v = train_iter(model, varloader)

        # # ctf = plt.contourf(pi.view(ny, nx).numpy(), levels=50, cmap='jet')
        # plt.cla()
        # ct = plt.contour(zs.view(ny, nx).cpu().numpy(), levels=[7000])
        # ctf = plt.contourf(u[3].cpu().numpy(), levels=50, cmap='jet')
        # plt.colorbar(ctf)
        # # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'res2/res{i}.jpeg', dpi=300)
        # plt.clf()

        # print(i)
