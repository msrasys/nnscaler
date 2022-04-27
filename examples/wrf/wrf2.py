import torch
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)


class WRF(torch.nn.Module):
    def __init__(self, dt, ntau, nz, ny, nx, dz, dy, dx, device):
        super().__init__()
        # simulation domain settings
        self.dt = dt
        self.ntau = ntau
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.delta_x = dx
        self.delta_y = dy
        self.delta_z = dz

        # physics constant
        self.g = 9.8            # acceleration of gravity, unit in m/s^2
        self.GAMMA = 1.4        # the ratio of heat capacities for dry air
        self.PREF = 101325.     # sea level pressure, unit in Pa
        self.RD = 287.          # gas constant for dry air J*kg^-1*K^-1
        self.RE = 6.4e6         # radius of earth, unit in m
        self.OMEGA = 7.292e-5   # angular speed of the Earth s^-1

        self.device = torch.device(device)

    def init(self, theta, Ptop=250e2):
        eta = torch.linspace(0, 1, self.nz + 1, device=self.device)
        pi = self.PREF - Ptop
        p0 = Ptop + pi * eta
        self.p0 = ((p0[:-1] + p0[1:]) / 2).view(self.nz, 1, 1) * torch.ones((1, self.ny, self.nx), device=self.device)

        self.mu0 = torch.ones((self.nz, self.ny, self.nx), device=self.device) * pi
        mu1 = torch.zeros((self.nz, self.ny, self.nx), device=self.device)

        self.alpha0 = (self.RD * theta) / self.PREF * (self.p0 / self.PREF)**(-1. / self.GAMMA)

        phi0 = torch.zeros((self.nz + 1, self.ny, self.nx), device=self.device)
        phi1 = torch.zeros((self.nz - 1, self.ny, self.nx), device=self.device)
        phi0[-1] = self.alpha0[-1] * self.mu0[-1]
        for i in range(self.nz - 1, -1, -1):
            phi0[i] = self.alpha0[i] * self.mu0[i] * self.delta_z + phi0[i + 1]
        self.phi0 = phi0[1:-1]  # phi0 with shape (nz - 1, ny, nx)
        self.phit = phi0[0].view(1, self.ny, self.nx)    # model top hight
        self.phis = phi0[-1].view(1, self.ny, self.nx)    # earth surface hight

        self.ztop = (self.phit / self.g).view(1, self.ny, self.nx)

        Theta = theta * self.mu0

        return Theta, phi1, mu1

    def forward(self, U, V, W, O, Theta, phi1, mu1):
        r"""
        Args:
            U (Tensor): (nz, ny, nx - 1)
            V (Tensor): (nz, ny - 1, nx)
            W (Tensor): (nz - 1, ny, nx)
            O (Tensor): (nz - 1, ny, nx)
            Theta (Tensor): (nz, ny, nx)
            phi1 (Tensor): (nz - 1, ny, nx)
            mu1 (Tensor): (nz, ny, nx)
        """
        R_U, R_V, R_W, R_Theta, R_phi, R_mu = self.RHS(U, V, W, O, Theta, phi1, mu1)
        U_, V_, W_, O_, Theta_, phi1_, mu1_ = \
            self.step(self.dt / 3, 1,
                      U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu)

        R_U, R_V, R_W, R_Theta, R_phi, R_mu = self.RHS(U_, V_, W_, O_, Theta_, phi1_, mu1_)
        # U_, V_, W_, O_, Theta_, phi1_, mu1_ = \
        #     self._step(self.dt / 2, self.ntau // 2,
        #               U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu)
        U_, V_, W_, O_, Theta_, phi1_, mu1_ = \
            self.step(self.dt / self.ntau, self.ntau // 2,
                      U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu)

        R_U, R_V, R_W, R_Theta, R_phi, R_mu = self.RHS(U_, V_, W_, O_, Theta_, phi1_, mu1_)
        U, V, W, O, Theta, phi1, mu1 = \
            self.step(self.dt / self.ntau, self.ntau,
                      U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu)
        # U, V, W, O, Theta, phi1, mu1 = \
        #     self._step(self.dt, self.ntau,
        #               U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu)

        # print()
        # print('R_U\t', R_U.abs().min(), R_U.abs().max())
        # print('R_V\t', R_V.abs().min(), R_V.abs().max())
        # print('R_W\t', R_W.abs().min(), R_W.abs().max())
        # print('R_phi\t', R_phi.abs().min(), R_phi.abs().max())
        # print('R_mu\t', R_mu.abs().min(), R_mu.abs().max())
        # print('R_Theta\t', R_Theta.abs().min(), R_Theta.abs().max())

        return U, V, W, O, Theta, phi1, mu1

    def _step(self, dtau, ntau, U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu):
        U += R_U * dtau
        V += R_V * dtau
        W += R_W * dtau
        Theta += R_Theta * dtau
        phi1 += R_phi * dtau
        mu1 += R_mu * dtau

        phi = phi1 + self.phi0
        O = self.g * W / self.dz(self.bz(self.pzphi(phi)))

        return U, V, W, O, Theta, phi1, mu1

    def step(self, dtau, ntau, U, V, W, O, Theta, phi1, mu1, R_U, R_V, R_W, R_Theta, R_phi, R_mu):
        # initialize perturbed varibles
        U2 = torch.zeros(U.shape, device=self.device)
        V2 = torch.zeros(V.shape, device=self.device)
        W2 = torch.zeros(W.shape, device=self.device)
        O2 = torch.zeros(O.shape, device=self.device)
        Theta2 = torch.zeros(Theta.shape, device=self.device)
        phi2 = torch.zeros(phi1.shape, device=self.device)
        mu2 = torch.zeros(mu1.shape, device=self.device)
        pi2 = torch.zeros((self.ny, self.nx), device=self.device)

        phi = self.phi0 + phi1
        mu = self.mu0 + mu1
        alpha = - self.dz(self.pzphi(phi)) / mu
        p = self.PREF * (self.RD * Theta / mu / self.PREF / alpha)**self.GAMMA

        for i in range(ntau):
            U2, V2, W2, O2, Theta2, phi2, mu2, pi2 = \
                self.ac_step(dtau,
                             U2, V2, W2, O2, Theta2, phi2, mu2, pi2,
                             R_U, R_V, R_W, R_Theta, R_phi, R_mu,
                             U, V, Theta, phi, mu, alpha, p)

        return U + U2, V + V2, W + W2, O + O2, Theta + Theta2, phi1 + phi2, mu1 + mu2

    def ac_step(self, dtau,
                U2, V2, W2, O2, Theta2, phi2, mu2, pi2,
                R_U, R_V, R_W, R_Theta, R_phi, R_mu,
                U, V, Theta, phi, mu, alpha, p):
        r"""one acoustic step"""
        # diagnostic variables
        alpha2 = - (self.dz(self.pz(phi2)) + alpha * mu2) / mu
        cs2 = self.GAMMA * p * alpha  # square of sound speed
        C = cs2 / mu / alpha**2
        p2 = self.GAMMA * p * (Theta2 / Theta - alpha2 / alpha - mu2 / mu)
        theta = Theta / mu

        # prognostic variables
        U2_ = U2 + dtau * (
            R_U - self.bx(mu) * (
                self.bx(alpha) * self.dx(p2) +
                self.bx(alpha2) * self.dx(self.p0) +
                self.dx(self.bz(self.pz(phi2)))) -
            self.dx(self.bz(self.pzphi(phi))) * (self.dz(self.bx(self.pzp1(self.bz(p2)))) - self.bx(mu2))
        )
        V2_ = V2 + dtau * (
            R_V - self.by(mu) * (
                self.by(alpha) * self.dy(p2) +
                self.by(alpha2) * self.dy(self.p0) +
                self.dy(self.bz(self.pz(phi2)))) -
            self.dy(self.bz(self.pzphi(phi))) * (self.dz(self.by(self.pzp1(self.bz(p2)))) - self.by(mu2))
        )

        # W2_ = W2 + dtau * R_W
        # O2_ = self.g * W2_ / self.dz(self.bz(self.pzphi(phi)))
        # mu2_ = mu2 + dtau * R_mu

        dpi2 = - (self.dx(self.px(U2_ + U)) + self.dy(self.py(V2_ + V))).sum(0) * self.delta_z
        pi2 = pi2 + dpi2 * dtau

        O2_ = torch.zeros(O2.shape, device=O2.device)
        mu2_ = torch.zeros(mu2.shape, device=mu2.device)
        for i in range(1, O2.shape[0] + 1):
            O2_[-i] = i * self.delta_z * dpi2 + \
                (self.dx(self.px(U2_)) + self.dy(self.py(V2_)) - R_mu)[-i:].view(
                    -1, self.ny, self.nx).sum(0) * self.delta_z
        for i in range(mu2.shape[0]):
            mu2_[i] = pi2

        # self.O2_ = O2_

        Theta2_ = Theta2 + dtau * (
            R_Theta
            - self.dx(self.px(U2_ * self.bx(theta)))
            - self.dy(self.py(V2_ * self.by(theta)))
            - self.dz(self.pz(O2_ * self.bz(theta)))
        )
        # print('Theta2_:\t', Theta2_.min(), Theta2_.max())
        # Theta2_ = torch.zeros(Theta2_.shape, device=Theta2_.device)

        def f(x):
            phi2_ = phi2 + dtau * (
                R_phi - (O2_ * self.dz(self.bz(self.pzphi(phi))) - self.g * (x + W2) * 0.5) / self.bz(mu))
            return (
                R_W + (
                    self.dz(C * self.dz(self.pz(phi2_))) + self.dz(self.GAMMA * p * Theta2_ / Theta) - self.bz(mu2_) +
                    self.dz(C * self.dz(self.pz(phi2))) + self.dz(self.GAMMA * p * Theta2 / Theta) - self.bz(mu2)
                ) * 0.5 * self.g
            ) * dtau + W2 - x

        W2_ = self.solve_tridiagonal(f)
        if torch.abs(f(W2_)).max() > 1e-6:
            print("Triangular solver warning:\t", torch.abs(f(W2_)).max())
        W2_ = W2_ / (1 + self.damping(phi, 0.2, self.ztop * 0.75) * dtau)
        # print((1 + self.damping(phi, 0.8, self.ztop * 0.75) * dtau)[:, 64, 64])

        # W2_ = W2 + dtau * R_W

        phi2_ = phi2 + dtau * (
            R_phi - (O2_ * self.dz(self.bz(self.pzphi(phi))) - self.g * (W2_ + W2) * 0.5) / self.bz(mu))
        # phi2_ = phi2 + dtau * R_phi

        return U2_, V2_, W2_, O2_, Theta2_, phi2_, mu2_, pi2

    def damping(self, phi, gamma, zd):
        z = phi / self.g
        res = gamma * torch.sin(torch.pi / 2 * (1 - (self.ztop - z) / (self.ztop - zd)))**2
        return res * z.gt(zd).double()

    def RHS(self, U, V, W, O, Theta, phi1, mu1):
        mu = self.mu0 + mu1
        phi = self.phi0 + phi1
        alpha = - self.dz(self.pzphi(phi)) / mu
        alpha1 = alpha - self.alpha0
        theta = Theta / mu
        p = self.PREF * (self.RD * theta / self.PREF / alpha)**self.GAMMA
        p1 = p - self.p0

        u = U / self.bx(mu)
        v = V / self.by(mu)
        w = W / self.bz(mu)

        R_U = (
            # pressure term
            - self.bx(mu) * (
                + self.dx(self.bz(self.pz(phi1)))
                + self.bx(alpha) * self.dx(p1)
                + self.bx(alpha1) * self.dx(self.p0))
            - self.dx(self.bz(self.pzphi(phi))) * (self.dz(self.bx(self.pzp1(self.bz(p1)))) - self.bx(mu1))
            # advection term
            - self.dx(self.bx(self.px(U * u)))
            - self.dy(self.bx(self.py(V * self.by(self.bx(self.px(u))))))
            - self.dz(self.bx(self.pz(O * self.bz(self.bx(self.px(u))))))
        )
        R_V = (
            # pressure term
            - self.by(mu) * (
                + self.dy(self.bz(self.pz(phi1)))
                + self.by(alpha) * self.dy(p1)
                + self.by(alpha1) * self.dy(self.p0))
            - self.dy(self.bz(self.pzphi(phi))) * (self.dz(self.by(self.pzp1(self.bz(p1)))) - self.by(mu1))
            # advection term
            - self.dx(self.by(self.px(U * self.bx(self.by(self.py(v))))))
            - self.dy(self.by(self.py(V * v)))
            - self.dz(self.by(self.pz(O * self.bz(self.by(self.py(v))))))
        )
        R_W = (
            # pressure term
            + self.g * (self.dz(p1) - self.bz(self.mu0) * 0.0) - self.bz(mu1) * self.g
            # advection term
            - self.dx(self.px(self.bz(U) * self.bx(w)))
            - self.dy(self.py(self.bz(V) * self.by(w)))
            - self.dz(self.bz(self.pz(O * w)))
        )
        R_Theta = (
            - self.dx(self.px(U * self.bx(theta)))
            - self.dy(self.py(V * self.by(theta)))
            - self.dz(self.pz(O * self.bz(theta)))
        )
        R_phi = (
            # advection term
            - self.bx(self.px(self.bz(U) * self.dx(phi)))
            - self.by(self.py(self.bz(V) * self.dy(phi)))
            - O * self.dz(self.bz(self.pzphi(phi)))
            # gravity term
            + self.g * W
        ) / self.bz(mu)
        R_mu = (
            # advection term
            - self.dx(self.px(U))
            - self.dy(self.py(V))
            - self.dz(self.pz(O))
        )

        return R_U, R_V, R_W, R_Theta, R_phi, R_mu

    def dx(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 1, 2)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz, ny, nx - 1) / self.delta_x

    def dy(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz, ny - 1, nx) / self.delta_y

    def dz(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([-1., 1.], device=X.device).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz - 1, ny, nx) / self.delta_z

    def px(self, X):
        return F.pad(X, (1, 1), "circular")

    def py(self, X):
        nz, ny, nx = X.shape
        return F.pad(X.view(1, nz, ny, nx), (0, 0, 1, 1), "constant").view(nz, ny + 2, nx)

    def pz(self, X):
        nz, ny, nx = X.shape
        return F.pad(X.view(1, 1, nz, ny, nx), (0, 0, 0, 0, 1, 1), "constant").view(nz + 2, ny, nx)

    def pzphi(self, X):
        """pad phi in z axis"""
        return torch.cat((self.phit, X, self.phis), 0)

    def pzp1(self, X):
        """pad p1 in z axis"""
        nz, ny, nx = X.shape
        p1t = torch.zeros((1, ny, nx), device=X.device)
        p1s = X[-1].view(1, ny, nx)
        return torch.cat((p1t, X, p1s), 0)

    def bx(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 1, 1, 2)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz, ny, nx - 1) / 2.

    def by(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 1, 2, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz, ny - 1, nx) / 2.

    def bz(self, X):
        nz, ny, nx = X.shape
        filter = torch.tensor([1., 1.], device=X.device).view(1, 1, 2, 1, 1)
        return F.conv3d(X.view(1, 1, nz, ny, nx), filter).view(nz - 1, ny, nx) / 2.

    def solve_tridiagonal(self, f):
        r"""Solve tridiagonal system f(x) = Ax - b = 0

        Args:
            f (Callable, return Tensor): Tridiagonal system (nz - 1, ny, nx) -> (nz - 1, ny, nx)

        Returns:
            Tensor: Solution of the linear system with shape (D, H, W)
        """
        b = - f(torch.zeros((self.nz - 1, self.ny, self.nx), device=self.device))

        idx0 = torch.tensor([1., 0, 0], device=self.device).view(3, 1, 1)
        idx0 = idx0.repeat((self.nz - 1) // 3 + 1, self.ny, self.nx)[:self.nz - 1]
        r0 = f(idx0) + b

        idx1 = torch.tensor([0., 1, 0], device=self.device).view(3, 1, 1)
        idx1 = idx1.repeat((self.nz - 1) // 3 + 1, self.ny, self.nx)[:self.nz - 1]
        r1 = f(idx1) + b

        idx2 = torch.tensor([0., 0, 1], device=self.device).view(3, 1, 1)
        idx2 = idx2.repeat((self.nz - 1) // 3 + 1, self.ny, self.nx)[:self.nz - 1]
        r2 = f(idx2) + b

        d = (torch.stack([r0, r1, r2], 1) * torch.stack([idx0, idx1, idx2], 1)).sum(1)
        l = (torch.stack([r2, r0, r1], 1) * torch.stack([idx0, idx1, idx2], 1)).sum(1)[1:]
        u = (torch.stack([r1, r2, r0], 1) * torch.stack([idx0, idx1, idx2], 1)).sum(1)[:-1]

        # forward sweep
        for i in range(1, d.shape[0]):
            w = l[i - 1] / d[i - 1]
            d[i] = d[i] - w * u[i - 1]
            b[i] = b[i] - w * b[i - 1]

        # backward substitution
        x = torch.zeros(b.shape, device=b.device)
        x[-1] = b[-1] / d[-1]
        for i in range(x.shape[0] - 2, -1, -1):
            x[i] = (b[i] - u[i] * x[i + 1]) / d[i]

        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    nz = 16
    dz = 1. / 16
    ny = 128
    dy = 1e3
    nx = 128
    dx = 1e3

    dt = 1.

    x = torch.linspace(-1., 1., 128).cuda()
    y = torch.linspace(-1., 1., 128).cuda()
    theta0 = (torch.linspace(1, 0, nz).cuda() * 500 + 300).view(nz, 1, 1) * torch.ones((1, ny, nx)).cuda()
    theta1 = torch.exp(-0.5 * (x / 0.1)**2).view(1, 1, nx) * torch.exp(-0.5 * (y / 0.1)**2).view(1, ny, 1) * 0.01
    wrf = WRF(dt, 10, nz, ny, nx, dz, dy, dx, 'cuda')
    Theta, phi1, mu1 = wrf.init(theta0)
    Theta += theta1 * wrf.mu0

    U = torch.zeros((nz, ny, nx - 1)).cuda()
    V = torch.zeros((nz, ny - 1, nx)).cuda()
    W = torch.zeros((nz - 1, ny, nx)).cuda()
    O = torch.zeros((nz - 1, ny, nx)).cuda()

    for i in range(10):
        U, V, W, O, Theta, phi1, mu1 = wrf(U, V, W, O, Theta, phi1, mu1)
        mu = wrf.mu0 + mu1
        u = U / wrf.bx(mu)
        v = V / wrf.by(mu)
        w = W / wrf.bz(mu)
        o = O / wrf.bz(mu)
        theta = Theta / mu

        interval = 1
        if i % interval == 0:
            # plt.cla()
            # fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            #
            # ctf = ax[0, 0].contourf(u[nz // 2, :, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[0, 0].set_title('u')
            # plt.colorbar(ctf, ax=ax[0, 0], format='%.1e')
            #
            # ctf = ax[0, 1].contourf(v[nz // 2, :, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[0, 1].set_title('v')
            # plt.colorbar(ctf, ax=ax[0, 1], format='%.1e')
            #
            # ctf = ax[0, 2].contourf(w[:, 32, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[0, 2].set_title('w')
            # plt.colorbar(ctf, ax=ax[0, 2], format='%.1e')
            #
            # ctf = ax[1, 0].contourf(mu1[nz // 2, :, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[1, 0].set_title(r'$\mu^\prime$')
            # plt.colorbar(ctf, ax=ax[1, 0], format='%.1e')
            #
            # ctf = ax[1, 1].contourf(phi1[:, 32, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[1, 1].set_title(r'$\phi^\prime$')
            # plt.colorbar(ctf, ax=ax[1, 1], format='%.1e')
            #
            # ctf = ax[1, 2].contourf(o[:, 32, :].cpu().numpy(), levels=50, cmap='jet')
            # ax[1, 2].set_title(r'$\omega$')
            # plt.colorbar(ctf, ax=ax[1, 2], format='%.1e')
            #
            # fig.text(0.01, 0.95, f't={i * dt}s', size=18)
            # plt.tight_layout()
            # plt.savefig(f'res/res{i // interval}.jpeg', dpi=300)
            # plt.close()
            # plt.clf()

            print(i)
