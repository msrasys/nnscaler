import torch
import einops

############### custom op #################
def update_diag(w: torch.Tensor, F: torch.Tensor, G: torch.Tensor,
                delta_x_filter:torch.Tensor, delta_y_filter: torch.Tensor, deltaA:torch.Tensor,
                pi0:torch.Tensor, pi1:torch.Tensor, sigma:torch.Tensor, dt:torch.Tensor, dz:float):
    # def pre_conv3d_reshape(X):
    #     return einops.einops.rearrange(X, '(b0 b1 Nz) Ny Nx -> b0 b1 Nz Ny Nx', b0=1, b1=1)
    # def post_conv3d_reshape(X):
    #     return einops.einops.rearrange(X, 'b0 b1 Nz Ny Nx -> (b0 b1 Nz) Ny Nx')
    # def delta_x(X):
    #     return post_conv3d_reshape(F.conv3d(pre_conv3d_reshape(X), delta_x_filter))
    # def delta_y(X):
    #     return post_conv3d_reshape(F.conv3d(pre_conv3d_reshape(X), delta_y_filter))
    # # update diagnostic variable w (nz + 1, ny, nx)
    # for i in range(1, w.shape[0]):
    #     w[i] = - ((delta_x(F[:i]) + delta_y(G[:i])) * dz).sum(dim=0) / deltaA / pi1 \
    #         - sigma[i] * (pi1 - pi0) / dt / pi1

    return w

def strip_2_borders(w: torch.Tensor):
    return w[1:-1]