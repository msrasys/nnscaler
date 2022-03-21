from typing import Optional, List
import torch
import torch.nn.functional as TorchF


def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N  iC H  W
    weight: oC iC dH dW
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]
    """
    # switch H and W to match torch.nn.functional.pad
    padding = padding[len(padding) // 2:] + padding[0:len(padding) // 2]
    input = TorchF.pad(input, padding, 'constant', 0)
    return TorchF.conv2d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


def embedding(input: torch.Tensor, weight: torch.Tensor, start: int, stop: int):
    """
    Embedding

    Inputs:
        input: torch.Tensor [*]
        weight: [vocab size, embed size]
        start: int
        stop: int

    Outputs:
        output: [*, embed_size]
    """
    input = input.long()
    input_mask = (input < start) | (input >= stop)
    masked_input = input.clone() - start
    masked_input[input_mask] = 0
    output = TorchF.embedding(
        masked_input, weight,
        None, None, 2.0, False, False
    )
    output[input_mask, :] = 0.0
    return output

def einops(input: torch.Tensor, recipe_str, reduction_type: str):
    import pickle
    recipe = pickle.loads(recipe_str)
    from einops.einops import _apply_recipe
    output = _apply_recipe(recipe, input, reduction_type)
    return output

############### custom op #################
#TODO move me
def update_diag(w: torch.Tensor, F: torch.Tensor, G: torch.Tensor,
                delta_x_filter:torch.Tensor, delta_y_filter: torch.Tensor, deltaA:torch.Tensor,
                pi0:torch.Tensor, pi1:torch.Tensor, sigma:torch.Tensor, dz, dt):
    def pre_conv3d_reshape(X):
        return einops.einops.rearrange(X, '(b0 b1 Nz) Ny Nx -> b0 b1 Nz Ny Nx', b0=1, b1=1)
    def post_conv3d_reshape(X):
        return einops.einops.rearrange(X, 'b0 b1 Nz Ny Nx -> (b0 b1 Nz) Ny Nx')
    def delta_x(X):
        return post_conv3d_reshape(F.conv3d(pre_conv3d_reshape(X), delta_x_filter))
    def delta_y(X):
        return post_conv3d_reshape(F.conv3d(pre_conv3d_reshape(X), delta_y_filter))
    # update diagnostic variable w (nz + 1, ny, nx)
    for i in range(1, w.shape[0]):
        w[i] = - ((delta_x(F[:i]) + delta_y(G[:i])) * dz).sum(dim=0) / deltaA / pi1 \
            - sigma[i] * (pi1 - pi0) / dt / pi1

    return w

def strip_2_borders(w: torch.Tensor):
    return w[1:-1]

