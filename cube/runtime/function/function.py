from typing import Optional, List, Tuple
import torch
import torch.nn.functional as TorchF


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """
    identity forward
    """
    return tensor

def anchor(name: str):
    """
    anchor operation for graph navigation
    """
    return None


def multiref(tensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]:
    """
    identity forward. Create multiple same tensor.
    """
    assert times > 1, "multiref can only be used for num of tensor >= 2"
    return tuple([tensor] * times)


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

def conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N iC D H W,
    weight: oC iC dH dW, oC
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]

    output: N oC oD oH oW
    """
    # switch D, H and W to match torch.nn.functional.pad
    pad_padding = [padding[-1 - (i // 2)] for i in range(len(padding) * 2)]
    input = TorchF.pad(input, pad_padding, 'constant', 0)
    return TorchF.conv3d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


def embedding(input: torch.Tensor, weight: torch.Tensor, padding_idx: Optional[int], start: int, stop: int):
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
        masked_input, weight, padding_idx,
        None, 2.0, False, False
    )
    output[input_mask, :] = 0.0
    return output


# 'torch.select_scatter' isn't supported by Torch2ONNX yet.
# Implement it with 'torch.masked_scatter' which is supported with ONNX opset=11.
def select_scatter(input:torch.Tensor, src:torch.Tensor, dim:int, index:int):
    # e.g. [..., 1, -1, 1, ...]
    shape = [1] * input.ndim
    shape[dim] = -1

    d = input.shape[dim]
    mask = torch.zeros([d], dtype=torch.bool, device=input.device)
    mask[index] = True
    mask = mask.reshape(shape)

    return torch.masked_scatter(input, mask, src)


def einops(input: torch.Tensor, recipe_str, reduction_type: str):
    import pickle
    recipe = pickle.loads(recipe_str)
    from einops.einops import _apply_recipe
    output = _apply_recipe(recipe, input, reduction_type)
    return output

############### custom op #################
#TODO move me
def update_diag_(w: torch.Tensor, F: torch.Tensor, G: torch.Tensor,
                delta_x_filter:torch.Tensor, delta_y_filter: torch.Tensor, deltaA:torch.Tensor,
                pi0:torch.Tensor, pi1:torch.Tensor, sigma:torch.Tensor, dt, dz):
    import einops.einops
    def pre_conv3d_reshape(X):
        return einops.einops.rearrange(X, '(b0 b1 Nz) Ny Nx -> b0 b1 Nz Ny Nx', b0=1, b1=1)
    def post_conv3d_reshape(X):
        return einops.einops.rearrange(X, 'b0 b1 Nz Ny Nx -> (b0 b1 Nz) Ny Nx')
    def delta_x(X):
        return post_conv3d_reshape(torch.nn.functional.conv3d(pre_conv3d_reshape(X), delta_x_filter))
    def delta_y(X):
        return post_conv3d_reshape(torch.nn.functional.conv3d(pre_conv3d_reshape(X), delta_y_filter))
    # update diagnostic variable w (nz + 1, ny, nx)
    import warnings
    warnings.warn("detaching w in update_diag_...")
    w.detach_() #to prevent ERROR: A leaf Variable that requires grad is being used in an in-place operation.
    for i in range(1, w.shape[0]):
        w[i] = - ((delta_x(F[:i]) + delta_y(G[:i])) * dz).sum(dim=0) / deltaA / pi1 \
            - sigma[i] * (pi1 - pi0) / dt / pi1

    return w

def update_geopotential_(phi: torch.Tensor, zs: torch.Tensor, P: torch.Tensor, P_: torch.Tensor, theta: torch.Tensor, g, CPD, nz):
    import warnings
    warnings.warn("detaching phi in update_geopotential_...")
    phi.detach_()
    phi[-1] = g * zs - CPD * (P[-1] - P_[-1]) * theta[-1]
    for i in range(1, nz):
        tmp = phi[-i] - CPD * (P_[-i - 1] - P[-i]) * theta[-i]
        phi[-1 - i] = tmp - CPD * (P[-1 - i] - P_[-1 - i]) * theta[-1 - i]

    return phi

def strip_2_borders(w: torch.Tensor):
    return w[1:-1]
