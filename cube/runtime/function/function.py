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
    return tensor if times == 1 else tuple([tensor] * times)


def accum(*tensors: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    accumulate tensors in to one tensor
    """
    if len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return torch.sum(torch.stack(tensors, dim=0), dim=0)


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


def layer_norm(input: torch.Tensor,
               weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
               normalized_shape: List[int], eps: float = 1e-05) -> torch.Tensor:
    """
    LayerNorm
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


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

