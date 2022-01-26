from typing import Optional, List
import torch
import torch.nn.functional as TorchF


def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N  iC H  W
    weight: oC iC dH dW
    bias:   oC
    padding: int, List[int], e.g., 1, [1, 1], [1, 0, 1, 0]
    """
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
