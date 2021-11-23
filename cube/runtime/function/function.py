import torch
import torch.nn.functional as F


def embedding(input: torch.Tensor, weight: torch.Tensor, start: int, stop: int):
    """
    Embedding
    """
    input_mask = (input < start) | (input >= stop)
    masked_input = input.clone() - start
    masked_input[input_mask] = 0
    output = F.embedding(
        masked_input, weight,
        None, None, 2.0, False, False
    )
    output[input_mask, :] = 0.0
    return output

