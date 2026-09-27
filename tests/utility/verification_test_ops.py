import torch

import nnscaler


_runtime_state = None


def setup_runtime_state():
    global _runtime_state
    _runtime_state = None


def snapshot_runtime_state():
    return _runtime_state


@nnscaler.register_op(
    "l n -> l",
    verify_setup_fn=setup_runtime_state,
    verify_state_fn=snapshot_runtime_state,
)
def locally_normalized_sum(x: torch.Tensor):
    global _runtime_state
    _runtime_state = x.detach().clone()
    return x.sum(dim=1) / x.shape[0]


@nnscaler.register_op(
    "l n -> l n",
    verify_setup_fn=setup_runtime_state,
    verify_state_fn=snapshot_runtime_state,
)
def stateful_identity(x: torch.Tensor):
    global _runtime_state
    _runtime_state = x.detach().clone()
    return x


@nnscaler.register_op("l n -> l n")
def stateless_scale(x: torch.Tensor):
    return x * 2
