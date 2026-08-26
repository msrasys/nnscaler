"""Reproduce why FBW clamps gradients at overlapping boundary nodes.

Run with:

    /data/weijiangxu/uvenv/py3.12torch2.10/bin/python \
        repro_fbw_boundary_clamp.py
"""

import torch
from torch.autograd.graph import GradientEdge, Node


def get_grad_accumulator(tensor: torch.Tensor) -> Node:
    """Materialize and return a leaf tensor's AccumulateGrad node."""
    view_grad_fn = tensor.view_as(tensor).grad_fn
    assert view_grad_fn is not None
    grad_accumulator = view_grad_fn.next_functions[0][0]
    assert grad_accumulator is not None
    return grad_accumulator


def main() -> None:
    # Forward graph:
    #
    #     weight --(* 2)--> ancestor --(* 3)--> descendant
    #
    # Treat both `ancestor` and `descendant` as FBW boundary intermediates.
    weight = torch.tensor(2.0, requires_grad=True)
    ancestor = weight * 2
    descendant = ancestor * 3
    ancestor_node = ancestor.grad_fn
    descendant_node = descendant.grad_fn
    assert ancestor_node is not None
    assert descendant_node is not None

    saved_grads: dict[str, tuple[torch.Tensor | None, ...]] = {}

    def capture_ancestor(
        grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        saved_grads["ancestor"] = grad_outputs
        print("B captures ancestor grad_outputs:", grad_outputs)

    def capture_descendant(
        grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        saved_grads["descendant"] = grad_outputs
        print("B captures descendant grad_outputs:", grad_outputs)

    capture_handles = [
        ancestor_node.register_prehook(capture_ancestor),
        descendant_node.register_prehook(capture_descendant),
    ]
    try:
        reference_dweight = torch.autograd.grad(
            descendant,
            weight,
            retain_graph=True,
        )[0]
    finally:
        for handle in capture_handles:
            handle.remove()

    # This mirrors stage_backward_weight: every saved boundary edge becomes a
    # root in one GraphTask and receives the gradient captured during B.
    boundary_edges = (
        GradientEdge(ancestor_node, 0),
        GradientEdge(descendant_node, 0),
    )
    boundary_grads = (
        saved_grads["ancestor"][0],
        saved_grads["descendant"][0],
    )
    weight_edge = GradientEdge(get_grad_accumulator(weight), 0)

    double_counted_dweight = torch.autograd.grad(
        boundary_edges,
        weight_edge,
        grad_outputs=boundary_grads,
        retain_graph=True,
    )[0]

    def clamp_ancestor(
        current_grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        print("W ancestor pre-hook receives accumulated:", current_grad_outputs)
        print("W ancestor pre-hook restores saved:     ", saved_grads["ancestor"])
        return saved_grads["ancestor"]

    def clamp_descendant(
        current_grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        print("W descendant pre-hook receives:         ", current_grad_outputs)
        return saved_grads["descendant"]

    clamp_handles = [
        ancestor_node.register_prehook(clamp_ancestor),
        descendant_node.register_prehook(clamp_descendant),
    ]
    try:
        clamped_dweight = torch.autograd.grad(
            boundary_edges,
            weight_edge,
            grad_outputs=boundary_grads,
        )[0]
    finally:
        for handle in clamp_handles:
            handle.remove()

    print()
    print("Reference dWeight:               ", reference_dweight.item())
    print("Two boundary roots, no clamp:    ", double_counted_dweight.item())
    print("Two boundary roots, with clamp:  ", clamped_dweight.item())

    torch.testing.assert_close(reference_dweight, torch.tensor(6.0))
    torch.testing.assert_close(double_counted_dweight, torch.tensor(12.0))
    torch.testing.assert_close(clamped_dweight, reference_dweight)


if __name__ == "__main__":
    main()
