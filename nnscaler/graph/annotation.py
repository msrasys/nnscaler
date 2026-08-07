"""Optional pre-policy IR graph annotation."""

import os
from typing import TYPE_CHECKING

from nnscaler.utils import load_type

if TYPE_CHECKING:
    from nnscaler.graph.graph import IRGraph
    from nnscaler.parallel import ComputeConfig

_GRAPH_ANNOTATOR_ENV = "NNSCALER_GRAPH_ANNOTATOR"


def apply_graph_annotator(graph: "IRGraph", cfg: "ComputeConfig") -> None:
    """Invoke the configured graph annotator before policy transformations."""

    annotator_name = os.environ.get(_GRAPH_ANNOTATOR_ENV)
    if not annotator_name:
        return
    annotator = load_type(annotator_name)
    if not callable(annotator):
        raise TypeError(f"{_GRAPH_ANNOTATOR_ENV} must resolve to a callable")
    annotator(graph, cfg)
