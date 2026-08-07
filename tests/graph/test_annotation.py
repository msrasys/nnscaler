from nnscaler.graph import annotation


def test_graph_annotator_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NNSCALER_GRAPH_ANNOTATOR", raising=False)
    monkeypatch.setattr(
        annotation,
        "load_type",
        lambda name: (_ for _ in ()).throw(AssertionError(name)),
    )

    annotation.apply_graph_annotator(object(), object())


def test_graph_annotator_loads_and_invokes_configured_callable(monkeypatch):
    graph = object()
    cfg = object()
    calls = []

    def annotator(actual_graph, actual_cfg):
        calls.append((actual_graph, actual_cfg))

    monkeypatch.setenv("NNSCALER_GRAPH_ANNOTATOR", "example.annotate")
    monkeypatch.setattr(annotation, "load_type", lambda name: annotator)

    annotation.apply_graph_annotator(graph, cfg)

    assert calls == [(graph, cfg)]
