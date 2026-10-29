# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
import pytest
import torch

from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.tensor import IRFullTensor


def graph(monkeypatch, *, hook=False, input_is_root=False, return_input=False, extra_read=False):
    x = IRFullTensor((4, 4), name='x', dtype=torch.float32, requires_grad=True).tosub()
    y = IRFullTensor((4, 4), name='y', dtype=torch.float32, requires_grad=True).tosub()
    gx, gy = x.parent.grad.tosub(), y.parent.grad.tosub()
    fw = IRSegment([], [x], [y])
    bw = IRSegment([], [gy], [gx])
    monkeypatch.setattr(bw, 'isfw', lambda: False)
    if hook:
        bw.pre_hook = lambda *args: None
    monkeypatch.setattr(FuncEmission, 'get_backward_callsite_io_tensors',
                        lambda self, node: ([x], [x if input_is_root else y], [gy], [gx]))
    nodes = [fw]
    if extra_read:
        nodes.append(IRAdapter([x], []))
    nodes.append(bw)
    outputs = [x] if return_input else []
    life = LifeCycle(nodes, [x], outputs)
    return x, life, nodes, outputs


@pytest.mark.parametrize('extra_read', [False, True])
def test_input_identity_capture_follows_last_actual_read(monkeypatch, extra_read):
    x, life, nodes, outputs = graph(monkeypatch, extra_read=extra_read)
    captures = life.input_references_after_data_use(nodes, outputs)
    assert captures == {int(extra_read): [x]}
    assert x in life.release_tensors_after_line(len(nodes) - 1)


@pytest.mark.parametrize('reason', ['hook', 'input_is_root', 'return_input'])
def test_data_needed_by_hook_backward_root_or_caller_is_retained(monkeypatch, reason):
    _, life, nodes, outputs = graph(monkeypatch, **{reason: True})
    assert life.input_references_after_data_use(nodes, outputs) == {}


def test_tensor_named_by_hook_metadata_keeps_its_data_reference(monkeypatch):
    x, life, nodes, outputs = graph(monkeypatch)
    nodes[0].pre_hook = lambda *args: None
    nodes[0].hook_meta = {'external_data': x}
    assert life.input_references_after_data_use(nodes, outputs) == {}


@pytest.mark.parametrize('input_is_root', [False, True])
def test_output_edges_are_captured_only_after_all_data_reads(monkeypatch, input_is_root):
    x, life, nodes, outputs = graph(monkeypatch, input_is_root=input_is_root)
    root = x if input_is_root else nodes[0].outputs()[0]
    nodes.insert(1, IRAdapter([root], []))
    life = LifeCycle(nodes, [x], outputs)
    inputs, roots = life.pipeline_references_after_data_use(
        nodes, outputs, release_inputs=True, release_outputs=True)
    assert roots == {1: [root]}
    assert inputs == ({} if input_is_root else {0: [x]})
    assert root in life.release_tensors_after_line(len(nodes)-1)


@pytest.mark.parametrize('reason', ['hook', 'return_input'])
def test_output_references_keep_data_read_by_hooks_or_returned_to_caller(monkeypatch, reason):
    _, life, nodes, outputs = graph(monkeypatch, input_is_root=True, **{reason: True})
    assert life.pipeline_references_after_data_use(
        nodes, outputs, release_inputs=True, release_outputs=True) == ({}, {})
