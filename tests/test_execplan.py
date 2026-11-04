#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

from nnscaler.execplan.execplan import ExeReuseCell, ExecutionPlan
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.cten import IRCell
from nnscaler.ir.operator import IRBpOperation
from nnscaler.ir.tensor import IRFullTensor


class _DispatchableCell(IRCell):
    def __init__(self, original_output, dispatched_output):
        super().__init__('dispatchable', 'dispatchable', 0, 1)
        self.set_output(0, original_output)
        self.device = [0, 1]

        self.dispatched = IRCell('dispatched', 'dispatched', 0, 1)
        self.dispatched.set_output(0, dispatched_output)
        self.dispatched.device = 0

    def dispatch(self, device: int):
        assert device == 0
        return self.dispatched


def test_reuse_dispatch_updates_equal_shape_value_partition():
    original = IRFullTensor((8,)).tosub()
    dispatched = original.parent.select(((0, 8),), (1, 2))
    cell = _DispatchableCell(original, dispatched)

    micro_output = IRFullTensor((8,)).tosub()
    reuse = ExeReuseCell(cell, [], [micro_output]).dispatch(0)

    output = reuse.output(0)
    assert output.parent == micro_output.parent
    assert output.shape == micro_output.shape
    assert output.valmap == (1, 2)


@pytest.mark.parametrize('backward_first', [False, True])
def test_execution_plan_reuses_dispatched_pair_and_placement(backward_first):
    class Cell(IRCell):
        def __init__(self):
            super().__init__('scheduled', 'scheduled', 0, 0)
            self.reads = 0
            self.calls = []

        @property
        def device(self):
            self.reads += 1
            return [0, 1]

        def dispatch(self, device):
            self.calls.append(device)
            forward = IRCell('physical', 'physical', 0, 0)
            backward = IRBpOperation((), ())
            forward.device = backward.device = device
            IRCell.make_pair(forward, backward)
            return forward

    forward = Cell()
    backward = IRBpOperation((), ())
    backward.device = [0, 1]
    IRCell.make_pair(forward, backward)
    nodes = [backward, forward] if backward_first else [forward, backward]
    graph = IRGraph(nodes, [], [], 'scheduled')
    forward.reads = 0
    plan = ExecutionPlan(graph, nodes * 2)
    assert plan.devices() == [0, 1]
    for device in plan.devices():
        seq = plan.seq(device)
        assert seq[0] is seq[2] and seq[1] is seq[3]
        assert seq[0].mirror is seq[1] and seq[1].mirror is seq[0]
        assert all(cell.device == (device,) for cell in seq)
        assert [cell.isfw() for cell in seq] == [node.isfw() for node in nodes * 2]
    assert forward.calls == [0, 1]
    assert forward.reads == 1
