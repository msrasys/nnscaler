#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.ir.cten import IRCell
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
