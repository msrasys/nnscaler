from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType, ScheduleUnit

from cube.ir.cten import IRCell


def test_schedule_pool():

    SchedulePool().clear()
    assert len(SchedulePool()._nodes) == 0
    assert len(SchedulePool().nodes()) == 0

    cell = IRCell(
        name='test', signature='test', input_length=4, output_length=2
    )
    SchedulePool().add_node(cell)
    
    assert len(SchedulePool()._nodes) == 1
    assert len(SchedulePool().nodes()) == 1

    for record_node in SchedulePool().nodes():
        assert record_node == cell
