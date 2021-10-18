from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType, ScheduleUnit

from cube.ir.cten import IRCell, IRTensor


def test_schedule_pool():

    SchedulePool().clear()
    assert len(SchedulePool()._sus) == 0
    assert len(SchedulePool().sus()) == 0

    cell = IRCell(
        name='test', signature='test', input_length=4, output_length=2
    )
    su = ScheduleUnit([cell], SUType.Forward, name='su')
    SchedulePool().add_su(su)
    
    assert len(SchedulePool()._sus) == 1
    assert len(SchedulePool().sus()) == 1

    for record_su in SchedulePool().sus():
        assert record_su == su

