from typing import List
import copy


class SchedulePool:

    class __SchedulePool:

        def __init__(self):

            self._sus = list()

    instance = None

    def __init__(self):
        if not SchedulePool.instance:
            SchedulePool.instance = SchedulePool.__SchedulePool()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_su(self, su):
        self.instance._sus.append(su)

    def sus(self) -> List:
        return copy.copy(self.instance._sus)

    def clear(self):
        self.instance._sus = list()

    def __repr__(self):
        dscp = '\n'.join([repr(su) for su in self._sus])
        return dscp
