

class TSchedulePool:

    class __TSchedulePool:

        def __init__(self):

            self._sus = list()
            self._flow_id = -1

    instance = None

    def __init__(self):
        if not TSchedulePool.instance:
            TSchedulePool.instance = TSchedulePool.__TSchedulePool()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_su(self, su):
        self.instance._sus.append(su)

    def sus(self):
        return self.instance._sus

    def clear(self):
        self.instance._sus = list()
        self.instance._flow_id = -1

    def gen_id(self) -> int:
        """
        Generate an unique action id
        """
        self.instance._flow_id += 1
        return self.instance._flow_id

    def __repr__(self):
        dscp = '\n'.join([repr(su) for su in self._sus])
        return dscp
