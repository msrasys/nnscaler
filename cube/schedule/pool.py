from typing import List, Any
import copy


class SchedulePool:

    class __SchedulePool:

        def __init__(self):

            self._nodes = list()
            self._tapes = dict()

    instance = None

    def __init__(self):
        if not SchedulePool.instance:
            SchedulePool.instance = SchedulePool.__SchedulePool()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_node(self, node):
        self.instance._nodes.append(node)

    def nodes(self) -> List:
        return copy.copy(self.instance._nodes)

    def tape(self, tensor, trace: Any):
        """
        Record the trace generated to this tensor
        """
        self.instance._tapes[tensor._id] = trace

    def get_tape(self, tensor):
        """
        Get the trace given the tensor
        """
        if tensor._id not in self.instance._tapes:
            return None
        else:
            return self.instance._tapes[tensor._id]

    def clear(self):
        self.instance._nodes = list()
        self.instance._tapes = dict()

    def __repr__(self):
        dscp = '\n'.join([repr(node) for node in self._nodes])
        return dscp
