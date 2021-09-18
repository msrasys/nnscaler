from typing import List, Callable

from cube.tschedule.action import Action


class TSchedulePool:

    class __TSchedulePool:

        def __init__(self):

            self._actions: List[Action] = list()

    instance = None

    def __init__(self):
        if not TSchedulePool.instance:
            TSchedulePool.instance = TSchedulePool.__TSchedulePool()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_action(self, action: Action):
        self.instance._actions.append(action)

    def actions(self):
        return self.instance._actions

    def clear(self):
        self.instance._actions = list()

    def __repr__(self):
        dscp = '\n'.join([repr(action) for action in self._actions])
        return dscp


def schedule(fn: Callable, policy=None, *args, **kwargs):
    """
    AI Scientist calls like:

        @cube.tschedule.schedule
        def train_step(model, optimizer, datas, labels):
            for (data, label) in datas:
                loss = model(data, label)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ...
        for datas, labels in dataloader():
            train_step(model, optimizer, datas, labels)
        ...
    """
    raise NotImplementedError