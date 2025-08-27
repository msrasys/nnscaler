#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, List

from nnscaler.cli.train_hook import TrainHook, TrainHookHost


class A(TrainHook):
    pass

class B(TrainHook):
    pass

class C(TrainHook, TrainHookHost):
    def _get_hook_objects(self) -> List[Any]:
        return [A(), B(), self]


class D(TrainHookHost):
    def _get_hook_objects(self) -> List[Any]:
        return [self, A(), C()]

def test_hook():
    hooks = D().get_hooks()
    assert len(hooks) == 4
    assert isinstance(hooks[0], A)
    assert isinstance(hooks[1], C)
    assert isinstance(hooks[2], A)
    assert isinstance(hooks[3], B)
