#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import OrderedDict
from dataclasses import dataclass
import sys
import pytest
import torch

from nnscaler.utils import (
    select_many, classproperty, fields, set_member_by_name, unchecked_fields,
    transform_recursively, first, first_or, StepwiseConfig, recursion_limit,
)


def test_recursion_limit():
    original_limit = sys.getrecursionlimit()
    temporary_limit = original_limit + 100

    with recursion_limit(temporary_limit):
        assert sys.getrecursionlimit() == temporary_limit

        with recursion_limit(original_limit, increase_only=True):
            assert sys.getrecursionlimit() == temporary_limit

    assert sys.getrecursionlimit() == original_limit

    with pytest.raises(RuntimeError):
        with recursion_limit(temporary_limit):
            raise RuntimeError

    assert sys.getrecursionlimit() == original_limit


def test_select_many():
    assert list(select_many([1, 2], lambda k: [])) == []
    assert list(select_many([1, [2, 3]], lambda k: k if isinstance(k, list) else [k])) == [1, 2, 3]
    with pytest.raises(TypeError):
        list(select_many([1, [2, 3]], lambda k: k))


def test_classproperty_int():
    class A:
        _x = 1234567
        @classproperty
        def value(cls):
            return cls._x

    assert A.value == 1234567
    assert id(A().value) == id(A.value)

    with pytest.raises(AttributeError):
        A().value = 43

    assert A.value == 1234567


def test_classproperty_dict():
    class A:
        _x = {}
        @classproperty
        def cfg(cls):
            return cls._x.setdefault('a', {})

    x = A.cfg
    x[1] = 2
    assert A.cfg == {1: 2}
    assert id(A().cfg) == id(x)


def test_fields():
    @dataclass
    class A:
        x: int
        y: int

    assert fields(A).x == 'x'
    assert fields(A).y == 'y'
    with pytest.raises(AttributeError):
        fields(A).z

    assert unchecked_fields(A).x == 'x'
    assert unchecked_fields(A).y == 'y'
    assert unchecked_fields(A).z == 'z'

    a = A(x=0, y=0)
    assert unchecked_fields(a).x == 'x'
    assert unchecked_fields(a).y == 'y'
    assert unchecked_fields(a).z == 'z'

    class B:
        def __init__(self):
            self.a = A(x=1, y=2)

    assert unchecked_fields(B).x == 'x'
    b = B()
    assert unchecked_fields(b).x == 'x'
    assert unchecked_fields(b.a).x == 'x'


def test_set_member_by_name():
    model = torch.nn.Module()
    set_member_by_name(model, "x", 42)
    assert model.x == 42
    with pytest.raises(AttributeError):
        set_member_by_name(model, 'x.y.z', 43)

    set_member_by_name(model, 'a.b.c', 44)
    assert model.a.b.c == 44

    model = torch.nn.Module()
    child_module = torch.nn.Module()
    set_member_by_name(model, "x.y", child_module)
    assert model.x.y == child_module

    set_member_by_name(model, 'x.y.z', 45)
    assert model.x.y == child_module
    assert model.x.y.z == 45


def test_transform_recursively():
    data = {
        'a': torch.tensor([1]),
        'b': [torch.tensor(4), {'c': torch.tensor([5])}],
        'd': (7, torch.tensor(8)),
        'e': {1: 9, 2: torch.tensor(10)}.keys(),
        'f': {1: 9, 2: torch.tensor(11)}.items(),
        'g': {1: 9, 2: torch.tensor(12)}.values(),
        'h': {1: 9, 2: torch.tensor(13)},
        'i': slice(0, 10, None),
        'j': torch.Size([11, 12]),
        'k': OrderedDict({1: 9, 2: 10}),
        'l': {1: 9, 2: 10}.values(),
        'm': [1, 2, 3],
        'n': slice(0, 10, torch.tensor(2)),
        'o': {torch.tensor(1): 9, torch.tensor(2): 10},
        'p': {torch.tensor(1): 9, torch.tensor(2): 10}.items(),
        'q': {torch.tensor(1): 9, torch.tensor(2): 10}.keys()
    }

    def fn(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return x

    result1 = transform_recursively(
        data, fn,
        target_types=torch.Tensor,
        collection_types=None,
        skip_dict_keys=True,
    )

    result2 = transform_recursively(
        data, fn,
        target_types=torch.Tensor,
        collection_types=None,
        skip_dict_keys=False,
    )
    target = {
        'a': 1,
        'b': [4, {'c': 5}],
        'd': (7, 8),
        'e': {1: 1, 2: 2}.keys(),
        'f': dict([(1, 9), (2, 11)]).items(),
        'g': {1: 9, 2: 12}.values(),
        'h': {1: 9, 2: 13},
        'i': slice(0, 10, None),
        'j': torch.Size([11, 12]),
        'k': OrderedDict({1: 9, 2: 10}),
        'l': data['l'],
        'm': [1, 2, 3],
        'n': slice(0, 10, 2),
    }
    # dict values are not comparable.
    assert list(target['g']) == list(result1.pop('g'))
    assert list(target['g']) == list(result2.pop('g'))
    target.pop('g')


    skip_key_target = {
        **target,
        'o': {torch.tensor(1): 9, torch.tensor(2): 10},
        'p': {torch.tensor(1): 9, torch.tensor(2): 10}.items(),
        'q': {1: 9, 2: 10}.keys()
    }
    noskip_key_target = {
        **target,
        'o': {1: 9, 2: 10},
        'p': dict([(1, 9), (2, 10)]).items(),
        'q': {1: 9, 2: 10}.keys()
    }

    from tests.parallel_module.common import assert_equal

    assert_equal(list(skip_key_target.pop('o')), list(result1.pop('o')))
    assert_equal(list(skip_key_target.pop('p')), list(result1.pop('p')))
    assert_equal(list(skip_key_target.pop('q')), list(result1.pop('q')))

    assert_equal(result1, skip_key_target)
    assert_equal(result2, noskip_key_target)


def test_first_no_filter():
    assert first([1, 2, 3]) == 1


def test_first_no_filter_falsy_first():
    """first() without filter should return falsy values like 0, None, '', False."""
    assert first([0, 1, 2]) == 0
    assert first([None, 1]) is None
    assert first(['', 'a']) == ''
    assert first([False, True]) is False


def test_first_with_filter():
    assert first([1, 2, 3], lambda x: x > 1) == 2


def test_first_with_filter_falsy_items():
    """Filter should still skip items that don't match, even if they are falsy."""
    assert first([0, 0, 3], lambda x: x > 0) == 3


def test_first_empty():
    with pytest.raises(ValueError, match="No element satisfies the condition"):
        first([])


def test_first_no_match():
    with pytest.raises(ValueError, match="No element satisfies the condition"):
        first([1, 2, 3], lambda x: x > 10)


def test_first_or_no_filter():
    assert first_or([1, 2, 3]) == 1


def test_first_or_no_filter_falsy_first():
    assert first_or([0, 1, 2]) == 0
    assert first_or([None, 1]) is None
    assert first_or([False, True]) is False


def test_first_or_with_filter():
    assert first_or([1, 2, 3], lambda x: x > 1) == 2


def test_first_or_empty():
    assert first_or([]) is None
    assert first_or([], default=42) == 42


def test_first_or_no_match():
    assert first_or([1, 2, 3], lambda x: x > 10) is None
    assert first_or([1, 2, 3], lambda x: x > 10, default=-1) == -1


# ---------------------------------------------------------------------------
# StepwiseConfig
# ---------------------------------------------------------------------------

def _ref_value_at(config, step):
    """Reference (naive) implementation of StepwiseConfig.value_at."""
    if isinstance(config, int):
        return config
    keys = sorted(config)
    active = config[keys[0]]  # smallest key's value applies before the first threshold
    for k in keys:
        if k <= step:
            active = config[k]
        else:
            break
    return active


def _ref_sum_value(config, start_step, end_step):
    return sum(_ref_value_at(config, s) for s in range(start_step, end_step))


def _ref_steps_to_consume(config, num_items, start_step):
    remaining = num_items
    step = start_step
    nsteps = 0
    while remaining > 0:
        remaining -= _ref_value_at(config, step)
        step += 1
        nsteps += 1
    return nsteps


def _ref_step_and_offset(config, num_items, start_step):
    step = start_step
    remaining = num_items
    while True:
        cap = _ref_value_at(config, step)
        if remaining < cap:
            return step, remaining
        remaining -= cap
        step += 1


# a mix of int and dict schedules used across the parametrized tests
_SCHEDULES = [
    3,
    1,
    {0: 2},
    {0: 2, 5: 4, 10: 8},
    {3: 4, 8: 2},         # first threshold > 0, values not monotonic
    {2: 1, 4: 3, 9: 5, 20: 2},
    {0: 5, 5: 1},         # decreasing
]


def test_stepwise_value_at_int():
    assert StepwiseConfig.value_at(3, 0) == 3
    assert StepwiseConfig.value_at(3, 100) == 3
    assert StepwiseConfig.value_at(3, -5) == 3


def test_stepwise_value_at_dict_basic():
    config = {0: 2, 5: 4, 10: 8}
    expected = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8]
    assert [StepwiseConfig.value_at(config, s) for s in range(len(expected))] == expected


def test_stepwise_value_at_before_first_threshold():
    # smallest key's value applies before the first threshold
    config = {3: 4, 8: 2}
    assert StepwiseConfig.value_at(config, 0) == 4
    assert StepwiseConfig.value_at(config, 2) == 4
    assert StepwiseConfig.value_at(config, 3) == 4
    assert StepwiseConfig.value_at(config, 7) == 4
    assert StepwiseConfig.value_at(config, 8) == 2
    assert StepwiseConfig.value_at(config, 100) == 2


def test_stepwise_value_at_unsorted_keys():
    # keys need not be provided in sorted order
    config = {10: 8, 0: 2, 5: 4}
    assert StepwiseConfig.value_at(config, 7) == 4
    assert StepwiseConfig.value_at(config, 12) == 8


def test_stepwise_value_at_empty_dict_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.value_at({}, 0)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_value_at_matches_reference(config):
    for step in range(0, 30):
        assert StepwiseConfig.value_at(config, step) == _ref_value_at(config, step)


def test_stepwise_sum_value_int():
    assert StepwiseConfig.sum_value(3, 0, 5) == 15
    assert StepwiseConfig.sum_value(3, 2, 2) == 0
    assert StepwiseConfig.sum_value(4, 10, 13) == 12


def test_stepwise_sum_value_dict():
    config = {0: 2, 5: 4, 10: 8}
    # steps 0..6 -> 2,2,2,2,2,4,4 = 18
    assert StepwiseConfig.sum_value(config, 0, 7) == 18
    # steps 5..11 -> 4,4,4,4,4,8,8 = 36
    assert StepwiseConfig.sum_value(config, 5, 12) == 36


def test_stepwise_sum_value_empty_range():
    assert StepwiseConfig.sum_value({0: 2, 5: 4}, 4, 4) == 0


def test_stepwise_sum_value_empty_dict_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.sum_value({}, 0, 5)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_sum_value_matches_reference(config):
    for start in range(0, 25):
        for end in range(start, 30):
            assert StepwiseConfig.sum_value(config, start, end) == _ref_sum_value(config, start, end)


def test_stepwise_steps_to_consume_int():
    assert StepwiseConfig.steps_to_consume(3, 10) == 4      # ceil(10/3)
    assert StepwiseConfig.steps_to_consume(3, 9) == 3
    assert StepwiseConfig.steps_to_consume(3, 0) == 0
    assert StepwiseConfig.steps_to_consume(3, 1) == 1


def test_stepwise_steps_to_consume_dict():
    config = {0: 2, 5: 4, 10: 8}
    # 5 steps of freq 2 consume exactly 10
    assert StepwiseConfig.steps_to_consume(config, 10, start_step=0) == 5
    # start after the last threshold: constant freq 8
    assert StepwiseConfig.steps_to_consume(config, 20, start_step=10) == 3  # ceil(20/8)


def test_stepwise_steps_to_consume_zero():
    assert StepwiseConfig.steps_to_consume({0: 2, 5: 4}, 0, start_step=3) == 0


def test_stepwise_steps_to_consume_empty_dict_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.steps_to_consume({}, 5)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_steps_to_consume_matches_reference(config):
    for num in range(0, 120):
        for start in range(0, 25):
            assert StepwiseConfig.steps_to_consume(config, num, start) == \
                _ref_steps_to_consume(config, num, start)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_steps_to_consume_consistent_with_sum(config):
    # consuming `steps_to_consume(num)` steps must cover at least `num` items,
    # and one fewer step must not be enough (when num > 0)
    for num in range(1, 60):
        for start in range(0, 15):
            nsteps = StepwiseConfig.steps_to_consume(config, num, start)
            assert StepwiseConfig.sum_value(config, start, start + nsteps) >= num
            assert StepwiseConfig.sum_value(config, start, start + nsteps - 1) < num


def test_stepwise_step_and_offset_int():
    # constant freq 3: item 0,1,2 -> step0; item 3,4,5 -> step1; ...
    assert StepwiseConfig.step_and_offset(3, 0) == (0, 0)
    assert StepwiseConfig.step_and_offset(3, 2) == (0, 2)
    assert StepwiseConfig.step_and_offset(3, 3) == (1, 0)
    assert StepwiseConfig.step_and_offset(3, 7) == (2, 1)
    # with a start_step offset
    assert StepwiseConfig.step_and_offset(3, 7, start_step=10) == (12, 1)


def test_stepwise_step_and_offset_dict():
    config = {0: 2, 5: 4, 10: 8}
    # steps 0..4 hold 2 items each (items 0..9), step 5 holds 4 (items 10..13)
    assert StepwiseConfig.step_and_offset(config, 0) == (0, 0)
    assert StepwiseConfig.step_and_offset(config, 9) == (4, 1)
    assert StepwiseConfig.step_and_offset(config, 10) == (5, 0)
    assert StepwiseConfig.step_and_offset(config, 13) == (5, 3)
    assert StepwiseConfig.step_and_offset(config, 14) == (6, 0)


def test_stepwise_step_and_offset_start_step():
    config = {0: 2, 5: 4, 10: 8}
    # start at step 3 (freq 2): items 0,1 -> step3; 2,3 -> step4; 4,5,6,7 -> step5 (freq 4)
    assert StepwiseConfig.step_and_offset(config, 0, start_step=3) == (3, 0)
    assert StepwiseConfig.step_and_offset(config, 3, start_step=3) == (4, 1)
    assert StepwiseConfig.step_and_offset(config, 5, start_step=3) == (5, 1)


def test_stepwise_step_and_offset_negative_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.step_and_offset(3, -1)


def test_stepwise_step_and_offset_empty_dict_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.step_and_offset({}, 5)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_step_and_offset_matches_reference(config):
    for num in range(0, 120):
        for start in range(0, 25):
            assert StepwiseConfig.step_and_offset(config, num, start) == \
                _ref_step_and_offset(config, num, start)


@pytest.mark.parametrize("config", _SCHEDULES)
def test_stepwise_step_and_offset_consistent_with_sum(config):
    # the items before `step` plus `offset` must equal `num`
    for num in range(0, 80):
        for start in range(0, 15):
            step, offset = StepwiseConfig.step_and_offset(config, num, start)
            assert 0 <= offset < StepwiseConfig.value_at(config, step)
            assert StepwiseConfig.sum_value(config, start, step) + offset == num


def _ref_steps_per_epoch_list(config, items_per_period, n_periods):
    # uncompressed per-period step counts, steps counted globally across periods
    steps = []
    accum = 0
    for _ in range(n_periods):
        s = _ref_steps_to_consume(config, items_per_period, accum)
        steps.append(s)
        accum += s
    return steps


def test_stepwise_steps_per_period_int():
    # constant freq: every period consumes ceil(items / freq) steps
    assert StepwiseConfig.steps_per_period(3, 10) == {0: 4}
    assert StepwiseConfig.steps_per_period(2, 8) == {0: 4}


def test_stepwise_steps_per_period_dict():
    # items=10, freq 2 for steps 0..4, freq 4 from step 5.
    # epoch0 starts at step0: 10 items / 2 = 5 steps (steps 0..4)
    # epoch1 starts at step5: 10 items / 4 = 3 steps -> converged (>= max key 5)
    assert StepwiseConfig.steps_per_period({0: 2, 5: 4}, 10) == {0: 5, 1: 3}


def test_stepwise_steps_per_period_empty_dict_raises():
    with pytest.raises(ValueError):
        StepwiseConfig.steps_per_period({}, 10)


def test_stepwise_steps_per_period_zero_items():
    # an empty period cannot advance and is rejected
    with pytest.raises(ValueError):
        StepwiseConfig.steps_per_period(3, 0)
    with pytest.raises(ValueError):
        StepwiseConfig.steps_per_period({0: 2, 5: 4}, 0)


@pytest.mark.parametrize("config", _SCHEDULES)
@pytest.mark.parametrize("items", [1, 2, 3, 5, 7, 10, 13])
def test_stepwise_steps_per_period_matches_reference(config, items):
    # the compressed dict, read back via value_at, must equal the uncompressed
    # per-epoch simulation for every epoch (values converge once past all keys)
    result = StepwiseConfig.steps_per_period(config, items)
    ref = _ref_steps_per_epoch_list(config, items, 60)
    for epoch in range(60):
        assert StepwiseConfig.value_at(result, epoch) == ref[epoch]


@pytest.mark.parametrize("config", _SCHEDULES)
@pytest.mark.parametrize("items", [3, 7, 10])
@pytest.mark.parametrize("max_periods", [1, 2, 3, 5])
def test_stepwise_steps_per_period_max_periods(config, items, max_periods):
    result = StepwiseConfig.steps_per_period(config, items, max_periods=max_periods)
    ref = _ref_steps_per_epoch_list(config, items, max_periods)
    # no stored key may exceed the period cap, and values match within the cap
    assert all(k < max_periods for k in result)
    for epoch in range(max_periods):
        assert StepwiseConfig.value_at(result, epoch) == ref[epoch]


@pytest.mark.parametrize("config", _SCHEDULES)
@pytest.mark.parametrize("items", [3, 7, 10])
@pytest.mark.parametrize("max_total_steps", [1, 4, 10, 30])
def test_stepwise_steps_per_period_max_total_steps(config, items, max_total_steps):
    result = StepwiseConfig.steps_per_period(config, items, max_total_steps=max_total_steps)
    # every emitted period matches the uncompressed simulation
    ref = _ref_steps_per_epoch_list(config, items, 60)
    for epoch in result:
        assert result[epoch] == ref[epoch]
    # enumeration stops once cumulative steps reach the limit
    last_period = max(result)
    consumed = sum(
        StepwiseConfig.value_at(result, e) for e in range(last_period)
    )
    assert consumed < max_total_steps or last_period == 0


