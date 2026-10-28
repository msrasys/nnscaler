# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace
import copy

import pytest
import torch

from nnscaler.runtime.module import AttrMeta, ParallelModule
from nnscaler.parallel import (_construct_optim_state_nonzero, _construct_optim_state_zero,
                               _construct_optim_state_zero3)


@pytest.mark.parametrize('case', ['missing', 'empty', 'keys', 'step'])
def test_zero_merge_rejects_incomplete_or_conflicting_bucket(case):
    fullmap = {'p_0': AttrMeta(0, True, 'p', (4,), (slice(0, 4),), 1, torch.float32, (4,))}
    states = [{'state': {0: {'step': torch.tensor(1.), 'momentum': torch.tensor([1., 2.])}},
               'param_groups': [{'params': [0]}]} for _ in range(2)]
    if case == 'missing':
        states[1]['state'].clear()
    elif case == 'empty':
        states[1]['state'][0] = {}
    elif case == 'keys':
        states[1]['state'][0]['variance'] = torch.ones(2)
    else:
        states[1]['state'][0]['step'] += 1
    zero_map = ({0: (0, 0, 4, (4,))}, {0: ([0, 1], 4)}, 1, None)
    with pytest.raises(ValueError, match='optimizer state'):
        ParallelModule.merge_opt_state_dicts([fullmap, fullmap], states, [zero_map, zero_map])


def _restore_module(with_bucket=True):
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    named = [(f'p{i}_{i}', param) for i, param in enumerate(params)]
    fullmap = {name: AttrMeta(i, True, f'p{i}', (2,), (slice(0, 2),), 1, torch.float32, (2,))
               for i, (name, _) in enumerate(named)}
    bucket = SimpleNamespace(params=params[:2], _contiguous_params=torch.empty(4))
    offsets = {id(p): i * 2 for i, p in enumerate(params[:2])}
    reducer = SimpleNamespace(params=params[:2], buckets=[bucket],
        get_param_info=lambda p: SimpleNamespace(bucket_param_buffer_start=offsets[id(p)]))
    return SimpleNamespace(
        fullmap=fullmap, dist_param_map={f'p{i}': f'p{i}' for i in range(3)},
        parameters=lambda: iter(params), named_parameters=lambda: iter(named),
        reducers=[reducer] if with_bucket else [],
        get_zero3_attr_meta=lambda name: None,
        parameters_for_optimizer=lambda: [torch.empty(4), params[2]] if with_bucket else params,
        _get_zero_subranks=lambda r: (0, [0]),
    )


@pytest.mark.parametrize('zero', [0, 1, 3])
@pytest.mark.parametrize('presence', ['empty_bucket', 'partial_bucket', 'all'])
def test_sparse_optimizer_restore_preserves_bucket_positions(zero, presence):
    module = _restore_module()
    source = {f'p{i}': {'step': torch.tensor(2.), 'momentum': torch.tensor([float(i), float(i + 1)])}
              for i in range(3)}
    if presence != 'all':
        del source['p1']
    if presence == 'empty_bucket':
        del source['p0']
    original = copy.deepcopy(source)
    construct = {0: _construct_optim_state_nonzero, 1: _construct_optim_state_zero,
                 3: _construct_optim_state_zero3}[zero]
    if presence == 'partial_bucket':
        with pytest.raises(ValueError, match='Inconsistent optimizer state'):
            construct(module, source)
    else:
        result = construct(module, source)
        outside = 2 if zero == 0 else 1
        assert set(result) == ({outside} if presence == 'empty_bucket' else set(range(outside + 1)))
        torch.testing.assert_close(result[outside]['momentum'], source['p2']['momentum'])
    for name, state in source.items():
        for key, value in state.items():
            torch.testing.assert_close(value, original[name][key])


def test_sparse_restore_without_reducer():
    result = _construct_optim_state_nonzero(_restore_module(False), {'p1': {'momentum': torch.ones(2)}})
    assert set(result) == {1}


@pytest.mark.parametrize('zero', [1, 3])
@pytest.mark.parametrize('conflict', ['keys', 'step'])
def test_zero_restore_rejects_conflicting_bucket_state(zero, conflict):
    source = {f'p{i}': {'step': torch.tensor(2.), 'momentum': torch.ones(2)} for i in range(3)}
    if conflict == 'keys':
        source['p1']['variance'] = torch.ones(2)
    else:
        source['p1']['step'] += 1
    construct = _construct_optim_state_zero if zero == 1 else _construct_optim_state_zero3
    with pytest.raises(ValueError, match='optimizer state'):
        construct(_restore_module(), source)


def test_optimizer_merge_does_not_rewrite_input_param_groups():
    fullmaps = [
        {'p': AttrMeta(0, True, 'p', (2,), (slice(0, 2),), 1, torch.float32, (2,))},
        {'q': AttrMeta(1, True, 'q', (2,), (slice(0, 2),), 1, torch.float32, (2,))},
    ]
    states = [{'state': {0: {'momentum': torch.ones(2)}}, 'param_groups': [{'params': [0]}]}
              for _ in range(2)]
    merged = ParallelModule.merge_opt_state_dicts(fullmaps, states)
    assert merged['param_groups'][0]['params'] == [0, 1]
    assert all(s['param_groups'][0]['params'] == [0] for s in states)


@pytest.mark.parametrize('empty_entry', [False, True])
def test_merge_rejects_a_missing_spatial_state_shard(empty_entry):
    fullmaps = [{f'p_{rank}': AttrMeta(rank, True, 'p', (4,),
                 (slice(rank * 2, rank * 2 + 2),), 1, torch.float32, (2,))}
                for rank in range(2)]
    states = [
        {'state': {0: {}} if empty_entry else {}, 'param_groups': [{'params': [0]}]},
        {'state': {0: {'momentum': torch.ones(2)}}, 'param_groups': [{'params': [0]}]},
    ]
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(fullmaps, states)


def test_zero_merge_can_recover_from_later_replica_after_empty_state():
    fullmap = {'p_0': AttrMeta(0, True, 'p', (2,), (slice(0, 2),), 1, torch.float32, (2,))}
    zero_map = ({0: 0}, {0: None}, 1, None)
    merged = ParallelModule.merge_opt_state_dicts(
        [fullmap, fullmap],
        [{'state': {0: {}}, 'param_groups': [{'params': [0]}]},
         {'state': {0: {'momentum': torch.tensor([3., 4.])}}, 'param_groups': [{'params': [0]}]}],
        [zero_map, zero_map],
    )
    torch.testing.assert_close(merged['state'][0]['momentum'], torch.tensor([3., 4.]))


def test_merge_sparse_optimizer_state():
    fullmap = {
        f'p{index}_{index}': AttrMeta(
            index,
            True,
            f'p{index}',
            (2,),
            (slice(0, 2),),
            1,
            torch.float32,
            (2,),
        )
        for index in range(2)
    }
    state = {
        'step': torch.tensor(1.0),
        'exp_avg': torch.ones(2),
        'exp_avg_sq': torch.ones(2),
    }

    merged = ParallelModule.merge_opt_state_dicts(
        [fullmap],
        [{'state': {0: state}, 'param_groups': [{'params': [0, 1]}]}],
    )

    assert set(merged['state']) == {0}
    assert merged['param_groups'][0]['params'] == [0, 1]

    sharded_fullmaps = [
        {
            f'p0_{rank}': AttrMeta(
                rank,
                True,
                'p0',
                (4,),
                (slice(rank * 2, (rank + 1) * 2),),
                1,
                torch.float32,
                (2,),
            )
        }
        for rank in range(2)
    ]
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(
            sharded_fullmaps,
            [
                {'state': {0: state}, 'param_groups': [{'params': [0]}]},
                {'state': {}, 'param_groups': [{'params': [0]}]},
            ],
        )

    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(
            sharded_fullmaps,
            [
                {'state': {0: {}}, 'param_groups': [{'params': [0]}]},
                {'state': {0: state}, 'param_groups': [{'params': [0]}]},
            ],
        )

    merged = ParallelModule.merge_opt_state_dicts(
        sharded_fullmaps,
        [
            {'state': {0: {}}, 'param_groups': [{'params': [0]}]},
            {'state': {0: {}}, 'param_groups': [{'params': [0]}]},
        ],
    )
    assert merged['state'] == {}
    assert merged['param_groups'][0]['params'] == [0]


def test_merge_sharded_optimizer_states_with_multiple_parameters():
    fullmaps = []
    optimizer_state_dicts = []
    for rank in range(2):
        fullmaps.append({
            'buffer': AttrMeta(
                2,
                False,
                'buffer',
                (1,),
                (slice(0, 1),),
                1,
                torch.float32,
                (1,),
            ),
            **{
                f'p{param}': AttrMeta(
                    param,
                    True,
                    f'p{param}',
                    (2,),
                    (slice(rank, rank + 1),),
                    1,
                    torch.float32,
                    (1,),
                )
                for param in range(2)
            },
        })
        optimizer_state_dicts.append({
            'state': {
                param: {
                    'step': torch.tensor(1.0),
                    'exp_avg': torch.tensor([rank * 10 + param], dtype=torch.float32),
                    'exp_avg_sq': torch.tensor([rank * 10 + param + 1], dtype=torch.float32),
                }
                for param in range(2)
            },
            'param_groups': [{'params': [0, 1]}],
        })

    merged = ParallelModule.merge_opt_state_dicts(fullmaps, optimizer_state_dicts)

    torch.testing.assert_close(merged['state'][0]['exp_avg'], torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(merged['state'][0]['exp_avg_sq'], torch.tensor([1.0, 11.0]))
    torch.testing.assert_close(merged['state'][1]['exp_avg'], torch.tensor([1.0, 11.0]))
    torch.testing.assert_close(merged['state'][1]['exp_avg_sq'], torch.tensor([2.0, 12.0]))
    assert merged['param_groups'][0]['params'] == [0, 1]


def test_merge_optimizer_states_with_rank_specific_parameter_layouts():
    fullmaps = [
        {
            'buffer': AttrMeta(
                3, False, 'buffer', (1,), (slice(0, 1),), 1, torch.float32, (1,)
            ),
            'p0': AttrMeta(
                0, True, 'p0', (1,), (slice(0, 1),), 1, torch.float32, (1,)
            ),
            'shared': AttrMeta(
                1, True, 'shared', (2,), (slice(0, 1),), 1, torch.float32, (1,)
            ),
        },
        {
            'p1': AttrMeta(
                2, True, 'p1', (1,), (slice(0, 1),), 1, torch.float32, (1,)
            ),
            'buffer': AttrMeta(
                3, False, 'buffer', (1,), (slice(0, 1),), 1, torch.float32, (1,)
            ),
            'shared': AttrMeta(
                1, True, 'shared', (2,), (slice(1, 2),), 1, torch.float32, (1,)
            ),
        },
    ]
    optimizer_state_dicts = [
        {
            'state': {
                0: {'momentum': torch.tensor([10.0])},
                1: {'momentum': torch.tensor([20.0])},
            },
            'param_groups': [{'params': [0, 1]}],
        },
        {
            'state': {
                0: {'momentum': torch.tensor([30.0])},
                1: {'momentum': torch.tensor([40.0])},
            },
            'param_groups': [{'params': [0, 1]}],
        },
    ]

    merged = ParallelModule.merge_opt_state_dicts(fullmaps, optimizer_state_dicts)

    torch.testing.assert_close(merged['state'][0]['momentum'], torch.tensor([10.0]))
    torch.testing.assert_close(merged['state'][1]['momentum'], torch.tensor([20.0, 40.0]))
    torch.testing.assert_close(merged['state'][2]['momentum'], torch.tensor([30.0]))
    assert merged['param_groups'][0]['params'] == [0, 1, 2]


def test_merge_replicated_optimizer_states():
    fullmaps = [
        {
            'p0': AttrMeta(
                rank, True, 'p0', (2,), (slice(0, 2),), 1, torch.float32, (2,)
            )
        }
        for rank in range(2)
    ]
    state = {'momentum': torch.tensor([1.0, 2.0])}

    merged = ParallelModule.merge_opt_state_dicts(
        fullmaps,
        [
            {'state': {0: state}, 'param_groups': [{'params': [0]}]},
            {'state': {0: state}, 'param_groups': [{'params': [0]}]},
        ],
    )
    torch.testing.assert_close(merged['state'][0]['momentum'], state['momentum'])

    with pytest.raises(ValueError, match='Conflict in merging p0.momentum'):
        ParallelModule.merge_opt_state_dicts(
            fullmaps,
            [
                {'state': {0: state}, 'param_groups': [{'params': [0]}]},
                {
                    'state': {0: {'momentum': torch.tensor([1.0, 3.0])}},
                    'param_groups': [{'params': [0]}],
                },
            ],
        )

