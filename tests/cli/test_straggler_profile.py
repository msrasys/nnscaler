import math
from collections import Counter, deque
from types import SimpleNamespace

import pytest
import torch

from nnscaler.cli.straggler_profile import finite_pearson, finite_statistics
from nnscaler.cli.trainer import Trainer


def test_finite_statistics_ignores_unavailable_ranks():
    values = torch.tensor([1.0, math.nan, 4.0, math.inf, 2.0], dtype=torch.float64)

    stats = finite_statistics(values)

    assert stats['count'] == 3
    assert stats['p50'] == 2.0
    assert stats['max'] == 4.0
    assert stats['min'] == 1.0
    assert stats['max_rank'] == 2


def test_finite_statistics_returns_none_without_samples():
    assert finite_statistics(torch.tensor([math.nan, math.inf])) is None


def test_finite_pearson_uses_only_finite_pairs():
    x = torch.tensor([1.0, 2.0, math.nan, 3.0])
    y = torch.tensor([2.0, 4.0, 100.0, 6.0])

    assert finite_pearson(x, y) == pytest.approx(1.0)


def test_finite_pearson_is_undefined_for_constant_values():
    x = torch.tensor([1.0, 1.0, 1.0])
    y = torch.tensor([1.0, 2.0, 3.0])

    assert math.isnan(finite_pearson(x, y))


def test_rank_profile_attributes_late_arrivals_to_node_and_replica(caplog):
    caplog.set_level('INFO', logger='nnscaler.cli.trainer')
    trainer = Trainer.__new__(Trainer)
    trainer.train_args = SimpleNamespace(
        straggler_profile_topk=2,
        compute_config=SimpleNamespace(plan_ngpus=2),
    )
    trainer.train_status = SimpleNamespace(finished_train_steps=10)
    trainer.world_size = 4
    trainer.local_world_size = 2
    trainer._straggler_profile_rank_metadata = [
        {
            'rank': rank,
            'local_rank': rank % 2,
            'node_rank': rank // 2,
            'hostname': f'node-{rank // 2}',
            'pod_name': f'pod-{rank // 2}',
            'node_name': f'node-{rank // 2}',
        }
        for rank in range(4)
    ]
    trainer._straggler_profile_node_counts = Counter()
    trainer._straggler_profile_replica_counts = Counter()
    trainer._straggler_profile_history = deque(maxlen=100)
    names = (
        'train_step',
        'local_step',
        'reducer_last_ready_from_iter_s',
        'attention_cost_sum',
    )
    all_values = torch.tensor([
        [10.0, 11.0, 8.0, 100.0],
        [10.0, 11.0, 8.5, 100.0],
        [20.0, 21.0, 18.0, 200.0],
        [20.0, 21.0, 18.5, 200.0],
    ], dtype=torch.float64)

    correlations = trainer._log_straggler_profile(all_values, names)

    assert trainer._straggler_profile_node_counts == {'node-1': 1}
    assert trainer._straggler_profile_replica_counts == {1: 1}
    assert correlations['corr_train_step_attention_cost_sum'] == pytest.approx(1.0)
    assert '"last_rank": 2' in caplog.text
    assert '"last_rank": 3' in caplog.text
