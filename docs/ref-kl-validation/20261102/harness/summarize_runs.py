"""Assert completed paired trajectories and summarize measured resources."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(name, reference):
    run = ROOT / 'runs' / name
    result = json.loads((run / 'train.result.json').read_text())
    assert result['exit_code'] == 0 and not result['guard_failure'], result
    plan = json.loads((run / 'plan.json').read_text())
    ranks = {}
    batch_sequences = []
    for rank in range(6):
        records = rows(run / 'audit' / f'rank{rank}.jsonl')
        baseline = rows(ROOT / 'runs' / reference / 'audit' / f'rank{rank}.jsonl')
        assert any(r['kind'] == 'complete' and r['horizon'] == 20 for r in records)
        actual_metrics = [r for r in records if r['kind'] == 'metrics']
        expected_metrics = [r for r in baseline if r['kind'] == 'metrics']
        assert [r['step'] for r in actual_metrics] == list(range(1, 21))
        diffs = {key: max(abs(a['values'][key] - b['values'][key])
                         for a, b in zip(actual_metrics, expected_metrics))
                 for key in ('train_loss', 'gnorm', 'lr', 'moe_loss', 'mtp_loss')}
        assert all(value == 0 for value in diffs.values()), diffs
        actual_batches = [r for r in records if r['kind'] == 'batch']
        expected_batches = [r for r in baseline if r['kind'] == 'batch']
        assert [r['index'] for r in actual_batches] == list(range(240))
        hashes = [r['sha256'] for r in actual_batches]
        assert hashes == [r['sha256'] for r in expected_batches]
        batch_sequences.append(hashes)
        actual_states = [r for r in records if r['kind'] == 'snapshot']
        expected_states = [r for r in baseline if r['kind'] == 'snapshot']
        assert [(r['stage'], r['step']) for r in actual_states] == [
            ('initial', 0), ('poststep', 1), ('poststep', 20)]
        assert [r['state'] for r in actual_states] == [r['state'] for r in expected_states]
        assert next(r['sha256'] for r in records if r['kind'] == 'initial_rng') == next(
            r['sha256'] for r in baseline if r['kind'] == 'initial_rng')
        frozen = next(r['state'] for r in records if r['kind'] == 'reference_initial')
        assert frozen
        frozen_checks = [r for r in records if r['kind'] == 'reference_poststep']
        assert [r['step'] for r in frozen_checks] == [1, 20]
        assert all(r['state'] == frozen for r in frozen_checks)
        memory = [r for r in records if r['kind'] == 'allocator_metrics' and r.get('tag') == 'train']
        assert [r['step'] for r in memory] == list(range(1, 21))
        ranks[rank] = dict(
            metric_max_abs_differences=diffs, batch_hashes_equal=True,
            images=sum(r['images'] for r in actual_batches),
            model_and_optimizer_fingerprints_equal=True, reference_buffers_frozen=True,
            peak_allocated_gib=max(r['values']['cuda_gb_allocated'] for r in memory),
            peak_reserved_gib=max(r['values']['cuda_gb_reserved'] for r in memory),
            median_step_s=statistics.median(r['values']['train_wall'] for r in actual_metrics if r['step'] >= 6),
        )
    assert all(sequence == batch_sequences[0] for sequence in batch_sequences)
    resources = rows(run / 'train.resources.jsonl')
    return dict(name=name, plan=plan, ranks=ranks,
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in ranks.values()),
        rank0_median_step_s=ranks[0]['median_step_s'],
        minimum_free_disk_gib=min(r['disk_free_gib'] for r in resources),
        minimum_container_headroom_gib=min(r.get('cgroup_remaining_gib', float('inf')) for r in resources),
        sampled_gpu_peak_gib=max(max(r.get('gpu_used_gib', {}).values(), default=0) for r in resources),
        limitations=['15B surrogate, not 150B cross-node training',
                     'deterministic attention; single paired measurement',
                     'parameter and optimizer hashes at steps 0/1/20; no direct full gradient snapshots'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--reference', default='baseline-20-content')
    args = parser.parse_args()
    output = [summarize(name, args.reference) for name in args.names]
    directory = ROOT / 'reports'
    directory.mkdir(exist_ok=True)
    for report in output:
        (directory / (report['name'] + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print({key: report[key] for key in ('name', 'peak_allocated_gib', 'rank0_median_step_s')})


if __name__ == '__main__':
    main()
