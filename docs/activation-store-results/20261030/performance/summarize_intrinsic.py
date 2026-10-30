import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
summary={}
for revision in ('v4','v5'):
    result={}
    for path in sorted(ROOT.glob(f'probe-{revision}-*.json')):
        value=json.loads(path.read_text())
        modes={}
        for mode, row in value['results'].items():
            walls=[sample['wall_ms'] for sample in row['samples']]
            modes[mode]=dict(median_ms=statistics.median(walls), mean_ms=statistics.mean(walls),
                min_ms=min(walls), max_ms=max(walls), submit_ms=row['median_submit_ms'],
                gpu_allocated_MiB=max(s['gpu_peak_allocated_bytes'] for s in row['samples'])/2**20,
                gpu_reserved_MiB=max(s['gpu_peak_reserved_bytes'] for s in row['samples'])/2**20,
                pinned_pool_MiB=row['pool']['allocated_pinned_bytes']/2**20,
                counters=row['counters'])
        result[path.stem.removeprefix(f'probe-{revision}-')]=dict(
            configuration=value['configuration'], bytes_per_direction=value['bytes_each_direction_per_iteration'],
            results=modes)
    summary[revision]=result
for name in summary['v5']:
    old=summary['v4'][name]['results'];new=summary['v5'][name]['results']
    print(name)
    for mode in new:
        print(mode,round(old[mode]['median_ms'],3),'->',round(new[mode]['median_ms'],3),
              'delta%',round((new[mode]['median_ms']/old[mode]['median_ms']-1)*100,1),
              'CPU',round(new[mode]['submit_ms'],3))
(ROOT/'intrinsic-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
