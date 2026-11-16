# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Measure communication interval unions in PyTorch Chrome traces.

Run with ``python -m nnscaler.profiler.stream_overlap TRACE.json[.gz] ...``.
Uncovered communication is a timeline diagnostic, not a prediction of speedup:
concurrent kernels can contend for SMs, and CPU launch stalls remain separate.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re


DEEPEP_KERNELS = r'(?i)(?:(?:deep_ep|deepep|intranode|internode).*?(?:dispatch|combine)|get_dispatch_layout)'
OTHER_COMM_KERNELS = r'(?i)(?:nccl|nvshmem|memcpy|memset)'


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(left, right):
    left, right = union(left), union(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def analyze(events, pattern=DEEPEP_KERNELS):
    communication = re.compile(pattern)
    other_communication = re.compile(OTHER_COMM_KERNELS)
    scopes = [(e['ts'], e['ts'] + e['dur']) for e in events
              if e.get('ph') == 'X' and e.get('name', '').startswith('stream_overlap/step/')]
    kernels = defaultdict(list)
    for event in events:
        if event.get('cat') == 'kernel' and event.get('ph') == 'X':
            kernels[str(event.get('args', {}).get('device', event.get('pid')))].append(event)
    results = {}
    for device, device_kernels in kernels.items():
        spans = scopes or [(min(e['ts'] for e in device_kernels),
                            max(e['ts'] + e['dur'] for e in device_kernels))]
        comm, compute, by_name, streams = [], [], defaultdict(list), defaultdict(list)
        for event in device_kernels:
            intervals = intersection([(event['ts'], event['ts'] + event['dur'])], spans)
            if not intervals:
                continue
            name = event['name']
            by_name[name].extend(intervals)
            streams[str(event.get('args', {}).get('stream'))].extend(intervals)
            if communication.search(name):
                comm.extend(intervals)
            elif not other_communication.search(name):
                compute.extend(intervals)
        wall, comm_time = duration(spans), duration(comm)
        hidden = duration(intersection(comm, compute))
        results[device] = dict(
            scope='annotated_training_steps' if scopes else 'gpu_kernel_span',
            wall_ms=wall / 1000, communication_ms=comm_time / 1000,
            concurrent_compute_ms=hidden / 1000,
            uncovered_communication_ms=(comm_time - hidden) / 1000,
            communication_fraction=comm_time / wall if wall else 0,
            uncovered_fraction=(comm_time - hidden) / wall if wall else 0,
            hidden_fraction=hidden / comm_time if comm_time else None,
            kernel_count=len(device_kernels),
            communication_kernel_names=sorted(n for n in by_name if communication.search(n)),
            stream_active_ms={s: duration(v) / 1000 for s, v in streams.items()},
            top_kernels_ms=sorted([(n, duration(v) / 1000) for n, v in by_name.items()],
                                  key=lambda item: item[1], reverse=True)[:30],
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--communication-regex', default=DEEPEP_KERNELS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {}
    for path in args.traces:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            trace = json.load(stream)
        report[str(path)] = analyze(trace['traceEvents'], args.communication_regex)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + '\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
