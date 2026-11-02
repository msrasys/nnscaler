"""Run one task-owned command with bounded time, memory and disk usage."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def tree_bytes(directory):
    # Writers atomically rename temporary files while this scan runs. A path
    # disappearing is expected; permission/I/O errors still fail the guard.
    root = Path(directory)
    root.stat()
    seen = set()
    total = 0
    def onerror(error):
        if not isinstance(error, FileNotFoundError):
            raise error
    for current, dirs, files in os.walk(root, onerror=onerror):
        for path in [Path(current), *(Path(current) / name for name in files),
                     *(Path(current) / name for name in dirs if (Path(current) / name).is_symlink())]:
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += info.st_size
    return total


def sample(gpu):
    stat = os.statvfs(ROOT)
    disk_free = stat.f_bavail * stat.f_frsize / GIB
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024 / GIB
    external = os.environ.get('REF_KL_EXTERNAL_ARTIFACT_ROOT')
    if external and shutil.disk_usage(external).free < 2 * 1024**4:
        raise RuntimeError('PVC free space below 2 TiB')
    result = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'disk_free_gib': disk_free, 'mem_available_gib': available}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            group = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            if (group / 'memory.max').exists():
                limit = (group / 'memory.max').read_text().strip()
                if limit != 'max':
                    used = int((group / 'memory.current').read_text())
                    result['cgroup_remaining_gib'] = (int(limit) - used) / GIB
                    result['cgroup_limit_gib'] = int(limit) / GIB
    if gpu:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        result['gpu_used_gib'] = {line.split(',')[0].strip(): float(line.split(',')[1]) / 1024
                                  for line in output.strip().splitlines()}
    return result


def check(value):
    if value['disk_free_gib'] < 180:
        raise RuntimeError('disk free space below 180 GiB')
    if value['mem_available_gib'] < 256:
        raise RuntimeError('host memory available below 256 GiB')
    if value.get('cgroup_remaining_gib', float('inf')) < 128:
        raise RuntimeError('container memory headroom below 128 GiB')
    if any(x > 160 for x in value.get('gpu_used_gib', {}).values()):
        raise RuntimeError('GPU memory use above 160 GiB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    lock = None
    if args.gpu:
        lock = (ROOT / 'gpu.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        if procs:
            raise RuntimeError('GPU compute processes already exist; refusing to start')
    first = sample(args.gpu)
    check(first)
    result = {'command': command, 'cwd': str(Path.cwd()), 'initial_resources': first}
    checkout = os.environ.get('NNSCALER_TEST_ROOT')
    if checkout:
        result['checkout'] = checkout
        result['sha'] = subprocess.check_output(['git', '-C', checkout, 'rev-parse', 'HEAD'], text=True).strip()
        result['tracked_diff_sha256'] = hashlib.sha256(subprocess.check_output(
            ['git', '-C', checkout, 'diff', 'HEAD', '--'])).hexdigest()
    started = time.monotonic()
    reason = None
    with args.log.open('w') as log, args.log.with_suffix('.resources.jsonl').open('w') as metrics:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result['pid'] = process.pid
        owned = {}

        def remember_children():
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            for child in children:
                try:
                    owned[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    pass

        def stop_owned_children():
            survivors = []
            for pid, created in owned.items():
                try:
                    child = psutil.Process(pid)
                    if child.create_time() == created and child.is_running():
                        child.terminate()
                        survivors.append(child)
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(survivors, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

        next_disk_size = 0
        try:
            while process.poll() is None:
                remember_children()
                elapsed = time.monotonic() - started
                if elapsed > args.timeout:
                    raise RuntimeError(f'command timeout after {args.timeout} seconds')
                current = sample(args.gpu)
                if elapsed >= next_disk_size:
                    current['task_bytes'] = tree_bytes(ROOT)
                    if current['task_bytes'] > float(os.environ.get('REF_KL_TASK_BUDGET_GIB', '120')) * GIB:
                        raise RuntimeError('task directory exceeds 120 GiB output budget')
                    if os.environ.get('REF_KL_EXTERNAL_ARTIFACT_ROOT'):
                        current['external_task_bytes'] = tree_bytes(os.environ['REF_KL_EXTERNAL_ARTIFACT_ROOT'])
                        if current['external_task_bytes'] > 950 * GIB:
                            raise RuntimeError('PVC task output exceeds 950 GiB')
                    next_disk_size = elapsed + 30
                metrics.write(json.dumps(current) + '\n')
                metrics.flush()
                check(current)
                time.sleep(5)
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
            remember_children()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            result.update(exit_code=process.wait(), duration_seconds=time.monotonic() - started,
                          guard_failure=reason)
            # A failed launcher must not leave its task-owned workers behind.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # torchrun can put workers in their own process groups. Match PID
            # creation times, so only descendants observed under this job stop.
            try:
                stop_owned_children()
            except OSError as cleanup_error:
                result['cleanup_warning'] = str(cleanup_error)
            result['owned_child_count'] = len(owned)
            if checkout:
                result['sha_at_end'] = subprocess.check_output(['git', '-C', checkout, 'rev-parse', 'HEAD'], text=True).strip()
                result['tracked_diff_sha256_at_end'] = hashlib.sha256(subprocess.check_output(
                    ['git', '-C', checkout, 'diff', 'HEAD', '--'])).hexdigest()
                if (result['sha_at_end'] != result['sha'] or
                        result['tracked_diff_sha256_at_end'] != result['tracked_diff_sha256']):
                    reason = result['guard_failure'] = 'tested source changed while command was running'
            args.log.with_suffix('.result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(124 if reason else result['exit_code'])


if __name__ == '__main__':
    main()
