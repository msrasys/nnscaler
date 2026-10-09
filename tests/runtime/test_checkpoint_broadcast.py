from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nnscaler.runtime.checkpoint_broadcast import _flat_chunks, broadcast_checkpoint_tensors
from nnscaler.utils import run_on_rank0


@pytest.mark.parametrize('budget', [1, 3, 17, 1000])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'step', 'expand', 'scalar', 'empty'])
def test_flat_chunks_preserve_logical_order_with_bounded_copies(budget, layout):
    x = torch.arange(9 * 11).reshape(9, 11)
    x = {'contiguous': x, 'transpose': x.T, 'step': x[::2, ::3],
         'expand': x[:1].expand(5, 11), 'scalar': torch.tensor(7), 'empty': x[:0]}[layout]
    pieces = list(_flat_chunks(x, budget))
    assert all(p.is_contiguous() and p.numel() <= budget for p in pieces)
    actual = torch.cat(pieces) if pieces else torch.empty(0, dtype=x.dtype)
    assert torch.equal(actual, x.reshape(-1))


def _inputs():
    return [torch.arange(221, dtype=torch.float32).reshape(13, 17).T,
            torch.arange(42, dtype=torch.float64)[::3],
            torch.tensor(9, dtype=torch.int64), torch.zeros((0, 3)),
            torch.tensor([True, False, True]),
            torch.arange(15, dtype=torch.bfloat16)]


def _worker(rank, world, rendezvous, backend, folder):
    torch.set_num_threads(1)
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    dist.init_process_group(backend, init_method=rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))
    try:
        for members in [(0, 1, 2), (1, 2, 3), (1,)]:
            group = dist.new_group(list(members), backend=backend)
            if rank not in members:
                continue
            for budget in [32, 127, 1024]:
                for source_on_cuda in ([False, True] if backend == 'nccl' else [False]):
                    expected = _inputs()
                    source = ([t.to(device) if source_on_cuda else t for t in expected]
                              if rank == members[0] else None)
                    actual = broadcast_checkpoint_tensors(
                        source, [(tuple(t.shape), t.dtype) for t in expected],
                        src=members[0], group=group, device=device, chunk_bytes=budget)
                    if backend == 'nccl':
                        # Immediate dependent work tests stream readiness.
                        actual = [t + torch.zeros_like(t) for t in actual]
                    assert all(torch.equal(a.cpu(), b) for a, b in zip(actual, expected))
                    assert all(a.dtype == b.dtype and a.device == device for a, b in zip(actual, expected))
        dist.barrier()
        if backend == 'nccl':
            # Exercise the default 64MiB budget with repeated slot reuse on a
            # non-default caller stream, rather than only tiny test chunks.
            size = 3 * (64 * 1024 * 1024 // 4) + 19
            expected = torch.arange(size, dtype=torch.float32)
            stream = torch.cuda.Stream(device=device)
            for source_on_cuda in [False, True]:
                with torch.cuda.stream(stream):
                    source = expected.to(device) if source_on_cuda else expected
                    actual, = broadcast_checkpoint_tensors(
                        [source] if rank == 0 else None,
                        [(tuple(expected.shape), expected.dtype)],
                        src=0, group=None, device=device,
                    )
                    assert torch.equal(actual.cpu(), expected)
            dist.barrier()
        counter = Path(folder) / 'rank0-calls.txt'
        def discover():
            with counter.open('a') as f:
                f.write(f'{rank}\n')
            return {'names': ['0.ckpt', '1.ckpt'], 'none': None}
        for _ in range(2):
            assert run_on_rank0(discover)['names'] == ['0.ckpt', '1.ckpt']
        if rank == 0:
            assert counter.read_text().splitlines() == ['0', '0']
        def fail():
            raise OSError('injected directory failure')
        with pytest.raises(RuntimeError, match='injected directory failure'):
            run_on_rank0(fail)
        with pytest.raises(RuntimeError, match='metadata operation failed'):
            run_on_rank0(lambda: (lambda: None))
        assert run_on_rank0(lambda: 17) == 17  # failure does not poison later calls
    finally:
        dist.destroy_process_group()


def test_cpu_distributed_broadcast_and_metadata(tmp_path):
    mp.spawn(_worker, args=(4, f'file://{tmp_path}/gloo-init', 'gloo', str(tmp_path)),
             nprocs=4, join=True)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='requires four free GPUs')
def test_cuda_distributed_broadcast_stream_ordering(tmp_path):
    mp.spawn(_worker, args=(4, f'file://{tmp_path}/nccl-init', 'nccl', str(tmp_path)),
             nprocs=4, join=True)
