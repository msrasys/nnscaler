from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.utils.data import Dataset

import pytest

from nnscaler import register_op
from nnscaler.graph.schedule.schedplan import StreamConfig, StreamContext
from nnscaler.parallel import IRGraph, SchedulePlan
from nnscaler.graph.segment import IRSegment
from nnscaler.policies import get_pas_ops
from nnscaler.cli.trainer import Trainer

from tests.launch_torchrun import launch_torchrun

# hyper-parameters
hidden     = 2048
expert_dim = 4096
vocab      = 8000
seq_len    = 128
num_micro  = 4


# ============================================================================
#  Data
# ============================================================================

class OverlapDataset(Dataset):
    def __init__(self):
        torch.manual_seed(0)
        self.data = []
        for _ in range(num_micro*100):
            ids = torch.randint(0, vocab, (seq_len,))
            tgt = torch.randint(0, vocab, (seq_len,))
            self.data.append({'input_ids': ids, 'targets': tgt})

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# ============================================================================
#  AllToAll autograd function  (real NCCL all-to-all on dim 0)
# ============================================================================

class AllToAllFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ws = dist.get_world_size()
        chunks = [t.contiguous() for t in x.chunk(ws, dim=0)]
        out = [torch.empty_like(c) for c in chunks]
        dist.all_to_all(out, chunks)
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ws = dist.get_world_size()
        chunks = [t.contiguous() for t in grad.chunk(ws, dim=0)]
        out = [torch.empty_like(c) for c in chunks]
        dist.all_to_all(out, chunks)
        return torch.cat(out, dim=0), None


def all_to_all_fake(x: torch.Tensor) -> torch.Tensor:
    return x


@register_op('*^ -> *^', 'all_to_all', fake_fn=all_to_all_fake)
def all_to_all(x: torch.Tensor) -> torch.Tensor:
    return AllToAllFunction.apply(x)


# ============================================================================
#  Segment modules
# ============================================================================

class InputProj(nn.Module):
    """Seg 0 (compute): embedding look-up + linear projection."""
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))          # (B, S, H)


class CommGather(nn.Module):
    """Seg 1 (comm): flatten → all-to-all dispatch tokens to experts."""
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, S, H)
        h = h.reshape(-1, h.size(-1))                   # (B*S, H)
        h = all_to_all(h)                               # all-to-all
        return h                                        # (B*S, H)


class ExpertFFN(nn.Module):
    """Seg 2 (compute): expert feed-forward network."""
    def __init__(self, hidden: int, expert_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, hidden),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)                               # (B*S, H)


class CommScatter(nn.Module):
    """Seg 3 (comm): all-to-all to scatter results back."""
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return all_to_all(h)                 # (B*S, H)


class OutputHead(nn.Module):
    """Seg 4 (compute): reshape → linear → cross-entropy loss."""
    def __init__(self, hidden: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, h: torch.Tensor, bsz: int, seq_len: int,
                targets: torch.Tensor) -> torch.Tensor:
        h = h.reshape(bsz, seq_len, -1)
        logits = self.head(h)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ============================================================================
#  Segmented model wrapper
# ============================================================================
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # manually define segments for simplicity
        self.segments = nn.ModuleList([
            InputProj(vocab, hidden),         # 0  compute
            CommGather(),                     # 1  comm
            ExpertFFN(hidden, expert_dim),    # 2  compute
            CommScatter(),                    # 3  comm
            OutputHead(hidden, vocab),        # 4  compute
        ])
    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = data['input_ids']
        targets = data['targets']
        bsz, seq_len = input_ids.shape
        h = self.segments[0](input_ids)
        h = self.segments[1](h)
        h = self.segments[2](h)
        h = self.segments[3](h)
        return self.segments[4](h, bsz, seq_len, targets)


def toy_policy(graph, cfg):
    """
    Segment assignment for the toy model.
    """
    from nnscaler.policies import OpPlan

    if cfg.use_async_reducer:
        raise ValueError("Overlapping does not support async reducer.")

    for node in get_pas_ops(graph):
        if InputProj in node.module_class_chain or not node.module_class_chain:
            yield OpPlan(node, stage_id=0)
        elif CommGather in node.module_class_chain:
            yield OpPlan(node, stage_id=1)
        elif ExpertFFN in node.module_class_chain:
            yield OpPlan(node, stage_id=2)
        elif CommScatter in node.module_class_chain:
            yield OpPlan(node, stage_id=3)
        elif OutputHead in node.module_class_chain:
            yield OpPlan(node, stage_id=4)
        else:
            raise ValueError(f"Unexpected module in graph: {node.module_class_chain}")


def sched_overlap(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
    """
    1F1B overlapped schedule with two CUDA streams.

    Warmup:   full forward  mb 0
    1F1B:     for i in 1..N-1  →  forward(mb_i) interleaved with backward(mb_{i-1})
    Cooldown: full backward  mb N-1

    Within each 1F1B step the interleaving pairs forward and backward segments
    that land on *different* streams so they truly execute concurrently:

        comp_stream: fwd0 │ bwd4 │ fwd2 │ bwd2 │ fwd4 │ bwd0
        comm_stream:      │ fwd1 │ bwd3 │ fwd3 │ bwd1 │
                          ↕      ↕      ↕      ↕
                       overlap overlap overlap overlap

    Each stream waits for the other stream's previous event at pair boundaries.
    """

    if num_microbatches != 4:
        raise ValueError("This toy schedule is hard-coded for num_microbatches=4")
    if num_stages != 5:
        raise ValueError("This toy schedule is hard-coded for num_stages=5")

    segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]
    assert len(fsegs) == num_stages, f"Mismatch of forward segment number ({len(fsegs)}) with num_stages ({num_stages})"

    # describe schedule
    sched = SchedulePlan(graph, num_microbatches)
    comm_sc = StreamContext(stream='comm',wait_streams=['comp'])
    comp_sc = StreamContext(stream='comp', wait_streams=['comm'])
    forward_event = 'forward'
    backward_event = 'backward'

    fseg_stream_name = ['comp', 'comm', 'comp', 'comm', 'comp']
    sched.stream_config = StreamConfig(
        dataloader=comp_sc,
        zero_grad=comp_sc,
        inter_segment_move=comm_sc,
        result_broadcast=comm_sc,
        grad_reduce=comm_sc,
    )
    # warm up: full forward of mb 0
    for sid in range(num_stages):
        sched.add_segment(fsegs[sid], 0, step=sid,
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_event=forward_event)
            if sid == 0 else
            StreamContext(stream=fseg_stream_name[sid], wait_events=[forward_event], record_event=forward_event)
        )

    # 1F1B: forward(i) + backward(i-1)
    for micro_idx in range(1, num_microbatches):
        # Phase 1: fwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0], micro_idx, step=sid,
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_event=forward_event)
        )

        # Phase 2-5: overlapping pairs
        for fwd_s, bwd_s in [(1, 4), (2, 3), (3, 2), (4, 1)]:
            sid += 1
            sched.add_segment(fsegs[fwd_s], micro_idx, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[fwd_s], record_event=forward_event, wait_events=[forward_event]
                )
            )
            sid += 1
            sched.add_segment(fsegs[bwd_s].mirror, micro_idx - 1, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[bwd_s], record_event=backward_event, wait_events=[backward_event]
                )
            )

        # Phase 6: bwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0].mirror, micro_idx - 1, step=sid,
                stream_context=StreamContext(
                stream=fseg_stream_name[0], record_event=backward_event, wait_events=[backward_event]
            )
        )

    # Cooldown: backward last mb
    for stage in reversed(range(num_stages)):
        sid += 1
        sched.add_segment(fsegs[stage].mirror, num_microbatches - 1, step=sid,
            stream_context=
                StreamContext(stream='comp',wait_streams=['comm'], record_event=backward_event)
                if stage == num_stages - 1 else
                StreamContext(stream=fseg_stream_name[stage], wait_events=[backward_event], record_event=backward_event)
        )

    sched.finish()
    return sched


def trainer_worker_overlap(save_dir, config_file):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = config_path.stem
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{config_path.stem}'

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_overlap(tmp_path):
    launch_torchrun(2, trainer_worker_overlap, tmp_path, 'trainer_args_pipeline_overlap.yaml')
