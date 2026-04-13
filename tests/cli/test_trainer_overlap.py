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
from tests.parallel_module.common import assert_equal

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
        return torch.cat(out, dim=0)


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
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_events=[forward_event])
            if sid == 0 else
            StreamContext(stream=fseg_stream_name[sid], wait_events=[forward_event], record_events=[forward_event])
        )

    # 1F1B: forward(i) + backward(i-1)
    backward_event_recorded = False
    for micro_idx in range(1, num_microbatches):
        # Phase 1: fwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0], micro_idx, step=sid,
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_events=[forward_event])
        )

        # Phase 2-5: overlapping pairs
        for fwd_s, bwd_s in [(1, 4), (2, 3), (3, 2), (4, 1)]:
            sid += 1
            sched.add_segment(fsegs[fwd_s], micro_idx, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[fwd_s], record_events=[forward_event], wait_events=[forward_event]
                )
            )
            sid += 1
            sched.add_segment(fsegs[bwd_s].mirror, micro_idx - 1, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[bwd_s],
                    record_events=[backward_event],
                    wait_events=[backward_event] if backward_event_recorded else []
                )
            )
            backward_event_recorded = True

        # Phase 6: bwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0].mirror, micro_idx - 1, step=sid,
                stream_context=StreamContext(
                stream=fseg_stream_name[0], record_events=[backward_event], wait_events=[backward_event]
            )
        )

    # Cooldown: backward last mb
    for stage in reversed(range(num_stages)):
        sid += 1
        sched.add_segment(fsegs[stage].mirror, num_microbatches - 1, step=sid,
            stream_context=
                StreamContext(stream='comp',wait_streams=['comm'], record_events=[backward_event])
                if stage == num_stages - 1 else
                StreamContext(stream=fseg_stream_name[stage], wait_events=[backward_event], record_events=[backward_event])
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
    if trainer.model.use_scheduler:
        assert trainer.model.nmicros_per_scheduler_step == 4

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_overlap(tmp_path):
    launch_torchrun(2, trainer_worker_overlap, tmp_path, 'trainer_args_pipeline_overlap.yaml')
    launch_torchrun(2, trainer_worker_overlap, tmp_path, 'trainer_args_pipeline_overlap_gt.yaml')

    overlap_ckpt = torch.load(tmp_path / 'merged_trainer_args_pipeline_overlap.pt', weights_only=False)
    gt_ckpt = torch.load(tmp_path / 'merged_trainer_args_pipeline_overlap_gt.pt', weights_only=False)

    assert_equal(overlap_ckpt['model'], gt_ckpt['model'])
    assert_equal(overlap_ckpt['optimizer']['state'], gt_ckpt['optimizer']['state'])

    # overlapped train_step looks like:
    # InputProj(vocab, hidden),         # 0  compute
    # CommGather(),                     # 1  comm
    # ExpertFFN(hidden, expert_dim),    # 2  compute
    # CommScatter(),                    # 3  comm
    # OutputHead(hidden, vocab),        # 4  compute
    # def _train_step(model, dataloader_86):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         model.zero_grad()
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         data_57 = next(*(dataloader_86, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         getitem_1_64, linear_69 = nnscaler.runtime.executor.fexecute('InputProj_Segment', model.InputProj_Segment, *(data_57, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_71, getitem_1_109 = nnscaler.runtime.executor.fexecute('CommGather_Segment', model.CommGather_Segment, *(getitem_1_64, linear_69, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_64

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         linear_2_78, getitem_1_111 = nnscaler.runtime.executor.fexecute('ExpertFFN_Segment', model.ExpertFFN_Segment, *(all_to_all_71, getitem_1_109, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_109

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_1_79, getitem_1_113 = nnscaler.runtime.executor.fexecute('CommScatter_Segment', model.CommScatter_Segment, *(linear_2_78, getitem_1_111, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_111

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         cross_entropy_62 = nnscaler.runtime.executor.fexecute('OutputHead_Segment', model.OutputHead_Segment, *(all_to_all_1_79, getitem_1_113, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_113

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         data_132 = next(*(dataloader_86, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         getitem_1_135, linear_138 = nnscaler.runtime.executor.fexecute('InputProj_Segment', model.InputProj_Segment, *(data_132, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_146, getitem_1_149 = nnscaler.runtime.executor.fexecute('CommGather_Segment', model.CommGather_Segment, *(getitem_1_135, linear_138, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_135
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         gall_to_all_1_102 = nnscaler.runtime.executor.backward('OutputHead_Segment', (all_to_all_1_79, ), (cross_entropy_62, ), (None, ))
    #         cross_entropy_62 = cross_entropy_62.detach()
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         linear_2_157, getitem_1_160 = nnscaler.runtime.executor.fexecute('ExpertFFN_Segment', model.ExpertFFN_Segment, *(all_to_all_146, getitem_1_149, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_149
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_2_101 = nnscaler.runtime.executor.backward('CommScatter_Segment', (linear_2_78, ), (all_to_all_1_79, ), (gall_to_all_1_102, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_1_79, gall_to_all_1_102

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_1_168, getitem_1_171 = nnscaler.runtime.executor.fexecute('CommScatter_Segment', model.CommScatter_Segment, *(linear_2_157, getitem_1_160, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_160
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_94 = nnscaler.runtime.executor.backward('ExpertFFN_Segment', (all_to_all_71, ), (linear_2_78, ), (glinear_2_101, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_2_78, glinear_2_101

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         cross_entropy_179 = nnscaler.runtime.executor.fexecute('OutputHead_Segment', model.OutputHead_Segment, *(all_to_all_1_168, getitem_1_171, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_171
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_92 = nnscaler.runtime.executor.backward('CommGather_Segment', (linear_69, ), (all_to_all_71, ), (gall_to_all_94, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_71, gall_to_all_94
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('InputProj_Segment', (), (linear_69, ), (glinear_92, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_69, glinear_92

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         data_183 = next(*(dataloader_86, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         getitem_1_186, linear_189 = nnscaler.runtime.executor.fexecute('InputProj_Segment', model.InputProj_Segment, *(data_183, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_197, getitem_1_200 = nnscaler.runtime.executor.fexecute('CommGather_Segment', model.CommGather_Segment, *(getitem_1_186, linear_189, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_186
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_1_169 = nnscaler.runtime.executor.backward('OutputHead_Segment', (all_to_all_1_168, ), (cross_entropy_179, ), (None, ))
    #         cross_entropy_179 = cross_entropy_179.detach()
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         linear_2_208, getitem_1_211 = nnscaler.runtime.executor.fexecute('ExpertFFN_Segment', model.ExpertFFN_Segment, *(all_to_all_197, getitem_1_200, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_200
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_2_158 = nnscaler.runtime.executor.backward('CommScatter_Segment', (linear_2_157, ), (all_to_all_1_168, ), (gall_to_all_1_169, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_1_168, gall_to_all_1_169

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_1_219, getitem_1_222 = nnscaler.runtime.executor.fexecute('CommScatter_Segment', model.CommScatter_Segment, *(linear_2_208, getitem_1_211, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_211
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_147 = nnscaler.runtime.executor.backward('ExpertFFN_Segment', (all_to_all_146, ), (linear_2_157, ), (glinear_2_158, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_2_157, glinear_2_158

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         cross_entropy_230 = nnscaler.runtime.executor.fexecute('OutputHead_Segment', model.OutputHead_Segment, *(all_to_all_1_219, getitem_1_222, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_222
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_139 = nnscaler.runtime.executor.backward('CommGather_Segment', (linear_138, ), (all_to_all_146, ), (gall_to_all_147, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_146, gall_to_all_147
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('InputProj_Segment', (), (linear_138, ), (glinear_139, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_138, glinear_139

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         data_234 = next(*(dataloader_86, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         getitem_1_237, linear_240 = nnscaler.runtime.executor.fexecute('InputProj_Segment', model.InputProj_Segment, *(data_234, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_248, getitem_1_251 = nnscaler.runtime.executor.fexecute('CommGather_Segment', model.CommGather_Segment, *(getitem_1_237, linear_240, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_237
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_1_220 = nnscaler.runtime.executor.backward('OutputHead_Segment', (all_to_all_1_219, ), (cross_entropy_230, ), (None, ))
    #         cross_entropy_230 = cross_entropy_230.detach()
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         linear_2_259, getitem_1_262 = nnscaler.runtime.executor.fexecute('ExpertFFN_Segment', model.ExpertFFN_Segment, *(all_to_all_248, getitem_1_251, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_251
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_2_209 = nnscaler.runtime.executor.backward('CommScatter_Segment', (linear_2_208, ), (all_to_all_1_219, ), (gall_to_all_1_220, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_1_219, gall_to_all_1_220

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         all_to_all_1_270, getitem_1_273 = nnscaler.runtime.executor.fexecute('CommScatter_Segment', model.CommScatter_Segment, *(linear_2_259, getitem_1_262, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_262
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_198 = nnscaler.runtime.executor.backward('ExpertFFN_Segment', (all_to_all_197, ), (linear_2_208, ), (glinear_2_209, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_2_208, glinear_2_209

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         cross_entropy_281 = nnscaler.runtime.executor.fexecute('OutputHead_Segment', model.OutputHead_Segment, *(all_to_all_1_270, getitem_1_273, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del getitem_1_273
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_190 = nnscaler.runtime.executor.backward('CommGather_Segment', (linear_189, ), (all_to_all_197, ), (gall_to_all_198, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_197, gall_to_all_198
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('InputProj_Segment', (), (linear_189, ), (glinear_190, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_189, glinear_190
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         gall_to_all_1_271 = nnscaler.runtime.executor.backward('OutputHead_Segment', (all_to_all_1_270, ), (cross_entropy_281, ), (None, ))
    #         cross_entropy_281 = cross_entropy_281.detach()
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_2_260 = nnscaler.runtime.executor.backward('CommScatter_Segment', (linear_2_259, ), (all_to_all_1_270, ), (gall_to_all_1_271, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_1_270, gall_to_all_1_271
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         gall_to_all_249 = nnscaler.runtime.executor.backward('ExpertFFN_Segment', (all_to_all_248, ), (linear_2_259, ), (glinear_2_260, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_2_259, glinear_2_260
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         glinear_241 = nnscaler.runtime.executor.backward('CommGather_Segment', (linear_240, ), (all_to_all_248, ), (gall_to_all_249, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del all_to_all_248, gall_to_all_249
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('InputProj_Segment', (), (linear_240, ), (glinear_241, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del linear_240, glinear_241

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp'))
    #         _ = nnscaler.runtime.executor.aexecute(model.reducer263, *(), requires_grad=False)

    #     return cross_entropy_62, cross_entropy_179, cross_entropy_230, cross_entropy_281
