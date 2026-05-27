import torch
import pytest

from nnscaler import parallelize, ComputeConfig
import nnscaler

from tests.autodist.spmd_solver.test_follow import apply_rotary_pos_emb, Model as RopeModel
from tests.parallel_module.test_gencode import replace_all_device_with, _gencode_contains, print_gencode


def rope_m_policy(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    for node in get_pas_ops(graph):
        if node.fn == apply_rotary_pos_emb:
            # split the `m` dimension
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))


@replace_all_device_with('cpu')
def test_rope_buggy_anno(tmp_path):
    from nnscaler.graph.parser.register import CustomizedOps
    CustomizedOps.kOpMap.pop('tests.autodist.spmd_solver.test_follow.apply_rotary_pos_emb', None)
    nnscaler.register_op('b h s^ d^, b m s^ d^, s^ d^, s^ d^, s^ -> b h s^ d^, b m s^ d^')(apply_rotary_pos_emb)

    bsz, seq_len, head_num, hidden_dim = 2, 128, 8, 512
    head_dim = hidden_dim // head_num
    dummy_input = {
        'x': torch.rand(bsz, seq_len, hidden_dim),
        'cos': torch.rand(seq_len, head_dim),
        'sin': torch.rand(seq_len, head_dim),
        'position_ids': torch.arange(seq_len, dtype=torch.long),
    }
    m = RopeModel(head_num, hidden_dim)

    m.train()
    m_new = parallelize(
        m,
        dummy_input,
        rope_m_policy,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # because of the bad annotation, a wrong grad allreduce is inserted
    assert _gencode_contains(tmp_path, RopeModel, 0, r'nnscaler.runtime.adapter.nn.identity_allreduce')
    # def segment143(self, x_44, cos_45, sin_46, position_ids_47):
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 42, in forward,  bsz, seq_len, hidden_dim = x.shape
    #     im_output_91 = builtins.getattr(x_44, 'shape')
    #     getattr_1_41 = im_output_91[0]
    #     getattr_1_42 = im_output_91[1]
    #     getattr_1_43 = im_output_91[2]
    #     del im_output_91
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 43, in forward,  q = self.q_proj(x)
    #     linear_50 = torch.nn.functional.linear(x_44, self.q_proj_weight_49, bias=None)
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 44, in forward,  k = self.k_proj(x)
    #     linear_1_52 = torch.nn.functional.linear(x_44, self.k_proj_weight_51, bias=None)
    #     del x_44
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 45, in forward,  q = q.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     view_53 = torch.Tensor.view(linear_50, size=(getattr_1_41, getattr_1_42, 8, 64))
    #     del linear_50
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 45, in forward,  q = q.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     transpose_54 = torch.transpose(view_53, dim0=1, dim1=2)
    #     del view_53
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 46, in forward,  k = k.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     view_1_55 = torch.Tensor.view(linear_1_52, size=(getattr_1_41, getattr_1_42, 8, 64))
    #     del linear_1_52
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 46, in forward,  k = k.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     transpose_1_56 = torch.transpose(view_1_55, dim0=1, dim1=2)
    #     del view_1_55
    #     transpose_54 = nnscaler.runtime.adapter.nn.identity_allreduce(transpose_54, ranks=[0, 1])
    #     transpose_1_73 = nnscaler.runtime.adapter.nn.split_allgather(transpose_1_56, dim=1, ranks=[0, 1])
    #     del transpose_1_56
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 47, in forward,  q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    #     apply_rotary_pos_emb_57, apply_rotary_pos_emb_75 = tests.autodist.spmd_solver.test_follow.apply_rotary_pos_emb(transpose_54, transpose_1_73, cos_45, sin_46, position_ids_47)
    #     del cos_45, sin_46, position_ids_47, transpose_54, transpose_1_73
    #     apply_rotary_pos_emb_58 = nnscaler.runtime.adapter.nn.allgather_split(apply_rotary_pos_emb_75, dim=1, ranks=[0, 1])
    #     del apply_rotary_pos_emb_75
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 48, in forward,  out = q + k
    #     add_59 = torch.add(apply_rotary_pos_emb_57, apply_rotary_pos_emb_58, alpha=1)
    #     del apply_rotary_pos_emb_57, apply_rotary_pos_emb_58
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 49, in forward,  return out.sum()
    #     sum_1_48 = torch.sum(add_59)
    #     del add_59
    #     return sum_1_48


@replace_all_device_with('cpu')
def test_rope_correct_anno(tmp_path):
    from nnscaler.graph.parser.register import CustomizedOps
    CustomizedOps.kOpMap.pop('tests.autodist.spmd_solver.test_follow.apply_rotary_pos_emb', None)

    nnscaler.register_op('b h s^ d^ : /m, b m s^ d^ : /h, s^ d^, s^ d^, s^ -> b h s^ d^, b m s^ d^')(apply_rotary_pos_emb)

    bsz, seq_len, head_num, hidden_dim = 2, 128, 8, 512
    head_dim = hidden_dim // head_num
    dummy_input = {
        'x': torch.rand(bsz, seq_len, hidden_dim),
        'cos': torch.rand(seq_len, head_dim),
        'sin': torch.rand(seq_len, head_dim),
        'position_ids': torch.arange(seq_len, dtype=torch.long),
    }
    m = RopeModel(head_num, hidden_dim)

    m.train()
    m_new = parallelize(
        m,
        dummy_input,
        rope_m_policy,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # grad allreduce will not be inserted because the annotation is fixed
    assert not _gencode_contains(tmp_path, RopeModel, 0, r'nnscaler.runtime.adapter.nn.identity_allreduce')

    # def segment137(self, x_44, cos_45, sin_46, position_ids_47):
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 42, in forward,  bsz, seq_len, hidden_dim = x.shape
    #     im_output_89 = builtins.getattr(x_44, 'shape')
    #     getattr_1_41 = im_output_89[0]
    #     getattr_1_42 = im_output_89[1]
    #     getattr_1_43 = im_output_89[2]
    #     del im_output_89
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 43, in forward,  q = self.q_proj(x)
    #     linear_50 = torch.nn.functional.linear(x_44, self.q_proj_weight_49, bias=None)
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 44, in forward,  k = self.k_proj(x)
    #     linear_1_52 = torch.nn.functional.linear(x_44, self.k_proj_weight_51, bias=None)
    #     del x_44
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 45, in forward,  q = q.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     view_53 = torch.Tensor.view(linear_50, size=(getattr_1_41, getattr_1_42, 8, 64))
    #     del linear_50
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 45, in forward,  q = q.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     transpose_54 = torch.transpose(view_53, dim0=1, dim1=2)
    #     del view_53
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 46, in forward,  k = k.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     view_1_55 = torch.Tensor.view(linear_1_52, size=(getattr_1_41, getattr_1_42, 8, 64))
    #     del linear_1_52
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 46, in forward,  k = k.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
    #     transpose_1_56 = torch.transpose(view_1_55, dim0=1, dim1=2)
    #     del view_1_55
    #     transpose_1_73 = nnscaler.runtime.adapter.nn.split_allgather(transpose_1_56, dim=1, ranks=[0, 1])
    #     del transpose_1_56
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 47, in forward,  q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    #     apply_rotary_pos_emb_57, apply_rotary_pos_emb_75 = tests.autodist.spmd_solver.test_follow.apply_rotary_pos_emb(transpose_54, transpose_1_73, cos_45, sin_46, position_ids_47)
    #     del cos_45, sin_46, position_ids_47, transpose_54, transpose_1_73
    #     apply_rotary_pos_emb_58 = nnscaler.runtime.adapter.nn.allgather_split(apply_rotary_pos_emb_75, dim=1, ranks=[0, 1])
    #     del apply_rotary_pos_emb_75
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 48, in forward,  out = q + k
    #     add_59 = torch.add(apply_rotary_pos_emb_57, apply_rotary_pos_emb_58, alpha=1)
    #     del apply_rotary_pos_emb_57, apply_rotary_pos_emb_58
    #     # File "/data/weijiangxu/nnscaler/tests/autodist/spmd_solver/test_follow.py", line 49, in forward,  return out.sum()
    #     sum_1_48 = torch.sum(add_59)
    #     del add_59
    #     return sum_1_48


class ValuePartitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, input):
        out = input @ self.weights
        return out


def value_partition_policy(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    for node in get_pas_ops(graph):
        if node.fn == torch.matmul:
            # split the `m` dimension
            yield OpPlan(node, partition=OpPartition(input=0, dim=1))


@replace_all_device_with('cpu')
def test_value_partition_anno(tmp_path):
    m = ValuePartitionModel()

    m.train()
    m_new = parallelize(
        m,
        {'input': torch.randn(4, 4)},
        value_partition_policy,
        ComputeConfig(2, 4, use_end2end=False),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert _gencode_contains(tmp_path, ValuePartitionModel, 0, r'nnscaler.runtime.adapter.nn.allreduce_identity')

    # code looks like:
    # def segment40(self, input_9):
    #     input_15 = nnscaler.runtime.adapter.nn.split_allgather(input_9, dim=1, ranks=[0, 1])
    #     del input_9
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode.py", line 2505, in forward,  out = input @ self.weights
    #     matmul_19 = torch.matmul(input_15, self.weights_17)
    #     del input_15
    #     matmul_10 = nnscaler.runtime.adapter.nn.allreduce_identity(matmul_19, ranks=[0, 1])
    #     del matmul_19
    #     return matmul_10


@nnscaler.register_op('a b : /, b c -> a c')
def _shared_skip_op(x, w):
    """Custom op that asks nnscaler to skip grad all-reduce on input 0."""
    return x @ w


class _SharedInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, raw_x):
        # The same `x` is consumed by two operators:
        #   - `_shared_skip_op`: annotation declares input 0 = ': /'
        #     so its grad-reduce should be skipped.
        #   - `torch.matmul`: standard op, full grad-reduce expected.
        x = torch.nn.functional.linear(raw_x, self.w0)
        a = _shared_skip_op(x, self.w1)
        b = torch.matmul(x, self.w2)
        return (a + b).sum()


def _shared_input_partition_policy(graph, cfg):
    """Partition both consumers along weight-dim 1 (= identifier 'c'),
    which causes `x` (replicated input) to need grad reduction on the
    `torch.matmul` side. The `_shared_skip_op` side is annotated with `: /`,
    so its grad reduction must be skipped — but this MUST NOT alter the
    `torch.matmul` side's gradient bookkeeping.
    """
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    for node in get_pas_ops(graph):
        if node.fn == torch.matmul:
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))
        elif node.fn == _shared_skip_op:
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))


@replace_all_device_with('cpu')
def test_shared_input_with_no_grad_reduce_consumer(tmp_path):
    """Two consumers of an internal activation `x`:
      * `_shared_skip_op` is annotated with `: /` -> grad-reduce skipped;
      * `torch.matmul` is a standard op -> grad-reduce required.

    Both are partitioned along weight-dim 1 (identifier `c`), so `x` is
    replicated across both ranks for both consumers. After fix, parallelize
    should succeed and emit one `identity_allreduce` on the matmul
    side.
    """
    m = _SharedInputModel()
    m.train()
    parallelize(
        m,
        {'raw_x': torch.randn(4, 4)},
        _shared_input_partition_policy,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # expected exactly one identity_allreduce (on torch.matmul's x input)
    assert len(_gencode_contains(tmp_path, _SharedInputModel, 0, r'.*identity_allreduce.*')) == 1

    # code looks like:
    # def segment110(self, raw_x_20):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 223, in forward,  x = torch.nn.functional.linear(raw_x, self.w0)
    #     linear_23 = torch.nn.functional.linear(raw_x_20, self.w0_22, bias=None)
    #     del raw_x_20
    #     # create at IRAdapterGener:autoref, comment before transformation: activation
    #     linear_52, linear_53 = nnscaler.runtime.function.multiref(linear_23, times=2)
    #     del linear_23
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 224, in forward,  a = _shared_skip_op(x, self.w1)
    #     _shared_skip_op_42 = tests.parallel_module.test_gencode_partition._shared_skip_op(linear_52, self.w1_40)
    #     del linear_52
    #     linear_53 = nnscaler.runtime.adapter.nn.identity_allreduce(linear_53, ranks=[0, 1])
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 225, in forward,  b = torch.matmul(x, self.w2)
    #     matmul_46 = torch.matmul(linear_53, self.w2_44)
    #     del linear_53
    #     _shared_skip_op_25 = nnscaler.runtime.adapter.nn.allgather_split(_shared_skip_op_42, dim=1, ranks=[0, 1])
    #     del _shared_skip_op_42
    #     matmul_27 = nnscaler.runtime.adapter.nn.allgather_split(matmul_46, dim=1, ranks=[0, 1])
    #     del matmul_46
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 226, in forward,  return (a + b).sum()
    #     add_28 = torch.add(_shared_skip_op_25, matmul_27, alpha=1)
    #     del _shared_skip_op_25, matmul_27
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 226, in forward,  return (a + b).sum()
    #     sum_1_21 = torch.sum(add_28)
    #     del add_28
    #     return sum_1_21


class _SharedInputModelReplicatedGradReduce(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, raw_x):
        # The same `x` is consumed by two operators:
        #   - `+`: replicated (and no grad reduce).
        #   - `torch.matmul`: standard op, full grad-reduce expected.
        x = torch.nn.functional.linear(raw_x, self.w0)
        a = x + self.w1
        b = torch.matmul(x, self.w2)
        return (a + b).sum()


@replace_all_device_with('cpu')
def test_shared_input_replicated_grad_reduce(tmp_path):
    """Two consumers of an internal activation `x`:
      * `+` replicated (and no grad reduce);
      * `torch.matmul` is a standard op -> grad-reduce required.
    """
    m = _SharedInputModelReplicatedGradReduce()
    m.train()
    parallelize(
        m,
        {'raw_x': torch.randn(4, 4)},
        _shared_input_partition_policy,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # expected exactly one identity_allreduce (on torch.matmul's x input)
    assert len(_gencode_contains(tmp_path, _SharedInputModelReplicatedGradReduce, 0, r'.*identity_allreduce.*')) == 1
    # code looks like:
    # def segment98(self, raw_x_20):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 309, in forward,  x = torch.nn.functional.linear(raw_x, self.w0)
    #     linear_23 = torch.nn.functional.linear(raw_x_20, self.w0_22, bias=None)
    #     del raw_x_20
    #     # create at IRAdapterGener:autoref, comment before transformation: activation
    #     linear_48, linear_49 = nnscaler.runtime.function.multiref(linear_23, times=2)
    #     del linear_23
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 310, in forward,  a = x + self.w1
    #     add_25 = torch.add(linear_48, self.w1_24, alpha=1)
    #     del linear_48
    #     linear_49 = nnscaler.runtime.adapter.nn.identity_allreduce(linear_49, ranks=[0, 1])
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 311, in forward,  b = torch.matmul(x, self.w2)
    #     matmul_42 = torch.matmul(linear_49, self.w2_40)
    #     del linear_49
    #     matmul_27 = nnscaler.runtime.adapter.nn.allgather_split(matmul_42, dim=1, ranks=[0, 1])
    #     del matmul_42
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 312, in forward,  return (a + b).sum()
    #     add_1_28 = torch.add(add_25, matmul_27, alpha=1)
    #     del add_25, matmul_27
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 312, in forward,  return (a + b).sum()
    #     sum_1_21 = torch.sum(add_1_28)
    #     del add_1_28
    #     return sum_1_21


class _SharedInputModelReplicatedNoGradReduce(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, raw_x):
        # The same `x` is consumed by two operators:
        #   - `+`: replicated (and no grad reduce).
        #   - `_shared_skip_op`: replicated and no grad reduce.
        x = torch.nn.functional.linear(raw_x, self.w0)
        a = x + self.w1
        b = _shared_skip_op(x, self.w2)
        return (a + b).sum()


@replace_all_device_with('cpu')
def test_shared_input_replicated_no_grad_reduce(tmp_path):
    """Two consumers of an internal activation `x`:
      * `+` replicated (and no grad reduce);
      * `_shared_skip_op` replicated and no grad reduce.
    """
    m = _SharedInputModelReplicatedNoGradReduce()
    m.train()
    parallelize(
        m,
        {'raw_x': torch.randn(4, 4)},
        _shared_input_partition_policy,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # expected no identity_allreduce (still have multiref added by `local_consumer_multiref` )
    assert not _gencode_contains(tmp_path, _SharedInputModelReplicatedNoGradReduce, 0, r'.*identity_allreduce.*')
    # code looks like:
    # def segment74(self, raw_x_20):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 371, in forward,  x = torch.nn.functional.linear(raw_x, self.w0)
    #     linear_23 = torch.nn.functional.linear(raw_x_20, self.w0_22, bias=None)
    #     del raw_x_20
    #     # created at IRAdapterGener:local_consumer_multiref
    #     linear_50, linear_54 = nnscaler.runtime.function.multiref(linear_23, times=2)
    #     del linear_23
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 372, in forward,  a = x + self.w1
    #     add_25 = torch.add(linear_50, self.w1_24, alpha=1)
    #     del linear_50
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 373, in forward,  b = _shared_skip_op(x, self.w2)
    #     _shared_skip_op_42 = tests.parallel_module.test_gencode_partition._shared_skip_op(linear_54, self.w2_40)
    #     del linear_54
    #     _shared_skip_op_27 = nnscaler.runtime.adapter.nn.allgather_split(_shared_skip_op_42, dim=1, ranks=[0, 1])
    #     del _shared_skip_op_42
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 374, in forward,  return (a + b).sum()
    #     add_1_28 = torch.add(add_25, _shared_skip_op_27, alpha=1)
    #     del add_25, _shared_skip_op_27
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 374, in forward,  return (a + b).sum()
    #     sum_1_21 = torch.sum(add_1_28)
    #     del add_1_28
    #     return sum_1_21


class MultileRRModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, input):
        x = torch.nn.functional.linear(input, self.w0)
        y = torch.nn.functional.linear(input, self.w1)
        z = torch.nn.functional.linear(input, self.w2)
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        return (a + b).sum()


def multirr_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    for node in get_pas_ops(graph):
        if node.fn == torch.matmul:
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))


@replace_all_device_with('cpu')
def test_multiref_multi_rr(tmp_path):
    """
    Test the case of two RR (replicate + allreduce) pattern
    """
    m = MultileRRModule()
    m.train()
    parallelize(
        m,
        {'input': torch.randn(4, 4)},
        multirr_pas,
        ComputeConfig(2, 4, use_end2end=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # no multiref are generated in `fn`
    assert not _gencode_contains(tmp_path, MultileRRModule, 0, r'activation')
    # code looks like:
    # def segment124(self, input_24):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 430, in forward,  x = torch.nn.functional.linear(input, self.w0)
    #     linear_27 = torch.nn.functional.linear(input_24, self.w0_26, bias=None)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 431, in forward,  y = torch.nn.functional.linear(input, self.w1)
    #     linear_1_29 = torch.nn.functional.linear(input_24, self.w1_28, bias=None)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 432, in forward,  z = torch.nn.functional.linear(input, self.w2)
    #     linear_2_31 = torch.nn.functional.linear(input_24, self.w2_30, bias=None)
    #     del input_24
    #     linear_27 = nnscaler.runtime.adapter.nn.identity_allreduce(linear_27, ranks=[0, 1])
    #     # created at IRAdapterGener:local_consumer_multiref
    #     linear_70, linear_74 = nnscaler.runtime.function.multiref(linear_27, times=2)
    #     del linear_27
    #     linear_1_48 = nnscaler.runtime.adapter.nn.split_allgather(linear_1_29, dim=1, ranks=[0, 1])
    #     del linear_1_29
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 433, in forward,  a = torch.matmul(x, y)
    #     matmul_50 = torch.matmul(linear_70, linear_1_48)
    #     del linear_70, linear_1_48
    #     linear_2_52 = nnscaler.runtime.adapter.nn.split_allgather(linear_2_31, dim=1, ranks=[0, 1])
    #     del linear_2_31
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 434, in forward,  b = torch.matmul(x, z)
    #     matmul_1_54 = torch.matmul(linear_74, linear_2_52)
    #     del linear_74, linear_2_52
    #     matmul_32 = nnscaler.runtime.adapter.nn.allgather_split(matmul_50, dim=1, ranks=[0, 1])
    #     del matmul_50
    #     matmul_1_33 = nnscaler.runtime.adapter.nn.allgather_split(matmul_1_54, dim=1, ranks=[0, 1])
    #     del matmul_1_54
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 435, in forward,  return (a + b).sum()
    #     add_34 = torch.add(matmul_32, matmul_1_33, alpha=1)
    #     del matmul_32, matmul_1_33
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 435, in forward,  return (a + b).sum()
    #     sum_1_25 = torch.sum(add_34)
    #     del add_34
    #     return sum_1_25


class StageOutputModel(torch.nn.Module):
    def __init__(self, no_grad_reduce=False):
        hidden_dim = 4
        self.no_grad_reduce = no_grad_reduce
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        y = torch.nn.functional.relu(x)
        if self.no_grad_reduce:
            z = _shared_skip_op(y, self.w)
        else:
            z = torch.matmul(y, self.w)
        w = y + z
        return w.sum()


def _stage_output_policy(graph, cfg):
    from nnscaler.policies import OpPlan, get_pas_ops, OpPartition

    matmul_found = False
    for node in get_pas_ops(graph):
        if matmul_found:
            yield OpPlan(node,stage_id=1)
        else:
            if node.fn == torch.matmul or node.fn == _shared_skip_op:
                yield OpPlan(node, stage_id=0, partition=OpPartition(input=1, dim=1))
                matmul_found = True
            else:
                yield OpPlan(node, stage_id=0)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('no_grad_reduce', [False, True])
def test_stage_output(tmp_path, no_grad_reduce):
    m = StageOutputModel(no_grad_reduce)
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4)},
        _stage_output_policy,
        ComputeConfig(4, 4, use_end2end=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_size': 2,
            }
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    if no_grad_reduce:
        assert not _gencode_contains(tmp_path, StageOutputModel, 0, r'nnscaler.runtime.function.multiref')
        # code looks like:
        # !!rank0
        # def segment17(self, x_13):
        #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 429, in forward,  y = torch.nn.functional.relu(x)
        #     relu_15 = torch.nn.functional.relu(x_13, inplace=False)
        #     del x_13
        #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 431, in forward,  z = _shared_skip_op(y, self.w)
        #     _shared_skip_op_40 = tests.parallel_module.test_gencode_partition._shared_skip_op(relu_15, self.w_38)
        #     _shared_skip_op_17 = nnscaler.runtime.adapter.nn.allgather_split(_shared_skip_op_40, dim=1, ranks=[0, 1])
        #     del _shared_skip_op_40
        #     return relu_15, _shared_skip_op_17
    else:
        assert _gencode_contains(tmp_path, StageOutputModel, 0, r'nnscaler.runtime.function.multiref')
        # code looks like:
        # !!rank0
        # def segment17(self, x_13):
        #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 430, in forward,  y = torch.nn.functional.relu(x)
        #     relu_15 = torch.nn.functional.relu(x_13, inplace=False)
        #     del x_13
        #     # create at IRAdapterGener:autoref, comment before transformation: activation
        #     relu_39 = nnscaler.runtime.function.multiref(relu_15, times=1)
        #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 434, in forward,  z = torch.matmul(y, self.w)
        #     matmul_42 = torch.matmul(relu_39, self.w_40)
        #     del relu_39
        #     matmul_17 = nnscaler.runtime.adapter.nn.allgather_split(matmul_42, dim=1, ranks=[0, 1])
        #     del matmul_42
        #     return relu_15, matmul_17


def _pp_shared_policy(graph, cfg):
    from nnscaler.policies import OpPlan, get_pas_ops, OpPartition

    relu_found = False
    for node in get_pas_ops(graph):
        if relu_found:
            yield OpPlan(node,stage_id=1)
        else:
            if node.fn == torch.nn.functional.relu:
                yield OpPlan(node, stage_id=0, partition=OpPartition(input=1, dim=1))
                relu_found = True
            else:
                yield OpPlan(node, stage_id=0)


class PPSharedModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        x = torch.matmul(x, self.w)
        x = torch.nn.functional.relu(x)
        x = torch.matmul(x, self.w)
        return x.sum()


@replace_all_device_with('cpu')
def test_pp_shared_model(tmp_path):
    m = PPSharedModel(4)
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4)},
        _pp_shared_policy,
        ComputeConfig(2, 4, use_end2end=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_size': 2,
            }
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # only the first stage (rank 0/2) should have the shared parameter,
    # and only it should register it as a parameter
    # (the other stage gets it as a non-parameter shared input)
    for rank in range(4):
        if rank % 2 == 0:
            assert _gencode_contains(tmp_path, PPSharedModel, rank, r'self.register_parameter\(')
        else:
            assert not _gencode_contains(tmp_path, PPSharedModel, rank, r'self.register_parameter\(')


class LossDataModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        l = x.sum()
        return l, l.data


def _loss_data_policy(graph, cfg):
    from nnscaler.policies import OpPlan, get_pas_ops, OpPartition

    for node in get_pas_ops(graph):
        if node.fn == torch.nn.functional.linear:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))
        if node.fn == torch.sum:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))


@replace_all_device_with('cpu')
def test_loss_insert_identity(tmp_path):
    m = LossDataModel()
    m.train()

    trace_data = torch.randn([2, 10], dtype=torch.float32)
    parallelize(
            m,
            {'x': trace_data},
            _loss_data_policy,
            ComputeConfig(2, 2, use_end2end=False),
            reuse='override',
            gen_savedir=tmp_path,
            load_module=False,
    )

    assert _gencode_contains(tmp_path, LossDataModel, 0, 'nnscaler.runtime.function.identity')
    # nnscaler.runtime.adapter.nn.split_allgather
    # and nnscaler.runtime.adapter.nn.allreduce_identity
    assert len(_gencode_contains(tmp_path, LossDataModel, 0, 'nnscaler.runtime.adapter.nn.')) == 2
    # code looks like:
    # def segment73(self, x_15):
    #     x_26 = nnscaler.runtime.adapter.nn.split_allgather(x_15, dim=0, ranks=[0, 1])
    #     del x_15
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 565, in forward,  x = self.fc(x)
    #     linear_28 = torch.nn.functional.linear(x_26, self.fc_weight_18, self.fc_bias_19)
    #     del x_26
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 566, in forward,  l = x.sum()
    #     sum_1_30 = torch.sum(linear_28)
    #     del linear_28
    #     sum_1_16 = nnscaler.runtime.adapter.nn.allreduce_identity(sum_1_30, ranks=[0, 1])
    #     del sum_1_30
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 568, in forward,  return l, l.data
    #     getattr_2_17 = builtins.getattr(sum_1_16, 'data')
    #     sum_1_34 = nnscaler.runtime.function.identity(sum_1_16)
    #     del sum_1_16
    #     return getattr_2_17, sum_1_34


from nnscaler.runtime.function import identity, multiref

class LossDataExplicitIdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        l = x.sum()
        z = identity(l)
        return z, l.data


@replace_all_device_with('cpu')
def test_loss_explicit_identity(tmp_path):
    m = LossDataExplicitIdentityModel()
    m.train()

    trace_data = torch.randn([2, 10], dtype=torch.float32)
    parallelize(
            m,
            {'x': trace_data},
            _loss_data_policy,
            ComputeConfig(2, 2, use_end2end=False),
            reuse='override',
            gen_savedir=tmp_path,
            load_module=False,
    )
    assert _gencode_contains(tmp_path, LossDataExplicitIdentityModel, 0, 'nnscaler.runtime.function.identity')
    # nnscaler.runtime.adapter.nn.split_allgather
    # and nnscaler.runtime.adapter.nn.allreduce_identity
    assert len(_gencode_contains(tmp_path, LossDataExplicitIdentityModel, 0, 'nnscaler.runtime.adapter.nn.')) == 2
    # code looks like:
    # def segment73(self, x_17):
    #     x_30 = nnscaler.runtime.adapter.nn.split_allgather(x_17, dim=0, ranks=[0, 1])
    #     del x_17
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 627, in forward,  x = self.fc(x)
    #     linear_32 = torch.nn.functional.linear(x_30, self.fc_weight_20, self.fc_bias_21)
    #     del x_30
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 628, in forward,  l = x.sum()
    #     sum_1_34 = torch.sum(linear_32)
    #     del linear_32
    #     sum_1_23 = nnscaler.runtime.adapter.nn.allreduce_identity(sum_1_34, ranks=[0, 1])
    #     del sum_1_34
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 629, in forward,  z = identity(l)
    #     identity_18 = nnscaler.runtime.function.identity(sum_1_23)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 630, in forward,  return z, l.data
    #     getattr_2_19 = builtins.getattr(sum_1_23, 'data')
    #     del sum_1_23
    #     return identity_18, getattr_2_19


class LossDataExplicitMultirefModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        l = x.sum()
        l1, l2 = multiref(l, times=2)
        return l1, l2.data


@replace_all_device_with('cpu')
def test_loss_explicit_multiref(tmp_path):
    """
    Test that if the user explicitly inserts a multiref for the loss,
    The gencode partitioning logic should still correctly insert the necessary adapters (e.g. allreduce_identity) to ensure the multiref outputs are correctly partitioned
    """
    m = LossDataExplicitMultirefModel()
    m.train()

    trace_data = torch.randn([2, 10], dtype=torch.float32)
    parallelize(
            m,
            {'x': trace_data},
            _loss_data_policy,
            ComputeConfig(2, 2, use_end2end=False),
            reuse='override',
            gen_savedir=tmp_path,
            load_module=False,
    )
    assert len(_gencode_contains(tmp_path, LossDataExplicitMultirefModel, 0, 'nnscaler.runtime.function.multiref')) == 1
    assert len(_gencode_contains(tmp_path, LossDataExplicitMultirefModel, 0, 'nnscaler.runtime.adapter.nn.split_allgather')) == 1
    assert len(_gencode_contains(tmp_path, LossDataExplicitMultirefModel, 0, 'nnscaler.runtime.adapter.nn.allreduce_identity')) == 1
    assert len(_gencode_contains(tmp_path, LossDataExplicitMultirefModel, 0, 'nnscaler.runtime.adapter.all_reduce')) == 1
    # code looks like:
    # def segment86(self, x_23):
    #     x_38 = nnscaler.runtime.adapter.nn.split_allgather(x_23, dim=0, ranks=[0, 1])
    #     del x_23
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 678, in forward,  x = self.fc(x)
    #     linear_40 = torch.nn.functional.linear(x_38, self.fc_weight_26, self.fc_bias_27)
    #     del x_38
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 679, in forward,  l = x.sum()
    #     sum_1_42 = torch.sum(linear_40)
    #     del linear_40
    #     # create at IRAdapterGener:autoref, comment before transformation: File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 680, in forward,  l1, l2 = multiref(l, times=2)
    #     multiref_52, multiref_53 = nnscaler.runtime.function.multiref(sum_1_42, times=2)
    #     del sum_1_42
    #     multiref_30 = nnscaler.runtime.adapter.all_reduce(multiref_53, ranks=[0, 1])
    #     del multiref_53
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_partition.py", line 681, in forward,  return l1, l2.data
    #     getattr_2_25 = builtins.getattr(multiref_30, 'data')
    #     del multiref_30
    #     multiref_24 = nnscaler.runtime.adapter.nn.allreduce_identity(multiref_52, ranks=[0, 1])
    #     del multiref_52
    #     return multiref_24, getattr_2_25
