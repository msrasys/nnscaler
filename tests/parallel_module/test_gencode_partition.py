import torch

from nnscaler import parallelize, ComputeConfig
import nnscaler

from tests.autodist.spmd_solver.test_follow import apply_rotary_pos_emb, Model as RopeModel
from tests.parallel_module.test_gencode import replace_all_device_with, _gencode_contains


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
