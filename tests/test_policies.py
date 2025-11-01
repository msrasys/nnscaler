#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from typing import *

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from nnscaler.policies import get_called_self_module_name, get_pas_ops
from tests.parallel_module.common import FFN
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode

from .utils import init_random, replace_all_device_with

MBS = 2
DIM = 16
LAYERS = 16

class MLP(nn.Module):
    def __init__(self, dim: int = DIM, nlayers: int = LAYERS):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss


def dummy_data():
    return {
        'data': torch.randn(
            MBS, DIM, device=torch.cuda.current_device()),
        'target': torch.rand(
            MBS, DIM, device=torch.cuda.current_device())
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_autodist():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            MLP(),
            {'data': dummy_data()},
            'autodist',
            ComputeConfig(2, 4, pas_config={
                    'update_freq': 1,
                    'task_name': 'test_autodist',
            }),
            gen_savedir=tempdir,
            load_module=False
        )
        assert m_new is None


def test_call_name():
    assert get_called_self_module_name('self.up_proj(x)') == 'up_proj'
    assert get_called_self_module_name('self.act_fn(self.gate_proj(x))') == 'act_fn'
    assert get_called_self_module_name('self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))') == 'down_proj'
    assert get_called_self_module_name('torch.tanh(x)') == ''
    assert get_called_self_module_name('x * y') == ''
    assert get_called_self_module_name('self.up_proj(x).transpose()') == ''


class FnPolicyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = FFN(4, 8)

    def forward(self, x):
        x = x * 2
        x =  self.ffn(x)
        x = x + 3
        return x


def megatron_ffn_policy(graph, cfg):
    from nnscaler.ir import IRSubTensor
    from nnscaler.policies import OpPlan, OpPartition

    for node in get_pas_ops(graph):
        if FFN not in node.module_class_chain: # work on FFN module
            continue

        if node.fn in [torch.tanh, torch.mul]:
            yield OpPlan(node, partition=OpPartition(input=0, dim=1))
            continue

        assert node.fn == torch.nn.functional.linear

        input1: IRSubTensor = node.input(1)
        if not input1.is_param():  # linear weight param
            continue

        # we will partition gate_proj/up_proj with column parallelism (tp=ngpus)
        # and partition down_proj with row parallelism (tp=ngpus)

        if input1.name.endswith('gate_proj.weight') or input1.name.endswith('up_proj.weight'):
            # gate_proj/up_proj
            # column parallelism
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
        elif input1.name.endswith('down_proj.weight'):
            # down_proj
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))


def megatron_ffn_policy_auto(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition

    linear_rank = 0
    for node in get_pas_ops(graph):
        if FFN not in node.module_class_chain: # work on FFN module
            continue

        if node.fn == torch.nn.functional.linear:
            if linear_rank in [0, 1]:
                # gate_proj/up_proj
                yield OpPlan(node, partition=OpPartition(input=1, dim=0))
            else:
                assert linear_rank == 2
                # down_proj
                yield OpPlan(node, partition=OpPartition(input=1, dim=1))
            linear_rank += 1
        else:
            # other ops
            yield OpPlan(node, partition='auto')


@replace_all_device_with('cpu')
@pytest.mark.parametrize('policy', [megatron_ffn_policy, megatron_ffn_policy_auto])
def test_codegen_fn(tmp_path, policy):
    parallelize(
        FnPolicyModule(),
        {'x': torch.randn(2, 4)},
        policy,
        ComputeConfig(2, 4),
        gen_savedir=tmp_path,
        load_module=False
    )
    for rank in range(4):
        module_class = _load_parallel_module_class(FnPolicyModule, gen_savedir=tmp_path, rank=rank)

        fullmap = {m.orig_name: m for m in module_class.attr_meta_maps[rank].values()}
        assert fullmap['ffn.gate_proj.weight'].shape == (8, 4) and fullmap['ffn.gate_proj.weight'].sub_shape == (4, 4)
        assert fullmap['ffn.up_proj.weight'].shape == (8, 4) and fullmap['ffn.up_proj.weight'].sub_shape == (4, 4)
        assert fullmap['ffn.down_proj.weight'].shape == (4, 8) and fullmap['ffn.down_proj.weight'].sub_shape == (4, 4)

        # will generate two communication ops
        # one for ffn input
        assert _gencode_contains(tmp_path, FnPolicyModule, rank, f'nnscaler.runtime.adapter.nn.identity_allreduce')
        # one for ffn output
        assert _gencode_contains(tmp_path, FnPolicyModule, rank, f'nnscaler.runtime.adapter.nn.allreduce_identity')

        assert len(_gencode_contains(tmp_path, FnPolicyModule, rank, f'nnscaler.runtime.adapter.nn.')) == 2

        # Generated code of rank 0 should looks like:

    #    def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
    #         super().__init__()
    #         # communication groups
    #         self.init_group(ranks=[0, 1])

    #         self.register_parameter('ffn_gate_proj_weight_49', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_gate_proj_weight_49', 5, True, 'ffn.gate_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_up_proj_weight_63', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_up_proj_weight_63', 11, True, 'ffn.up_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_down_proj_weight_77', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_down_proj_weight_77', 17, True, 'ffn.down_proj.weight', (4, 8), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self._post_init(init_params, build_buckets)

    #     def segment118(self, x_25):
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 1653, in forward,  x = x * 2
    #         mul_27 = torch.mul(x_25, 2)
    #         del x_25
    #         mul_27 = nnscaler.runtime.adapter.nn.identity_allreduce(mul_27, ranks=[0, 1])
    #         # created at IRAdapterGener:local_consumer_multiref
    #         mul_85, mul_89 = nnscaler.runtime.function.multiref(mul_27, times=2)
    #         del mul_27
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/common.py", line 119, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #         linear_51 = torch.nn.functional.linear(mul_85, self.ffn_gate_proj_weight_49, bias=None)
    #         del mul_85
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/common.py", line 119, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #         tanh_59 = torch.tanh(linear_51)
    #         del linear_51
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/common.py", line 119, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #         linear_1_65 = torch.nn.functional.linear(mul_89, self.ffn_up_proj_weight_63, bias=None)
    #         del mul_89
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/common.py", line 119, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #         mul_1_73 = torch.mul(tanh_59, linear_1_65)
    #         del tanh_59, linear_1_65
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/common.py", line 119, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #         linear_2_79 = torch.nn.functional.linear(mul_1_73, self.ffn_down_proj_weight_77, bias=None)
    #         del mul_1_73
    #         linear_2_35 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_2_79, ranks=[0, 1])
    #         del linear_2_79
    #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 1655, in forward,  x = x + 3
    #         add_26 = torch.add(linear_2_35, 3, alpha=1)
    #         del linear_2_35
    #         return add_26


class FFNDropout(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(down_proj)


class FnPolicyModuleList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = torch.nn.ModuleList([
            FFNDropout(4, 8),
            FFNDropout(4, 8),
        ])

    def forward(self, x):
        x = x * 2
        for ffn in self.ffn:
            x = ffn(x)
        x = x + 3
        return torch.sum(x) # make sure output is scalar loss (required by pipeline parallelism)


def megatron_ffn_policy_list(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name

    for node in get_pas_ops(graph):
        if FFNDropout not in node.module_class_chain: # work on FFN module
            continue

        ffn_idx = get_layer_index(node.fqn)
        module_called = get_called_self_module_name(node.call_expr)

        if node.fn == torch.nn.functional.linear:
            if module_called in ['gate_proj', 'up_proj']:
                # gate_proj/up_proj
                yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition=OpPartition(input=1, dim=0))
            else:
                # down_proj
                yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition=OpPartition(input=1, dim=1))
        else:
            # other ops
            yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition='auto')


@replace_all_device_with('cpu')
def test_codegen_fn_pipeline(tmp_path):
    parallelize(
        FnPolicyModuleList(),
        {'x': torch.randn(4, 4)},
        # 'pp',
        megatron_ffn_policy_list,
        ComputeConfig(4, 4, use_end2end=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_size': 2,
            }
        ),
        gen_savedir=tmp_path,
        load_module=False
    )

    for rank in range(4):
        module_class = _load_parallel_module_class(FnPolicyModuleList, gen_savedir=tmp_path, rank=rank)

        fullmap = {m.orig_name: m for m in module_class.attr_meta_maps[rank].values()}
        tp_idx = rank // 2
        assert fullmap[f'ffn.{tp_idx}.gate_proj.weight'].shape == (8, 4) and fullmap[f'ffn.{tp_idx}.gate_proj.weight'].sub_shape == (4, 4)
        assert fullmap[f'ffn.{tp_idx}.up_proj.weight'].shape == (8, 4) and fullmap[f'ffn.{tp_idx}.up_proj.weight'].sub_shape == (4, 4)
        assert fullmap[f'ffn.{tp_idx}.down_proj.weight'].shape == (4, 8) and fullmap[f'ffn.{tp_idx}.down_proj.weight'].sub_shape == (4, 4)

        # will generate two communication ops
        # one for ffn input
        assert _gencode_contains(tmp_path, FnPolicyModuleList, rank, f'nnscaler.runtime.adapter.nn.identity_allreduce')
        # one for ffn output
        assert _gencode_contains(tmp_path, FnPolicyModuleList, rank, f'nnscaler.runtime.adapter.nn.allreduce_identity')

        assert len(_gencode_contains(tmp_path, FnPolicyModuleList, rank, f'nnscaler.runtime.adapter.nn.')) == 2
        assert len(_gencode_contains(tmp_path, FnPolicyModuleList, rank, r'ckpt.checkpoint\(recompute')) == 1
        assert len(_gencode_contains(tmp_path, FnPolicyModuleList, rank, r'def recompute\(')) == 1


    # Generated code of rank 0 looks like:
    # class GenModel(nnscaler.runtime.module.ParallelModule):
    #     use_scheduler = True
    #     nmicros_per_scheduler_step = 2
    #     rank = 0
    #     world_size = 4

    #     def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
    #         super().__init__()
    #         # communication groups
    #         self.init_group(ranks=[0, 1])
    #         self.init_group(ranks=[2, 3])

    #         self.register_parameter('ffn_0_gate_proj_weight_168', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_gate_proj_weight_168', 5, True, 'ffn.0.gate_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_0_up_proj_weight_182', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_up_proj_weight_182', 11, True, 'ffn.0.up_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_0_down_proj_weight_196', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_down_proj_weight_196', 17, True, 'ffn.0.down_proj.weight', (4, 8), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self._post_init(init_params, build_buckets)

    #     def segment79(self, x_49):
    #         # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 243, in forward,  x = x * 2
    #         mul_51 = torch.mul(x_49, 2)
    #         del x_49
    #         mul_51 = nnscaler.runtime.adapter.nn.identity_allreduce(mul_51, ranks=[0, 1])

    #         def recompute(mul_51):
    #             # created at IRAdapterGener:local_consumer_multiref
    #             mul_246, mul_250 = nnscaler.runtime.function.multiref(mul_51, times=2)
    #             del mul_51
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_170 = torch.nn.functional.linear(mul_246, self.ffn_0_gate_proj_weight_168, bias=None)
    #             del mul_246
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             tanh_178 = torch.tanh(linear_170)
    #             del linear_170
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_1_184 = torch.nn.functional.linear(mul_250, self.ffn_0_up_proj_weight_182, bias=None)
    #             del mul_250
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             mul_1_192 = torch.mul(tanh_178, linear_1_184)
    #             del tanh_178, linear_1_184
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_2_198 = torch.nn.functional.linear(mul_1_192, self.ffn_0_down_proj_weight_196, bias=None)
    #             del mul_1_192
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             ffn_0_dropout_training_21 = self.training
    #             linear_2_59 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_2_198, ranks=[0, 1])
    #             del linear_2_198
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             dropout_60 = torch.
    # nn.functional.dropout(linear_2_59, p=0.1, training=ffn_0_dropout_training_21, inplace=False)
    #             del linear_2_59
    #             return dropout_60

    #         dropout_60 = ckpt.checkpoint(recompute, mul_51, use_reentrant=False)
    #         return dropout_60

    #     def adapter196(self, dropout_60):
    #         dropout_236 = nnscaler.runtime.adapter.chunk(dropout_60, dim=1, ranks=[0, 1])
    #         _ = nnscaler.runtime.adapter.move(dropout_236, shape=(4, 2), dtype=torch.float32, src=0, dst=2)
    #         return

    #     def adapter207(self):
    #         gdropout_242 = nnscaler.runtime.adapter.move((), shape=(4, 2), dtype=torch.float32, src=2, dst=0)
    #         gdropout_85 = nnscaler.runtime.adapter.all_gather(gdropout_242, dim=1, ranks=[0, 1])
    #         return gdropout_85

    #     def adapter160(self):
    #         sum_1_50 = nnscaler.runtime.adapter.move((), shape=(), dtype=torch.float32, src=2, dst=0)
    #         return sum_1_50

    #     def _forward_impl(self, *args, **kwargs):
    #         raise NotImplementedError("Code of forward is not generated. You should use module.train_step/module.infer_step instead.")

    # ########## Generated Schedule Code ###########
    # import torch
    # import nnscaler

    # def _train_step(model, dataloader_71):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     x_49 = next(*(dataloader_71, ))
    #     dropout_60 = nnscaler.runtime.executor.fexecute('segment79', model.segment79, *(x_49, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter196, *(dropout_60, ), requires_grad=False)
    #     x_278 = next(*(dataloader_71, ))
    #     dropout_286 = nnscaler.runtime.executor.fexecute('segment79', model.segment79, *(x_278, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter196, *(dropout_286, ), requires_grad=False)
    #     gdropout_85 = nnscaler.runtime.executor.aexecute(model.adapter207, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gx_73 = nnscaler.runtime.executor.backward('segment79', (x_49, ), (dropout_60, ), (gdropout_85, ))
    #     del x_49, dropout_60, gdropout_85, gx_73
    #     gdropout_287 = nnscaler.runtime.executor.aexecute(model.adapter207, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     gx_279 = nnscaler.runtime.executor.backward('segment79', (x_278, ), (dropout_286, ), (gdropout_287, ))
    #     del x_278, dropout_286, gdropout_287, gx_279
    #     sum_1_50 = nnscaler.runtime.executor.aexecute(model.adapter160, *(), requires_grad=True)
    #     sum_1_306 = nnscaler.runtime.executor.aexecute(model.adapter160, *(), requires_grad=True)

    # def _infer_step(model, dataloader_71):
    #     _ = None
    #     x_49 = next(*(dataloader_71, ))
    #     dropout_60 = nnscaler.runtime.executor.fexecute('segment79', model.segment79, *(x_49, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter196, *(dropout_60, ), requires_grad=False)
    #     x_278 = next(*(dataloader_71, ))
    #     dropout_286 = nnscaler.runtime.executor.fexecute('segment79', model.segment79, *(x_278, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter196, *(dropout_286, ), requires_grad=False)
    #     sum_1_50 = nnscaler.runtime.executor.aexecute(model.adapter160, *(), requires_grad=False)
    #     sum_1_306 = nnscaler.runtime.executor.aexecute(model.adapter160, *(), requires_grad=False)
    #     return sum_1_50, sum_1_306
    assert True


class FnPolicyModuleSharedWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = torch.nn.Linear(4, 4, bias=False)
        self.ffn = torch.nn.ModuleList([
            FFNDropout(4, 8),
            FFNDropout(4, 8),
        ])
        self.output_projection = torch.nn.Linear(4, 4, bias=False)
        self.output_projection.weight = self.input_projection.weight  # share weight

    def forward(self, x):
        x = self.input_projection(x)
        for ffn in self.ffn:
            x = ffn(x)
        x = self.output_projection(x)
        return torch.sum(x) # make sure output is scalar loss (required by pipeline parallelism)



@replace_all_device_with('cpu')
def test_codegen_fn_pipeline_shared_weight(tmp_path):
    parallelize(
        FnPolicyModuleSharedWeight(),
        {'x': torch.randn(4, 4)},
        # 'pp',
        megatron_ffn_policy_list,
        ComputeConfig(4, 4, use_end2end=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_size': 2,
            }
        ),
        gen_savedir=tmp_path,
        load_module=False
    )
    for rank in range(2):
        # the input projection is multiref'ed
        assert _gencode_contains(tmp_path, FnPolicyModuleSharedWeight, rank, r'nnscaler.runtime.function.multiref\(self.input_projection')

    for rank in range(2, 4):
        # receive shared weight projection via identity
        assert _gencode_contains(tmp_path, FnPolicyModuleSharedWeight, rank, r'nnscaler.runtime.function.identity\(input_projection')

    # Generated code of rank 0 looks like:
    # class GenModel(nnscaler.runtime.module.ParallelModule):
    #     use_scheduler = True
    #     nmicros_per_scheduler_step = 2
    #     rank = 1
    #     world_size = 4

    #     def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
    #         super().__init__()
    #         # communication groups
    #         self.init_group(ranks=[0, 1])
    #         self.init_group(ranks=[2, 3])

    #         self.register_parameter('input_projection_weight_55', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('input_projection_weight_55', 3, True, 'input_projection.weight', (4, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_0_gate_proj_weight_189', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_gate_proj_weight_189', 7, True, 'ffn.0.gate_proj.weight', (8, 4), (slice(4, 8, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_0_up_proj_weight_203', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_up_proj_weight_203', 13, True, 'ffn.0.up_proj.weight', (8, 4), (slice(4, 8, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_0_down_proj_weight_217', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_0_down_proj_weight_217', 19, True, 'ffn.0.down_proj.weight', (4, 8), (slice(0, 4, None), slice(4, 8, None)), 1)
    #         self._post_init(init_params, build_buckets)

    #     def segment83(self, x_53):
    #         # shared param
    #         input_projection_weight_173, input_projection_weight_174 = nnscaler.runtime.function.multiref(self.input_projection_weight_55, times=2)
    #         # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 441, in forward,  x = self.input_projection(x)
    #         linear_56 = torch.nn.functional.linear(x_53, input_projection_weight_173, bias=None)
    #         del x_53, input_projection_weight_173
    #         linear_56 = nnscaler.runtime.adapter.nn.identity_allreduce(linear_56, ranks=[0, 1])

    #         def recompute(linear_56):
    #             # created at IRAdapterGener:local_consumer_multiref
    #             linear_278, linear_282 = nnscaler.runtime.function.multiref(linear_56, times=2)
    #             del linear_56
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_1_191 = torch.nn.functional.linear(linear_278, self.ffn_0_gate_proj_weight_189, bias=None)
    #             del linear_278
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             tanh_199 = torch.tanh(linear_1_191)
    #             del linear_1_191
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_2_205 = torch.nn.functional.linear(linear_282, self.ffn_0_up_proj_weight_203, bias=None)
    #             del linear_282
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             mul_213 = torch.mul(tanh_199, linear_2_205)
    #             del tanh_199, linear_2_205
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_3_219 = torch.nn.functional.linear(mul_213, self.ffn_0_down_proj_weight_217, bias=None)
    #             del mul_213
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             ffn_0_dropout_training_23 = self.training
    #             linear_3_64 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_3_219, ranks=[0, 1])
    #             del linear_3_219
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             dropout_65 = torch.nn.functional.dropout(linear_3_64, p=0.1, training=ffn_0_dropout_training_23, inplace=False)
    #             del linear_3_64
    #             return dropout_65

    #         dropout_65 = ckpt.checkpoint(recompute, linear_56, use_reentrant=False)
    #         return dropout_65, input_projection_weight_174

    #     def adapter190(self, input_projection_weight_174):
    #         input_projection_weight_257 = nnscaler.runtime.adapter.chunk(input_projection_weight_174, dim=1, ranks=[0, 1])
    #         _ = nnscaler.runtime.adapter.move(input_projection_weight_257, shape=(4, 2), dtype=torch.float32, src=1, dst=3)
    #         return

    #     def adapter234(self, dropout_65):
    #         dropout_265 = nnscaler.runtime.adapter.chunk(dropout_65, dim=1, ranks=[0, 1])
    #         _ = nnscaler.runtime.adapter.move(dropout_265, shape=(4, 2), dtype=torch.float32, src=1, dst=3)
    #         return

    #     def adapter245(self):
    #         gdropout_267 = nnscaler.runtime.adapter.move((), shape=(4, 2), dtype=torch.float32, src=3, dst=1)
    #         gdropout_92 = nnscaler.runtime.adapter.all_gather(gdropout_267, dim=1, ranks=[0, 1])
    #         return gdropout_92

    #     def adapter201(self):
    #         ginput_projection_weight_263 = nnscaler.runtime.adapter.move((), shape=(4, 2), dtype=torch.float32, src=3, dst=1)
    #         ginput_projection_weight_177 = nnscaler.runtime.adapter.all_gather(ginput_projection_weight_263, dim=1, ranks=[0, 1])
    #         return ginput_projection_weight_177

    #     def adapter214(self):
    #         sum_1_54 = nnscaler.runtime.adapter.move((), shape=(), dtype=torch.float32, src=3, dst=1)
    #         return sum_1_54

    #     def _forward_impl(self, *args, **kwargs):
    #         raise NotImplementedError("Code of forward is not generated. You should use module.train_step/module.infer_step instead.")

    # ########## Generated Schedule Code ###########
    # import torch
    # import nnscaler

    # def _train_step(model, dataloader_76):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     x_53 = next(*(dataloader_76, ))
    #     dropout_65, input_projection_weight_174 = nnscaler.runtime.executor.fexecute('segment83', model.segment83, *(x_53, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter190, *(input_projection_weight_174, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter234, *(dropout_65, ), requires_grad=False)
    #     x_302 = next(*(dataloader_76, ))
    #     dropout_310, input_projection_weight_314 = nnscaler.runtime.executor.fexecute('segment83', model.segment83, *(x_302, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter190, *(input_projection_weight_314, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter234, *(dropout_310, ), requires_grad=False)
    #     gdropout_92 = nnscaler.runtime.executor.aexecute(model.adapter245, *(), requires_grad=False)
    #     ginput_projection_weight_177 = nnscaler.runtime.executor.aexecute(model.adapter201, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gx_78 = nnscaler.runtime.executor.backward('segment83', (x_53, ), (dropout_65, input_projection_weight_174, ), (gdropout_92, ginput_projection_weight_177, ))
    #     del x_53, dropout_65, input_projection_weight_174, gdropout_92, ginput_projection_weight_177, gx_78
    #     gdropout_311 = nnscaler.runtime.executor.aexecute(model.adapter245, *(), requires_grad=False)
    #     ginput_projection_weight_315 = nnscaler.runtime.executor.aexecute(model.adapter201, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     gx_303 = nnscaler.runtime.executor.backward('segment83', (x_302, ), (dropout_310, input_projection_weight_314, ), (gdropout_311, ginput_projection_weight_315, ))
    #     del x_302, dropout_310, input_projection_weight_314, gdropout_311, ginput_projection_weight_315, gx_303
    #     sum_1_54 = nnscaler.runtime.executor.aexecute(model.adapter214, *(), requires_grad=True)
    #     sum_1_349 = nnscaler.runtime.executor.aexecute(model.adapter214, *(), requires_grad=True)
    #     return sum_1_54, sum_1_349

    # def _infer_step(model, dataloader_76):
    #     _ = None
    #     x_53 = next(*(dataloader_76, ))
    #     dropout_65, input_projection_weight_174 = nnscaler.runtime.executor.fexecute('segment83', model.segment83, *(x_53, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter190, *(input_projection_weight_174, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter234, *(dropout_65, ), requires_grad=False)
    #     x_302 = next(*(dataloader_76, ))
    #     dropout_310, input_projection_weight_314 = nnscaler.runtime.executor.fexecute('segment83', model.segment83, *(x_302, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter190, *(input_projection_weight_314, ), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter234, *(dropout_310, ), requires_grad=False)
    #     sum_1_54 = nnscaler.runtime.executor.aexecute(model.adapter214, *(), requires_grad=False)
    #     sum_1_349 = nnscaler.runtime.executor.aexecute(model.adapter214, *(), requires_grad=False)
    #     return sum_1_54, sum_1_349

    # Generated code of rank 2 looks like:

    # class GenModel(nnscaler.runtime.module.ParallelModule):
    #     use_scheduler = True
    #     nmicros_per_scheduler_step = 2
    #     rank = 2
    #     world_size = 4

    #     def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
    #         super().__init__()
    #         # communication groups
    #         self.init_group(ranks=[0, 1])
    #         self.init_group(ranks=[2, 3])

    #         self.register_parameter('ffn_1_gate_proj_weight_222', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_1_gate_proj_weight_222', 26, True, 'ffn.1.gate_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_1_up_proj_weight_236', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_1_up_proj_weight_236', 32, True, 'ffn.1.up_proj.weight', (8, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self.register_parameter('ffn_1_down_proj_weight_250', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #         self.add_full_map('ffn_1_down_proj_weight_250', 38, True, 'ffn.1.down_proj.weight', (4, 8), (slice(0, 4, None), slice(0, 4, None)), 1)

    #         self._post_init(init_params, build_buckets)

    #     def adapter190(self):
    #         input_projection_weight_256 = nnscaler.runtime.adapter.move((), shape=(4, 2), dtype=torch.float32, src=0, dst=2)
    #         input_projection_weight_174 = nnscaler.runtime.adapter.all_gather(input_projection_weight_256, dim=1, ranks=[2, 3])
    #         return input_projection_weight_174

    #     def adapter234(self):
    #         dropout_264 = nnscaler.runtime.adapter.move((), shape=(4, 2), dtype=torch.float32, src=0, dst=2)
    #         dropout_65 = nnscaler.runtime.adapter.all_gather(dropout_264, dim=1, ranks=[2, 3])
    #         return dropout_65

    #     def segment93(self, dropout_65, input_projection_weight_174):
    #         input_projection_weight_184 = nnscaler.runtime.function.identity(input_projection_weight_174)
    #         del input_projection_weight_174
    #         dropout_180 = nnscaler.runtime.function.identity(dropout_65)
    #         del dropout_65
    #         dropout_180 = nnscaler.runtime.adapter.nn.identity_allreduce(dropout_180, ranks=[2, 3])

    #         def recompute(dropout_180):
    #             # created at IRAdapterGener:local_consumer_multiref
    #             dropout_286, dropout_290 = nnscaler.runtime.function.multiref(dropout_180, times=2)
    #             del dropout_180
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_4_224 = torch.nn.functional.linear(dropout_286, self.ffn_1_gate_proj_weight_222, bias=None)
    #             del dropout_286
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             tanh_1_232 = torch.tanh(linear_4_224)
    #             del linear_4_224
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_5_238 = torch.nn.functional.linear(dropout_290, self.ffn_1_up_proj_weight_236, bias=None)
    #             del dropout_290
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             mul_1_246 = torch.mul(tanh_1_232, linear_5_238)
    #             del tanh_1_232, linear_5_238
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 230, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #             linear_6_252 = torch.nn.functional.linear(mul_1_246, self.ffn_1_down_proj_weight_250, bias=None)
    #             del mul_1_246
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             ffn_1_dropout_training_42 = self.training
    #             linear_6_73 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_6_252, ranks=[2, 3])
    #             del linear_6_252
    #             # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 231, in forward,  return self.dropout(down_proj)
    #             dropout_1_74 = torch.nn.functional.dropout(linear_6_73, p=0.1, training=ffn_1_dropout_training_42, inplace=False)
    #             del linear_6_73
    #             return dropout_1_74

    #         dropout_1_74 = ckpt.checkpoint(recompute, dropout_180, use_reentrant=False)
    #         del dropout_180
    #         # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 444, in forward,  x = self.output_projection(x)
    #         linear_7_75 = torch.nn.functional.linear(dropout_1_74, input_projection_weight_184, bias=None)
    #         del input_projection_weight_184, dropout_1_74
    #         # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 445, in forward,  return torch.sum(x) # make sure output is scalar loss (required by pipeline parallelism)
    #         sum_1_54 = torch.sum(linear_7_75)
    #         del linear_7_75
    #         return sum_1_54

    #     def adapter245(self, gdropout_92):
    #         gdropout_266 = nnscaler.runtime.adapter.chunk(gdropout_92, dim=1, ranks=[2, 3])
    #         _ = nnscaler.runtime.adapter.move(gdropout_266, shape=(4, 2), dtype=torch.float32, src=2, dst=0)
    #         return

    #     def adapter201(self, ginput_projection_weight_177):
    #         ginput_projection_weight_262 = nnscaler.runtime.adapter.chunk(ginput_projection_weight_177, dim=1, ranks=[2, 3])
    #         _ = nnscaler.runtime.adapter.move(ginput_projection_weight_262, shape=(4, 2), dtype=torch.float32, src=2, dst=0)
    #         return

    #     def adapter214(self, sum_1_54):
    #         _ = nnscaler.runtime.adapter.move(sum_1_54, shape=(), dtype=torch.float32, src=2, dst=0)
    #         return sum_1_54

    #     def _forward_impl(self, *args, **kwargs):
    #         raise NotImplementedError("Code of forward is not generated. You should use module.train_step/module.infer_step instead.")

    # ########## Generated Schedule Code ###########
    # import torch
    # import nnscaler

    # def _train_step(model, dataloader_76):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     input_projection_weight_174 = nnscaler.runtime.executor.aexecute(model.adapter190, *(), requires_grad=True)
    #     dropout_65 = nnscaler.runtime.executor.aexecute(model.adapter234, *(), requires_grad=True)
    #     sum_1_54 = nnscaler.runtime.executor.fexecute('segment93', model.segment93, *(dropout_65, input_projection_weight_174, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gdropout_92, ginput_projection_weight_177 = nnscaler.runtime.executor.backward('segment93', (dropout_65, input_projection_weight_174, ), (sum_1_54, ), (None, ))
    #     sum_1_54 = sum_1_54.detach()
    #     input_projection_weight_314 = nnscaler.runtime.executor.aexecute(model.adapter190, *(), requires_grad=True)
    #     dropout_310 = nnscaler.runtime.executor.aexecute(model.adapter234, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter245, *(gdropout_92, ), requires_grad=False)
    #     del dropout_65, gdropout_92
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter201, *(ginput_projection_weight_177, ), requires_grad=False)
    #     del input_projection_weight_174, ginput_projection_weight_177
    #     sum_1_349 = nnscaler.runtime.executor.fexecute('segment93', model.segment93, *(dropout_310, input_projection_weight_314, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     gdropout_311, ginput_projection_weight_315 = nnscaler.runtime.executor.backward('segment93', (dropout_310, input_projection_weight_314, ), (sum_1_349, ), (None, ))
    #     sum_1_349 = sum_1_349.detach()
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter245, *(gdropout_311, ), requires_grad=False)
    #     del dropout_310, gdropout_311
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter201, *(ginput_projection_weight_315, ), requires_grad=False)
    #     del input_projection_weight_314, ginput_projection_weight_315
    #     x_302 = next(*(dataloader_76, ))
    #     del x_302
    #     x_53 = next(*(dataloader_76, ))
    #     del x_53
    #     sum_1_54 = nnscaler.runtime.executor.aexecute(model.adapter214, *(sum_1_54, ), requires_grad=True)
    #     sum_1_349 = nnscaler.runtime.executor.aexecute(model.adapter214, *(sum_1_349, ), requires_grad=True)
    #     return sum_1_54, sum_1_349

    # def _infer_step(model, dataloader_76):
    #     _ = None
    #     input_projection_weight_174 = nnscaler.runtime.executor.aexecute(model.adapter190, *(), requires_grad=False)
    #     dropout_65 = nnscaler.runtime.executor.aexecute(model.adapter234, *(), requires_grad=False)
    #     sum_1_54 = nnscaler.runtime.executor.fexecute('segment93', model.segment93, *(dropout_65, input_projection_weight_174, ), requires_grad=False)
    #     input_projection_weight_314 = nnscaler.runtime.executor.aexecute(model.adapter190, *(), requires_grad=False)
    #     dropout_310 = nnscaler.runtime.executor.aexecute(model.adapter234, *(), requires_grad=False)
    #     sum_1_349 = nnscaler.runtime.executor.fexecute('segment93', model.segment93, *(dropout_310, input_projection_weight_314, ), requires_grad=False)
    #     x_302 = next(*(dataloader_76, ))
    #     del x_302
    #     x_53 = next(*(dataloader_76, ))
    #     del x_53
    #     sum_1_54 = nnscaler.runtime.executor.aexecute(model.adapter214, *(sum_1_54, ), requires_grad=False)
    #     sum_1_349 = nnscaler.runtime.executor.aexecute(model.adapter214, *(sum_1_349, ), requires_grad=False)
    #     return sum_1_54, sum_1_349


class FnPolicySharedWeightModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = torch.nn.Linear(4, 4, bias=False)
        self.output_projection = torch.nn.Linear(4, 4, bias=False)
        self.output_projection.weight = self.input_projection.weight  # share weight

    def forward(self, x):
        x = self.input_projection(x)
        x = self.output_projection(x)
        return x


def shared_weight_different_partition_policy(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_called_self_module_name

    for node in get_pas_ops(graph):
        module_called = get_called_self_module_name(node.call_expr)

        if node.fn == torch.nn.functional.linear and module_called == 'output_projection':
            # input_projection.weight is used two times with different partition
            # x = self.input_projection(x) --> no partition
            # x = self.output_projection(x) --> partition dim=1
            yield OpPlan(node, partition=OpPartition(input=1, dim=1))


@replace_all_device_with('cpu')
def test_codegen_fn_shared_weight(tmp_path):
    parallelize(
        FnPolicySharedWeightModule(),
        {'x': torch.randn(4, 4)},
        # 'pp',
        shared_weight_different_partition_policy,
        ComputeConfig(2, 4),
        gen_savedir=tmp_path,
        load_module=False
    )

    for rank in range(4):
        module_class = _load_parallel_module_class(FnPolicySharedWeightModule, gen_savedir=tmp_path, rank=rank)

        fullmap = {m.orig_name: m for m in module_class.attr_meta_maps[rank].values()}
        # the input projection is multiref'ed
        assert _gencode_contains(tmp_path, FnPolicySharedWeightModule, rank, r'nnscaler.runtime.function.multiref\(self.input_projection')
        # input_projection.weight will not be splitted
        # because it is multiref'ed
        assert fullmap['input_projection.weight'].shape == (4, 4) and fullmap['input_projection.weight'].sub_shape == (4, 4)

    # Generated code of rank 0 looks like:
    # def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
    #     super().__init__()
    #     # communication groups
    #     self.init_group(ranks=[0, 2])
    #     self.init_group(ranks=[1, 3])
    #     self.init_group(ranks=[0, 1])
    #     self.init_group(ranks=[2, 3])

    #     self.register_parameter('input_projection_weight_15', torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32)))
    #     self.add_full_map('input_projection_weight_15', 3, True, 'input_projection.weight', (4, 4), (slice(0, 4, None), slice(0, 4, None)), 1)

    #     self.wreducer80 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=False, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_ngroups=1)
    #     self.wreducer80.add_param(self.input_projection_weight_15)
    #     self.add_reducer(self.wreducer80)

    #     self._post_init(init_params, build_buckets)

    # def segment76(self, x_13):
    #     # shared param
    #     input_projection_weight_32, input_projection_weight_33 = nnscaler.runtime.function.multiref(self.input_projection_weight_15, times=2)
    #     # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 763, in forward,  x = self.input_projection(x)
    #     linear_16 = torch.nn.functional.linear(x_13, input_projection_weight_32, bias=None)
    #     del x_13, input_projection_weight_32
    #     linear_22 = nnscaler.runtime.adapter.nn.split_allgather(linear_16, dim=1, ranks=[0, 1])
    #     del linear_16
    #     input_projection_weight_37 = nnscaler.runtime.adapter.nn.split_allgather(input_projection_weight_33, dim=1, ranks=[0, 1])
    #     del input_projection_weight_33
    #     # File "/home/weijiangxu/MagicCube/tests/test_policies.py", line 764, in forward,  x = self.output_projection(x)
    #     linear_1_26 = torch.nn.functional.linear(linear_22, input_projection_weight_37, bias=None)
    #     del linear_22, input_projection_weight_37
    #     linear_1_14 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_1_26, ranks=[0, 1])
    #     del linear_1_26
    #     return linear_1_14


class FnPolicyModuleList2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = torch.nn.ModuleList([
            FFNDropout(4, 8),
            FFNDropout(4, 8),
            FFNDropout(4, 8),
            FFNDropout(4, 8),
        ])

    def forward(self, x):
        x = x * 2
        for ffn in self.ffn:
            x = ffn(x)
        x = x + 3
        return torch.sum(x) # make sure output is scalar loss (required by pipeline parallelism)


@replace_all_device_with('cpu')
def test_codegen_fn_pipeline2(tmp_path):
    parallelize(
        FnPolicyModuleList2(),
        {'x': torch.randn(4, 4)},
        # 'pp',
        megatron_ffn_policy_list,
        ComputeConfig(4, 4, use_end2end=True,
            pas_config={
                'pipeline_nmicros': 2,
                # 4 stages, with pp=2
                'pipeline_size': 2,
                'pipeline_scheduler': '1f1b_interleaved',
            }
        ),
        gen_savedir=tmp_path,
        load_module=False
    )
    # should successfully generate code without error
    assert True
