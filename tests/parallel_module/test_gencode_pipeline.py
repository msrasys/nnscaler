import re

import torch
import pytest

from nnscaler import parallelize, ComputeConfig
from tests.parallel_module.test_gencode import replace_all_device_with, _gencode_contains, print_gencode


class PPModule1(torch.nn.Module):
    def __init__(self, dim: int = 1024, nlayers: int = 4, *, return_type: int = 0):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))
        self.return_type = return_type

    def forward(self, data: torch.Tensor):
        x = data
        y = data.shape
        z = torch.abs(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) // 2:
                w = x.shape

        loss = torch.sum(x)
        if self.return_type == 0:
            return loss
        elif self.return_type == 1:
            return loss, y # the second return is a non-tensor from first layer
        elif self.return_type == 2:
            return loss, data.shape
        elif self.return_type == 3:
            return loss, {'data': data}
        elif self.return_type == 4:
            return loss, z
        elif self.return_type == 5:
            return loss, w  # the second return is a non-tensor from middle layer
        else:
            raise ValueError(f"Unsupported return_type: {self.return_type}")


def pp_pas(graph, cfg, nlayers_per_stage=2):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops

    last_stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            partition = None
            yield OpPlan(node, stage_id=layer_idx // nlayers_per_stage, partition=partition)
            last_stage_id = layer_idx // nlayers_per_stage
        else:
            yield OpPlan(node, stage_id=last_stage_id)


@replace_all_device_with('cpu')
def test_gencode_correct_dataloader_order(tmp_path):
    m = PPModule1(return_type=3)
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=2),
        compute_config= ComputeConfig(
            2, 2,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_nstages=2,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # the dataloader order should be the same across different ranks,
    # otherwise it may cause weird bugs.
    rank0_dataloader_ops = _gencode_contains(tmp_path, PPModule1, 0, r'.*next\(\*\(dataloader\_.*')
    rank1_dataloader_ops = _gencode_contains(tmp_path, PPModule1, 1, r'.*next\(\*\(dataloader\_.*')
    assert rank0_dataloader_ops == rank1_dataloader_ops
    # code looks like:
    # rank1:
    # def _train_step(model, dataloader_33):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     linear_1_28 = nnscaler.runtime.executor.aexecute(model.adapter43, *(), requires_grad=True)
    #     sum_1_24 = nnscaler.runtime.executor.fexecute('segment19', model.segment19, *(linear_1_28, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     glinear_1_38 = nnscaler.runtime.executor.backward('segment19', (linear_1_28, ), (sum_1_24, ), (None, ))
    #     sum_1_24 = sum_1_24.detach()
    #     linear_1_52 = nnscaler.runtime.executor.aexecute(model.adapter43, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter49, *(glinear_1_38, ), requires_grad=False)
    #     del linear_1_28, glinear_1_38
    #     sum_1_65 = nnscaler.runtime.executor.fexecute('segment19', model.segment19, *(linear_1_52, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     glinear_1_53 = nnscaler.runtime.executor.backward('segment19', (linear_1_52, ), (sum_1_65, ), (None, ))
    #     sum_1_65 = sum_1_65.detach()
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter49, *(glinear_1_53, ), requires_grad=False)
    #     del linear_1_52, glinear_1_53
    #     data_23 = next(*(dataloader_33, ))
    #     data_48 = next(*(dataloader_33, ))
    #     sum_1_24 = nnscaler.runtime.executor.aexecute(model.adapter56, *(sum_1_24, ), requires_grad=True)
    #     sum_1_65 = nnscaler.runtime.executor.aexecute(model.adapter56, *(sum_1_65, ), requires_grad=True)
    #     return [sum_1_24, {'data': data_23}], [sum_1_65, {'data': data_48}]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('return_type', [1, 2, 5])
def test_gencode_irobject_output(tmp_path, return_type):
    m = PPModule1(return_type=return_type)
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=1),
        compute_config= ComputeConfig(
            4, 8,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=4,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    if return_type == 1:
        src_rank = 0
    elif return_type == 2:
        src_rank = 3
    else:
        src_rank = 2
    for rank in range(4):
        assert _gencode_contains(tmp_path, PPModule1, rank, rf'nnscaler.runtime.adapter.broadcast_object\(.*, src={src_rank}, ranks=\[0, 1, 2, 3\]\)')

    for rank in range(4, 8):
        assert _gencode_contains(tmp_path, PPModule1, rank, rf'nnscaler.runtime.adapter.broadcast_object\(.*, src={src_rank + 4}, ranks=\[4, 5, 6, 7\]\)')


class PPModule2(torch.nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(4):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data: torch.Tensor):
        x = self.layers[0](data)
        xs = x.shape[0]
        x = self.layers[1](x) + xs
        xs = x.shape[0]
        x = self.layers[2](x) + xs
        xs = x.shape[0]
        x = self.layers[3](x) + xs
        loss = torch.sum(x)
        return loss


@replace_all_device_with('cpu')
def test_gencode_shared_irobject(tmp_path):
    m = PPModule2()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=1),
        compute_config= ComputeConfig(
            4, 8,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=4,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    for rank in range(3):
        assert _gencode_contains(tmp_path, PPModule2, rank, rf'nnscaler.runtime.adapter.move_object\(.*, src={rank}, dst={rank + 1}\)')

    for rank in range(4, 7):
        assert _gencode_contains(tmp_path, PPModule2, rank, rf'nnscaler.runtime.adapter.move_object\(.*, src={rank}, dst={rank + 1}\)')


class PPModule3(PPModule2):
    def forward(self, data: torch.Tensor):
        loss = super().forward(data)
        return loss, loss.data


@replace_all_device_with('cpu')
def test_gencode_shared_irobject_loss_data(tmp_path):
    """
    Test the case when the segment output is also the graph output,
    and the segment output is replaced with an inserted identiy op
    (by `_identity_segment_output` in `fn`)
    """
    m = PPModule3()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=1),
        compute_config= ComputeConfig(
            4, 8,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=4,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    for rank in range(3):
        assert _gencode_contains(tmp_path, PPModule3, rank, rf'sum_.* = nnscaler.runtime.adapter.broadcast\(\(\), shape=\(\), dtype=torch.float32, src=3, ranks=\[0, 1, 2, 3\]\)')
    assert _gencode_contains(tmp_path, PPModule3, 3, rf'sum_.* = nnscaler.runtime.adapter.broadcast\(sum_.*, shape=\(\), dtype=torch.float32, src=3, ranks=\[0, 1, 2, 3\]\)')

    for rank in range(4, 7):
        assert _gencode_contains(tmp_path, PPModule3, rank, rf'sum_.* = nnscaler.runtime.adapter.broadcast\(\(\), shape=\(\), dtype=torch.float32, src=7, ranks=\[4, 5, 6, 7\]\)')
    assert _gencode_contains(tmp_path, PPModule3, 7, rf'sum_.* = nnscaler.runtime.adapter.broadcast\(sum_.*, shape=\(\), dtype=torch.float32, src=7, ranks=\[4, 5, 6, 7\]\)')


class SplitSegmentModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(4):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))
        self.layerx = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, data: torch.Tensor):
        x = self.layers[0](data)
        x = self.layers[1](x) + x
        x = self.layers[2](x)
        x = self.layers[3](x)
        loss = torch.sum(x)
        return loss


def split_segment_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops

    last_stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn) // 2
            yield OpPlan(node, stage_id=layer_idx, partition=OpPartition(input=0, dim=0))
            last_stage_id = layer_idx
        else:
            if node.fn == torch.sum or node.fn == torch.add:
                yield OpPlan(node, stage_id=last_stage_id, partition=OpPartition(input=0, dim=0))
            else:
                yield OpPlan(node, stage_id=last_stage_id)


@replace_all_device_with('cpu')
def test_gencode_split_segment(tmp_path):
    m = SplitSegmentModule()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 64)},
        pas_policy=split_segment_pas,
        compute_config= ComputeConfig(
            4, 8,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=4,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    assert _gencode_contains(
        tmp_path, SplitSegmentModule, 2, r'nnscaler.runtime.adapter.nn.allreduce_identity\(sum_.*, ranks=\[2, 3\]\)'
    )
    assert not _gencode_contains(
        tmp_path, SplitSegmentModule, 2, r'nnscaler.runtime.adapter.chunk'
    )
    assert not _gencode_contains(
        tmp_path, SplitSegmentModule, 2, r'nnscaler.runtime.adapter.nn.split_allgather'
    )
    assert _gencode_contains(
        tmp_path, SplitSegmentModule, 6, r'nnscaler.runtime.adapter.nn.allreduce_identity\(sum_.*, ranks=\[6, 7\]\)'
    )


class PassThroughSegmentModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(6):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))
        self.layerx = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, data: torch.Tensor):
        x0 = self.layers[0](data)
        x1 = self.layers[1](x0)
        x2 = self.layers[2](x1)
        x3 = self.layers[3](x2) + x0
        x4 = self.layers[4](x3)
        x5 = self.layers[5](x4)
        loss = torch.sum(x5)
        return loss


def per_layer_segment_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops

    last_stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            last_stage_id = get_layer_index(node.fqn)
        yield OpPlan(node, stage_id=last_stage_id)


@replace_all_device_with('cpu')
def test_gencode_split_segment_pass_through(tmp_path):
    m = PassThroughSegmentModule()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 64)},
        pas_policy=per_layer_segment_pas,
        compute_config= ComputeConfig(
            6, 12,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=6,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert _gencode_contains(
        tmp_path, PassThroughSegmentModule, 1,
        r'def segment.*'
        r'nnscaler.runtime.function.identity.*'
        r'nnscaler.runtime.function.multiref.*'
        r'torch.nn.functional.linear.*'
        r'return linear_.*, linear_.*'
        r'def _train_step.*',
        flags=re.DOTALL
    )
    assert _gencode_contains(
        tmp_path, PassThroughSegmentModule, 2,
        r'def segment.*'
        r'nnscaler.runtime.function.identity.*'
        r'nnscaler.runtime.function.identity.*'
        r'torch.nn.functional.linear.*'
        r'return linear_.*, linear_.*'
        r'def _train_step.*',
        flags=re.DOTALL
    )
    assert _gencode_contains(
        tmp_path, PassThroughSegmentModule, 3,
        r'def segment.*'
        r'nnscaler.runtime.function.identity.*'
        r'nnscaler.runtime.function.identity.*'
        r'torch.nn.functional.linear.*'
        r'torch.add.*'
        r'return add_.*'
        r'def _train_step.*',
        flags=re.DOTALL
    )
    # code in rank 1:
    # def segment36(self, linear_35):
    #     linear_66 = nnscaler.runtime.function.identity(linear_35)
    #     del linear_35
    #     # created at IRAdapterGener:local_consumer_multiref
    #     linear_100, linear_104 = nnscaler.runtime.function.multiref(linear_66, times=2)
    #     del linear_66
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 313, in forward,  x1 = self.layers[1](x0)
    #     linear_1_37 = torch.nn.functional.linear(linear_100, self.layers_1_weight_36, bias=None)
    #     del linear_100
    #     linear_94 = nnscaler.runtime.function.identity(linear_104)
    #     del linear_104
    #     return linear_1_37, linear_94

    # code in rank 2:
    # def segment40(self, linear_1_37, linear_94):
    #     linear_1_78 = nnscaler.runtime.function.identity(linear_1_37)
    #     del linear_1_37
    #     linear_70 = nnscaler.runtime.function.identity(linear_94)
    #     del linear_94
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 314, in forward,  x2 = self.layers[2](x1)
    #     linear_2_39 = torch.nn.functional.linear(linear_1_78, self.layers_2_weight_38, bias=None)
    #     del linear_1_78
    #     return linear_2_39, linear_70

    # code in rank 3:
    # def segment45(self, linear_2_39, linear_70):
    #     linear_2_82 = nnscaler.runtime.function.identity(linear_2_39)
    #     del linear_2_39
    #     linear_74 = nnscaler.runtime.function.identity(linear_70)
    #     del linear_70
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 315, in forward,  x3 = self.layers[3](x2) + x0
    #     linear_3_41 = torch.nn.functional.linear(linear_2_82, self.layers_3_weight_40, bias=None)
    #     del linear_2_82
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 315, in forward,  x3 = self.layers[3](x2) + x0
    #     add_42 = torch.add(linear_3_41, linear_74, alpha=1)
    #     del linear_74, linear_3_41
    #     return add_42


class SharedOutputSegmentModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))
        self.w3 = torch.nn.Parameter(torch.randn(4, 4))
        self.w4 = torch.nn.Parameter(torch.randn(4, 4))
        self.w5 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, data: torch.Tensor):
        x0 = data + self.w0
        x1 = x0 + self.w1

        x2 = x1 + self.w2 + x0  # x0 is a shared output
        x3 = x2 + self.w3

        x4 = x3 + self.w4
        x5 = x4 + self.w5

        loss = torch.sum(x5)
        return loss


def _shared_output_partition_policy(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    stage_id = 0
    for node in get_pas_ops(graph):

        if node.fn == torch.add:
            if node.input(1).is_param() and node.input(1).name in ['w0', 'w1']:
                stage_id = 0
            elif node.input(1).is_param() and node.input(1).name in ['w2', 'w3']:
                stage_id = 1
            elif node.input(1).is_param() and node.input(1).name in ['w4', 'w5']:
                stage_id = 2
            yield OpPlan(node, stage_id=stage_id, partition=OpPartition(input=0, dim=0))
        else:
            yield OpPlan(node, stage_id=stage_id, partition=None)


@replace_all_device_with('cpu')
def test_split_segment_with_shared_output(tmp_path):
    """
    Shared output will prevent the segment output from being split,
    TODO: we should support splitting the shared output, but currently we don't support it.
    """
    m = SharedOutputSegmentModule()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(4, 4)},
        pas_policy=_shared_output_partition_policy,
        compute_config= ComputeConfig(
            6, 6,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=3,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert _gencode_contains(
        tmp_path, SharedOutputSegmentModule, 0, r'nnscaler.runtime.adapter.nn.allgather_split'
    )

    # rank 0:
    # def segment39(self, data_32):
    #     data_64 = nnscaler.runtime.adapter.chunk(data_32, dim=0, ranks=[0, 1])
    #     del data_32
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 437, in forward,  x0 = data + self.w0
    #     add_68 = torch.add(data_64, self.w0_66, alpha=1)
    #     del data_64
    #     # create at IRAdapterGener:autoref, comment before transformation: fn activation
    #     add_120, add_158 = nnscaler.runtime.function.multiref(add_68, times=2)
    #     del add_68
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 438, in forward,  x1 = x0 + self.w1
    #     add_1_72 = torch.add(add_120, self.w1_70, alpha=1)
    #     del add_120
    #     add_113 = nnscaler.runtime.adapter.nn.allgather_split(add_158, dim=0, ranks=[0, 1])
    #     del add_158
    #     # fn identity for segment output
    #     add_106 = nnscaler.runtime.function.identity(add_113)
    #     del add_113
    #     return add_106, add_1_72

    # def adapter202(self, add_106):
    #     add_162 = nnscaler.runtime.adapter.chunk(add_106, dim=0, ranks=[0, 1])
    #     _ = nnscaler.runtime.adapter.move(add_162, shape=(2, 4), dtype=torch.float32, src=0, dst=2)
    #     return

    # def adapter232(self, add_1_72):
    #     _ = nnscaler.runtime.adapter.move(add_1_72, shape=(2, 4), dtype=torch.float32, src=0, dst=2)
    #     return

    # rank 2:
    # def segment43(self, add_162, add_1_72):
    #     # created at: segment dispatch: fix identity
    #     add_1_128 = nnscaler.runtime.function.identity(add_1_72)
    #     del add_1_72
    #     # created at: segment dispatch: fix identity
    #     add_136 = nnscaler.runtime.function.identity(add_162)
    #     del add_162
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 440, in forward,  x2 = x1 + self.w2 + x0  # x0 is a shared output
    #     add_2_76 = torch.add(add_1_128, self.w2_74, alpha=1)
    #     del add_1_128
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 440, in forward,  x2 = x1 + self.w2 + x0  # x0 is a shared output
    #     add_3_78 = torch.add(add_2_76, add_136, alpha=1)
    #     del add_136, add_2_76
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 441, in forward,  x3 = x2 + self.w3
    #     add_4_82 = torch.add(add_3_78, self.w3_80, alpha=1)
    #     del add_3_78
    #     return add_4_82

    # def adapter202(self):
    #     add_162 = nnscaler.runtime.adapter.move((), shape=(2, 4), dtype=torch.float32, src=0, dst=2)
    #     return add_162

    # def adapter232(self):
    #     add_1_72 = nnscaler.runtime.adapter.move((), shape=(2, 4), dtype=torch.float32, src=0, dst=2)
    #     return add_1_72


class SharedInputSegmentModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))
        self.w3 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, data: dict[str, torch.Tensor]):
        x0 = data['data'] + self.w0
        x1 = x0 + self.w1

        x2 = x1 + self.w2
        x3 = x2 + self.w3

        loss = torch.sum(x3 + data['loss'])  # data is a shared input
        return loss


@replace_all_device_with('cpu')
def test_shared_dict_input(tmp_path):
    m = SharedInputSegmentModule()
    m.train()
    parallelize(
        m,
        {'data': {'data': torch.randn(4, 4), 'loss': torch.randn(4, 4)}},
        pas_policy=_shared_output_partition_policy,
        compute_config= ComputeConfig(
            2, 4,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # input data will not pass to rank1.
    # It will directly read from dataloader in rank1

    # rank1's segment takes the dict `data` as an input and does getitem('loss') on it
    assert _gencode_contains(
        tmp_path, SharedInputSegmentModule, 1, r"_operator.getitem\(data_\d+, 'loss'\)"
    )
    # the dict input is NOT transmitted from rank0 to rank1 via any adapter
    assert not _gencode_contains(
        tmp_path, SharedInputSegmentModule, 0, r'nnscaler.runtime.adapter.move_object'
    )
    assert not _gencode_contains(
        tmp_path, SharedInputSegmentModule, 1, r'nnscaler.runtime.adapter.move_object'
    )
    # data_27 = next(*(dataloader_40, ))
    # sum_1_28 = nnscaler.runtime.executor.fexecute('segment28', model.segment28, *(add_1_33, data_27, ), requires_grad=True)
    assert len(_gencode_contains(
        tmp_path, SharedInputSegmentModule, 1, r"data_.* = next\(\*\(dataloader_.*, \)\)\s*sum_.* = nnscaler.runtime.executor.fexecute\('segment.*', model.segment.*, \*\(add_.*, data_.*, \), requires_grad=True\)"
    )) == 2
    assert len(_gencode_contains(
        tmp_path, SharedInputSegmentModule, 1, r"data_.* = next\(\*\(dataloader_.*, \)\)\s*sum_.* = nnscaler.runtime.executor.fexecute\('segment.*', model.segment.*, \*\(add_.*, data_.*, \), requires_grad=False\)"
    )) == 2

    # code in rank 0:
    # def segment24(self, data_27):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 563, in forward,  x0 = data['data'] + self.w0
    #     getitem_29 = _operator.getitem(data_27, 'data')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 563, in forward,  x0 = data['data'] + self.w0
    #     add_31 = torch.add(getitem_29, self.w0_30, alpha=1)
    #     del getitem_29
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 564, in forward,  x1 = x0 + self.w1
    #     add_1_33 = torch.add(add_31, self.w1_32, alpha=1)
    #     del add_31
    #     return add_1_33

    # code in rank 1:
    # def segment28(self, add_1_33, data_27):
    #     add_1_53 = nnscaler.runtime.function.identity(add_1_33)
    #     del add_1_33
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 566, in forward,  x2 = x1 + self.w2
    #     add_2_35 = torch.add(add_1_53, self.w2_34, alpha=1)
    #     del add_1_53
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 567, in forward,  x3 = x2 + self.w3
    #     add_3_37 = torch.add(add_2_35, self.w3_36, alpha=1)
    #     del add_2_35
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 569, in forward,  loss = torch.sum(x3 + data['loss'])  # data is a shared input
    #     getitem_1_38 = _operator.getitem(data_27, 'loss')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 569, in forward,  loss = torch.sum(x3 + data['loss'])  # data is a shared input
    #     add_4_39 = torch.add(add_3_37, getitem_1_38, alpha=1)
    #     del add_3_37, getitem_1_38
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 569, in forward,  loss = torch.sum(x3 + data['loss'])  # data is a shared input
    #     sum_1_28 = torch.sum(add_4_39)
    #     del add_4_39
    #     return sum_1_28

    # def _train_step(model, dataloader_40):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     add_1_33 = nnscaler.runtime.executor.aexecute(model.adapter58, *(), requires_grad=True)
    #     data_27 = next(*(dataloader_40, ))
    #     sum_1_28 = nnscaler.runtime.executor.fexecute('segment28', model.segment28, *(add_1_33, data_27, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_1_45 = nnscaler.runtime.executor.backward('segment28', (add_1_33, ), (sum_1_28, ), (None, ))
    #     sum_1_28 = sum_1_28.detach()
    #     add_1_65 = nnscaler.runtime.executor.aexecute(model.adapter58, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter64, *(gadd_1_45, ), requires_grad=False)
    #     del add_1_33, gadd_1_45
    #     data_61 = next(*(dataloader_40, ))
    #     sum_1_80 = nnscaler.runtime.executor.fexecute('segment28', model.segment28, *(add_1_65, data_61, ), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     gadd_1_66 = nnscaler.runtime.executor.backward('segment28', (add_1_65, ), (sum_1_80, ), (None, ))
    #     sum_1_80 = sum_1_80.detach()
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter64, *(gadd_1_66, ), requires_grad=False)
    #     del add_1_65, gadd_1_66
    #     sum_1_28 = nnscaler.runtime.executor.aexecute(model.adapter71, *(sum_1_28, ), requires_grad=True)
    #     sum_1_80 = nnscaler.runtime.executor.aexecute(model.adapter71, *(sum_1_80, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.reducer179, *(), requires_grad=False)
    #     return sum_1_28, sum_1_80


class SharedInputSegmentWithIntermediateModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))
        self.w3 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, data: dict[str, torch.Tensor]):
        data_loss = data['loss']
        x0 = data['data'] + self.w0
        x1 = x0 + self.w1

        x2 = x1 + self.w2
        x3 = x2 + self.w3

        loss = torch.sum(x3 + data_loss)  # data_loss is shared
        return loss


@replace_all_device_with('cpu')
def test_shared_dict_input_with_intermediate(tmp_path):
    m = SharedInputSegmentWithIntermediateModule()
    m.train()
    parallelize(
        m,
        {'data': {'data': torch.randn(4, 4), 'loss': torch.randn(4, 4)}},
        pas_policy=_shared_output_partition_policy,
        compute_config= ComputeConfig(
            2, 4,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # rank1's segment reads `data` from its local dataloader and does getitem('loss')
    assert _gencode_contains(
        tmp_path, SharedInputSegmentWithIntermediateModule, 1, r"_operator.getitem\(data_\d+, 'loss'\)"
    )
    # rank0 no longer computes data['loss'] (it was recomputed in the consumer stage)
    assert not _gencode_contains(
        tmp_path, SharedInputSegmentWithIntermediateModule, 0, r"_operator.getitem\(data_\d+, 'loss'\)"
    )
    # the dict input is NOT transmitted between stages via any adapter
    assert not _gencode_contains(
        tmp_path, SharedInputSegmentWithIntermediateModule, 0, r'nnscaler.runtime.adapter.move_object'
    )
    assert not _gencode_contains(
        tmp_path, SharedInputSegmentWithIntermediateModule, 1, r'nnscaler.runtime.adapter.move_object'
    )
    # rank1's segment takes the dict `data` as an input (read from its local dataloader)
    assert len(_gencode_contains(
        tmp_path, SharedInputSegmentWithIntermediateModule, 1, r"data_.* = next\(\*\(dataloader_.*, \)\)\s*sum_.* = nnscaler.runtime.executor.fexecute\('segment.*', model.segment.*, \*\(add_.*, data_.*, \), requires_grad=True\)"
    )) == 2

    # code in rank 0:
    # def segment25(self, data_27):
    # # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 685, in forward,  x0 = data['data'] + self.w0
    # getitem_1_30 = _operator.getitem(data_27, 'data')
    # # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 685, in forward,  x0 = data['data'] + self.w0
    # add_32 = torch.add(getitem_1_30, self.w0_31, alpha=1)
    # del getitem_1_30
    # # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 686, in forward,  x1 = x0 + self.w1
    # add_1_34 = torch.add(add_32, self.w1_33, alpha=1)
    # del add_32
    # return add_1_34

    # code in rank 1:
    # def segment29(self, add_1_34, data_27):
    #     add_1_55 = nnscaler.runtime.function.identity(add_1_34)
    #     del add_1_34
    #     # fn: clone dataloader index op in consumer stage
    #     getitem_52 = _operator.getitem(data_27, 'loss')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 688, in forward,  x2 = x1 + self.w2
    #     add_2_36 = torch.add(add_1_55, self.w2_35, alpha=1)
    #     del add_1_55
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 689, in forward,  x3 = x2 + self.w3
    #     add_3_38 = torch.add(add_2_36, self.w3_37, alpha=1)
    #     del add_2_36
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 691, in forward,  loss = torch.sum(x3 + data_loss)  # data_loss is shared
    #     add_4_39 = torch.add(add_3_38, getitem_52, alpha=1)
    #     del getitem_52, add_3_38
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 691, in forward,  loss = torch.sum(x3 + data_loss)  # data_loss is shared
    #     sum_1_28 = torch.sum(add_4_39)
    #     del add_4_39
    #     return sum_1_28


class ComplexSharedInputSegmentWithIntermediateModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(4, 4))
        self.w1 = torch.nn.Parameter(torch.randn(4, 4))
        self.w2 = torch.nn.Parameter(torch.randn(4, 4))
        self.w3 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, data: dict[str, torch.Tensor]):
        data_input, data_result = data['net'], data['result']
        data_loss = data_result['loss']
        x0 = data_input['data'] + self.w0
        x1 = x0 + self.w1

        x2 = x1 + self.w2
        x3 = x2 + self.w3

        loss = torch.sum(x3 + data_loss)  # data_loss is shared
        return loss


@replace_all_device_with('cpu')
def test_complex_shared_dict_input_with_intermediate(tmp_path):
    m = ComplexSharedInputSegmentWithIntermediateModule()
    m.train()
    parallelize(
        m,
        {'data': {'net': {'data': torch.randn(4, 4)}, 'result': {'loss': torch.randn(4, 4)}}},
        pas_policy=_shared_output_partition_policy,
        compute_config= ComputeConfig(
            2, 4,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_scheduler='1f1b'
            )
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # rank1's segment reads `data` from its local dataloader and does getitem('result')
    assert _gencode_contains(
        tmp_path, ComplexSharedInputSegmentWithIntermediateModule, 1, r"_operator.getitem\(data_\d+, 'result'\)"
    )
    # rank0 no longer computes data['result'] (it was recomputed in the consumer stage)
    assert not _gencode_contains(
        tmp_path, ComplexSharedInputSegmentWithIntermediateModule, 0, r"_operator.getitem\(data_\d+, 'result'\)"
    )
    # the dict input is NOT transmitted between stages via any adapter
    assert not _gencode_contains(
        tmp_path, ComplexSharedInputSegmentWithIntermediateModule, 0, r'nnscaler.runtime.adapter.move_object'
    )
    assert not _gencode_contains(
        tmp_path, ComplexSharedInputSegmentWithIntermediateModule, 1, r'nnscaler.runtime.adapter.move_object'
    )
    # rank1's segment takes the dict `data` as an input (read from its local dataloader)
    assert len(_gencode_contains(
        tmp_path, ComplexSharedInputSegmentWithIntermediateModule, 1, r"data_.* = next\(\*\(dataloader_.*, \)\)\s*sum_.* = nnscaler.runtime.executor.fexecute\('segment.*', model.segment.*, \*\(add_.*, data_.*, \), requires_grad=True\)"
    )) == 2

    # code in rank 0:
    # def segment28(self, data_29):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 778, in forward,  data_input, data_result = data['net'], data['result']
    #     getitem_30 = _operator.getitem(data_29, 'net')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 780, in forward,  x0 = data_input['data'] + self.w0
    #     getitem_3_34 = _operator.getitem(getitem_30, 'data')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 780, in forward,  x0 = data_input['data'] + self.w0
    #     add_36 = torch.add(getitem_3_34, self.w0_35, alpha=1)
    #     del getitem_3_34
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 781, in forward,  x1 = x0 + self.w1
    #     add_1_38 = torch.add(add_36, self.w1_37, alpha=1)
    #     del add_36
    #     return add_1_38

    # code in rank 1:
    # def segment32(self, add_1_38, data_29):
    #     add_1_60 = nnscaler.runtime.function.identity(add_1_38)
    #     del add_1_38
    #     # fn: clone dataloader index op in consumer stage
    #     getitem_1_55 = _operator.getitem(data_29, 'result')
    #     # fn: clone dataloader index op in consumer stage
    #     getitem_2_57 = _operator.getitem(getitem_1_55, 'loss')
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 783, in forward,  x2 = x1 + self.w2
    #     add_2_40 = torch.add(add_1_60, self.w2_39, alpha=1)
    #     del add_1_60
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 784, in forward,  x3 = x2 + self.w3
    #     add_3_42 = torch.add(add_2_40, self.w3_41, alpha=1)
    #     del add_2_40
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 786, in forward,  loss = torch.sum(x3 + data_loss)  # data_loss is shared
    #     add_4_43 = torch.add(add_3_42, getitem_2_57, alpha=1)
    #     del getitem_2_57, add_3_42
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_pipeline.py", line 786, in forward,  loss = torch.sum(x3 + data_loss)  # data_loss is shared
    #     sum_1_32 = torch.sum(add_4_43)
    #     del add_4_43
    #     return sum_1_32
