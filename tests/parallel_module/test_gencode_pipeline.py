from re import RegexFlag

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
        flags=RegexFlag.DOTALL
    )
    assert _gencode_contains(
        tmp_path, PassThroughSegmentModule, 2,
        r'def segment.*'
        r'nnscaler.runtime.function.identity.*'
        r'nnscaler.runtime.function.identity.*'
        r'torch.nn.functional.linear.*'
        r'return linear_.*, linear_.*'
        r'def _train_step.*',
        flags=RegexFlag.DOTALL
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
        flags=RegexFlag.DOTALL
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
