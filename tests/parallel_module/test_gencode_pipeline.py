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
    assert _gencode_contains(
        tmp_path, PPModule1, 0,
        r'.*reserve_send_bundle\(\(\(0, 1\),\)\)',
    )
    assert _gencode_contains(
        tmp_path, PPModule1, 1,
        r'.*reserve_send_bundle\(\(\(1, 0\),\)\)',
    )
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


@replace_all_device_with('cpu')
def test_gencode_shared_irobject_between_tp_stages(tmp_path):
    """IRObjects are replicated, not spatially split, between TP stages."""
    m = PPModule2()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=2),
        compute_config=ComputeConfig(
            4, 4,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=4,
                pipeline_size=2,
                pipeline_scheduler='1f1b',
            ),
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    for src, dst in ((0, 2), (1, 3)):
        assert _gencode_contains(
            tmp_path,
            PPModule2,
            src,
            rf'nnscaler.runtime.adapter.move_object\(.*, src={src}, dst={dst}\)',
        )
        assert _gencode_contains(
            tmp_path,
            PPModule2,
            dst,
            rf'nnscaler.runtime.adapter.move_object\(.*, src={src}, dst={dst}\)',
        )


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

    def forward(self, data: torch.Tensor):
        x = self.layers[0](data)
        x = self.layers[1](x) + x
        x = self.layers[2](x)
        x = self.layers[3](x)
        loss = torch.sum(x)
        return loss


def split_segment_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_pas_ops

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


class AliasedSegmentInputModule(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])

    def forward(self, data: torch.Tensor):
        shared = self.layers[0](data)
        hidden = self.layers[1](shared)
        hidden = self.layers[2](hidden + shared)
        return torch.sum(hidden)


def aliased_segment_input_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_pas_ops

    stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            stage_id = get_layer_index(node.fqn)
        elif node.fn == torch.add:
            stage_id = 2
        yield OpPlan(node, stage_id=stage_id, partition=OpPartition(input=0, dim=0))


@replace_all_device_with('cpu')
def test_gencode_narrows_aliased_segment_input(tmp_path):
    parallelize(
        AliasedSegmentInputModule(),
        {'data': torch.randn(64, 64)},
        pas_policy=aliased_segment_input_pas,
        compute_config=ComputeConfig(
            6,
            6,
            constant_folding=False,
            use_end2end=True,
            pas_config={
                'pp_size': 3,
                'pipeline_nmicros': 3,
                'pipeline_scheduler': '1f1b',
            },
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    for rank in range(6):
        assert not _gencode_contains(
            tmp_path,
            AliasedSegmentInputModule,
            rank,
            r'nnscaler\.runtime\.adapter\.(?:all_gather|nn\.split_allgather)',
        )
