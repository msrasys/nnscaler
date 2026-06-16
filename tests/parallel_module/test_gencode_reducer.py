import torch
import torch.nn.functional as F


import nnscaler
from nnscaler.parallel import parallelize, ComputeConfig
from tests.utils import replace_all_device_with

from .test_gencode import _gencode_contains, print_gencode


class NormalLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        # weight = self.weight / self.weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return F.linear(x, self.weight)


def normal_linear_split_policy(graph, cfg):
    from nnscaler.policies import get_pas_ops, OpPlan, OpPartition

    for node in get_pas_ops(graph):
        if node.name == 'linear':
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))


@replace_all_device_with('cpu')
def test_codegen_normal_split(tmp_path):
    m = NormalLinearModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 4)},
        normal_linear_split_policy,
        ComputeConfig(2, 4),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    assert _gencode_contains(tmp_path, NormalLinearModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=1\)"
    )
    # generated reducer:
    # self.wreducer32 = nnscaler.runtime.adapter.Reducer(ranks=[0, 1, 2, 3], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=1)


class NormLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        weight = self.weight / self.weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return F.linear(x, weight)


@replace_all_device_with('cpu')
def test_codegen_norm_linear_split(tmp_path):
    m = NormLinearModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 4)},
        normal_linear_split_policy,
        ComputeConfig(2, 4, reducer_replicated_params=False),
        instance_name = 'noreplicated_weights',
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    # cross replica reducer
    assert _gencode_contains(tmp_path, NormLinearModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'noreplicated_weights',
    )
    # will have identity allreduce for replicated weights
    assert _gencode_contains(tmp_path, NormLinearModule, 0,
        r"nnscaler.runtime.adapter.nn.identity_allreduce",
        instance_name = 'noreplicated_weights',
    )
    # generated reducer:
    # self.wreducer80 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=1)

    m = NormLinearModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 4)},
        normal_linear_split_policy,
        ComputeConfig(2, 4, reducer_replicated_params=True),
        instance_name = 'replicated_weights',
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    # all rank reducer, with nreplicas=2
    assert _gencode_contains(tmp_path, NormLinearModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=2\)",
        instance_name = 'replicated_weights',
    )
    # will have identity allreduce for replicated weights
    assert _gencode_contains(tmp_path, NormLinearModule, 0,
        r"nnscaler.runtime.adapter.nn.identity_allreduce",
        instance_name = 'replicated_weights',
    )
    # generated reducer:
    # self.wreducer64 = nnscaler.runtime.adapter.Reducer(ranks=[0, 1, 2, 3], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=2)


class MixedLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(8, 4))
        self.weight2 = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        weight = self.weight1 / self.weight1.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return F.linear(x, weight) + F.linear(x, self.weight2)


@replace_all_device_with('cpu')
def test_codegen_mixed_linear_split(tmp_path):
    m = MixedLinearModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 4)},
        normal_linear_split_policy,
        ComputeConfig(2, 4, reducer_replicated_params=False),
        instance_name = 'noreplicated_weights',
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    # cross replica reducer
    assert _gencode_contains(tmp_path, MixedLinearModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'noreplicated_weights',
    )
    # will have identity allreduce for replicated weights
    assert _gencode_contains(tmp_path, MixedLinearModule, 0,
        r"nnscaler.runtime.adapter.nn.identity_allreduce",
        instance_name = 'noreplicated_weights',
    )
    # generated reducer:
    # self.wreducer80 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=1)

    m = MixedLinearModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 4)},
        normal_linear_split_policy,
        ComputeConfig(2, 4, reducer_replicated_params=True),
        instance_name = 'replicated_weights',
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    # all rank reducer, with nreplicas=2
    assert _gencode_contains(tmp_path, MixedLinearModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=2\)",
        instance_name = 'replicated_weights',
    )
    # will have identity allreduce for replicated weights
    assert _gencode_contains(tmp_path, MixedLinearModule, 0,
        r"nnscaler.runtime.adapter.nn.identity_allreduce",
        instance_name = 'replicated_weights',
    )
    # generated reducer:
    # self.wreducer64 = nnscaler.runtime.adapter.Reducer(ranks=[0, 1, 2, 3], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=2)


class NormalAddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return self.weight + x


def normal_add_split_policy(graph, cfg):
    from nnscaler.policies import get_pas_ops, OpPlan, OpPartition

    for node in get_pas_ops(graph):
        if node.fn == torch.add:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))


@replace_all_device_with('cpu')
def test_codegen_normal_add_split(tmp_path):
    m = NormalAddModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4)},
        normal_add_split_policy,
        ComputeConfig(2, 4),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    # weight is partitioned, so the reducer should have rank=[0, 2] and nreplicas=1
    assert _gencode_contains(tmp_path, NormalAddModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)"
    )
    # generated reducer:
    # self.wreducer44 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=1)


class NormalAddMulMultirefModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        y = self.weight + x
        z = self.weight * x
        return y - z


@replace_all_device_with('cpu')
def test_codegen_normal_add_mul_multiref_split(tmp_path):
    m = NormalAddMulMultirefModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4)},
        normal_add_split_policy,
        ComputeConfig(2, 4),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'noreplicated_weights',
    )

    # weight is partitioned, so the reducer should have rank=[0, 2] and nreplicas=1
    assert _gencode_contains(tmp_path, NormalAddMulMultirefModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'noreplicated_weights',
    )
    assert _gencode_contains(tmp_path, NormalAddMulMultirefModule, 0,
        r"nnscaler.runtime.function.multiref\(self.weight_.*, times=2\)",
        instance_name = 'noreplicated_weights',
    )
    # generated reducer:
    # self.wreducer110 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=1)

    m = NormalAddMulMultirefModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(4, 4)},
        normal_add_split_policy,
        ComputeConfig(2, 4, reducer_replicated_params=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'replicated_weights',
    )

    assert _gencode_contains(tmp_path, NormalAddMulMultirefModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=2\)",
        instance_name = 'replicated_weights',
    )
    assert _gencode_contains(tmp_path, NormalAddMulMultirefModule, 0,
        r"nnscaler.runtime.function.multiref\(self.weight_.*, times=2\)",
        instance_name = 'replicated_weights',
    )
    # generated reducer:
    # self.wreducer94 = nnscaler.runtime.adapter.Reducer(ranks=[0, 1, 2, 3], reduce_op='sum', async_op=async_op, zero=0, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_param_level_sharding=zero_param_level_sharding,zero_ngroups=1,nreplicas=2)


from tests.graph.gener.test_reducer_gen import (
    ReducerModule,
    SimpleModule,
    SimpleModuleNoReduce,
    SimpleModule2ConsumersTp,
    SimpleModule2ConsumersSP,
    SimpleModule2ConsumersSP3
)


@replace_all_device_with('cpu')
def test_pp_replicated_tp1_shared_weight(tmp_path):
    m = ReducerModule()
    m.train()

    def _pas(graph, cfg):
        from nnscaler.policies import get_pas_ops, OpPlan, OpPartition, IRFwOperation
        [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)
        for node in get_pas_ops(graph):
            if node in [matmul1, matmul2]:
                yield OpPlan(node, stage_id=0)
            elif node in [add, sum]:
                yield OpPlan(node, stage_id=1)

    parallelize(
        m,
        {'x': torch.randn(128, 128)},
        _pas,
        ComputeConfig(2, 4, use_end2end=True, pas_config={
            'pipeline_nmicros': 2
        }),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'multiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 0,
        r"nnscaler.runtime.function.multiref\(self.param1_.*, times=2\)",
        instance_name = 'multiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'multiref_weight',
    )

    m = ReducerModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 128)},
        _pas,
        ComputeConfig(2, 4, use_end2end=True, pas_config={
            'pipeline_nmicros': 2,
            'pipeline_multiref_replicated_params': False,
        }),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'nomultiref_weight',
    )
    # param1 will be on both stages,
    # the reducer for param1 will be [0, 1, 2, 3] with nreplicas=1, and there will be no multiref
    assert not _gencode_contains(tmp_path, ReducerModule, 0,
        r"nnscaler.runtime.function.multiref\(self.param1_.*, times=2\)",
        instance_name = 'nomultiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'nomultiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 0,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=1\)",
        instance_name = 'nomultiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 0,
        r"self.register_parameter\('param1_.*",
        instance_name = 'nomultiref_weight',
    )

    assert not _gencode_contains(tmp_path, ReducerModule, 1,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 2\].*nreplicas=1\)",
        instance_name = 'nomultiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 1,
        r"nnscaler.runtime.adapter.Reducer\(ranks=\[0, 1, 2, 3\].*nreplicas=1\)",
        instance_name = 'nomultiref_weight',
    )
    assert _gencode_contains(tmp_path, ReducerModule, 1,
        r"self.register_parameter\('param1_.*",
        instance_name = 'nomultiref_weight',
    )


@replace_all_device_with('cpu')
def test_tp_partitioned_weight(tmp_path):
    def _pas(graph, cfg):
        from nnscaler.policies import get_pas_ops, OpPlan, OpPartition, IRFwOperation
        [matmul1, matmul2, sum] = graph.select(ntype=IRFwOperation)
        for node in get_pas_ops(graph):
            if node in [matmul1]:
                yield OpPlan(node, partition=OpPartition(input=0, dim=1))

    m = SimpleModule()
    m.train()

    parallelize(
        m,
        {'x': torch.randn(128, 128)},
        _pas,
        ComputeConfig(2, 4, use_end2end=True, reducer_replicated_params=False),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'noreducer_replicated_params',
    )

    m = SimpleModule()
    m.train()

    parallelize(
        m,
        {'x': torch.randn(128, 128)},
        _pas,
        ComputeConfig(2, 4, use_end2end=True, reducer_replicated_params=True),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
        instance_name = 'reducer_replicated_params',
    )
    assert True
