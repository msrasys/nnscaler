import torch

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
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        if self.return_type == 0:
            return loss
        elif self.return_type == 1:
            return loss, y # the second return is not tensor
        elif self.return_type == 2:
            return loss, data.shape
        elif self.return_type == 3:
            return loss, {'data': data}
        elif self.return_type == 4:
            return loss, z
        else:
            raise ValueError(f"Unsupported return_type: {self.return_type}")


def pp_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops

    found_layer = False
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            partition = None
            # if layer_idx == 1 or layer_idx == 2:
            #     partition = OpPartition(0, 0)
            yield OpPlan(node, stage_id=layer_idx // 2, partition=partition)
            found_layer = True
        else:
            yield OpPlan(node, stage_id=1 if found_layer else 0)


@replace_all_device_with('cpu')
def test_gencode_correct_dataloader_order(tmp_path):
    m = PPModule1(return_type=3)
    m.train()
    parallelize(
        m,
        {'data': torch.randn(64, 1024)},
        pas_policy=pp_pas,
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