#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
import pytest
import torch
from nnscaler import parallelize, ComputeConfig
from nnscaler.parallel import merge_state_dicts
from tests.launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from tests.utils import clear_dir_on_rank0, init_random, PYTEST_RUN_ID
from tests.parallel_module.common import init_distributed
from tests.parallel_module.test_gencode_pipeline import SplitSegmentModule, split_segment_pas


def _narrowed_boundary_worker(async_comm):
    from nnscaler.flags import CompileFlag
    CompileFlag.async_comm = async_comm
    torch.set_float32_matmul_precision("highest")
    dim = 16
    nmicros = 4
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'narrowed_boundary_{PYTEST_RUN_ID}') as tempdir:
        init_random()
        model = parallelize(
            SplitSegmentModule(dim),
            {'data': torch.randn(8, dim)},
            pas_policy=split_segment_pas,
            compute_config=ComputeConfig(
                4,
                4,
                constant_folding=False,
                use_end2end=True,
                pas_config={
                    'pipeline_nmicros': nmicros,
                    'pipeline_scheduler': '1f1b',
                },
            ),
            gen_savedir=tempdir,
            reuse='override',
        )
        model.cuda()
        init_random()
        samples = [torch.randn(8, dim) for _ in range(nmicros)]
        losses = model.train_step(samples)
        model.eval()
        inferred = model.infer_step(samples)
        for actual, expected in zip(inferred, losses):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        from nnscaler.runtime.executor import Executor, AsyncCommHandler
        Executor.check_clear()
        AsyncCommHandler().check_clear()
        grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        model._add_extra_state(grads, '')
        return clone_to_cpu_recursively(grads)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize("async_comm", [False, True])
def test_narrowed_pipeline_boundary(async_comm):
    torch.set_float32_matmul_precision("highest")
    dim = 16
    nmicros = 4
    results = launch_torchrun(4, _narrowed_boundary_worker, async_comm)
    merged_grads, _ = merge_state_dicts([results[rank] for rank in range(4)])

    torch.cuda.set_device(0)
    with torch.device('cuda:0'):
        init_random()
        reference = SplitSegmentModule(dim)
        init_random()
        losses = [reference(torch.randn(8, dim)) for _ in range(nmicros)]
        torch.stack(losses).sum().backward()

    reference_grads = {
        name: param.grad.cpu()
        for name, param in reference.named_parameters()
        if param.grad is not None
    }
    assert merged_grads.keys() == reference_grads.keys()
    for name, grad in reference_grads.items():
        torch.testing.assert_close(merged_grads[name], grad, atol=1e-5, rtol=1e-5)
