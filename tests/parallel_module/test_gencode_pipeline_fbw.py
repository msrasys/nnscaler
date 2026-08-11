import torch
import pytest

from nnscaler import ComputeConfig, parallelize
from tests.parallel_module.test_gencode import _gencode_contains, replace_all_device_with
from tests.parallel_module.test_gencode_pipeline import PPModule1, pp_pas


@replace_all_device_with('cpu')
@pytest.mark.parametrize(
    'pipeline_scheduler',
    [
        '1f1b',
        '1f1b_interleaved_fbw',
        '1f1b_interleaved_zero_bubble_steady',
        '1f1b_interleaved_zero_bubble',
    ],
)
def test_pipeline_codegen_emits_split_backward(tmp_path, pipeline_scheduler):
    model = PPModule1(dim=64, nlayers=4)
    model.train()
    parallelize(
        model,
        {'data': torch.randn(8, 64)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=2),
        compute_config=ComputeConfig(
            2,
            2,
            constant_folding=False,
            use_end2end=True,
            use_fbw=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_nstages': 2,
                'pipeline_scheduler': pipeline_scheduler,
            },
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    for rank in range(2):
        input_calls = _gencode_contains(
            tmp_path,
            PPModule1,
            rank,
            r'nnscaler\.runtime\.executor\.backward_input\(',
        )
        weight_calls = _gencode_contains(
            tmp_path,
            PPModule1,
            rank,
            r'nnscaler\.runtime\.executor\.backward_weight\(',
        )
        assert input_calls
        assert len(input_calls) == len(weight_calls)
        assert not _gencode_contains(
            tmp_path,
            PPModule1,
            rank,
            r'nnscaler\.runtime\.executor\.backward\(',
        )

        split_actions = _gencode_contains(
            tmp_path,
            PPModule1,
            rank,
            r'nnscaler\.runtime\.executor\.(backward_input|backward_weight)\(',
        )
        if pipeline_scheduler == '1f1b_interleaved_zero_bubble' \
                and rank == 1:
            # Rank 1 delays W by one I, including across cooldown. This checks
            # that the generated program is not merely adjacent I/W pairs.
            assert split_actions[:2] == [
                'backward_input',
                'backward_input',
            ]


@replace_all_device_with('cpu')
def test_zero_bubble_pending_weight_cap_changes_local_order(tmp_path):
    model = PPModule1(dim=64, nlayers=4)
    model.train()
    parallelize(
        model,
        {'data': torch.randn(8, 64)},
        pas_policy=lambda graph, cfg: pp_pas(graph, cfg, nlayers_per_stage=2),
        compute_config=ComputeConfig(
            2,
            2,
            constant_folding=False,
            use_end2end=True,
            use_fbw=True,
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_nstages': 2,
                'pipeline_scheduler': '1f1b_interleaved_zero_bubble',
                'zero_bubble_max_pending_weight_backwards': 1,
            },
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    split_actions = _gencode_contains(
        tmp_path,
        PPModule1,
        1,
        r'nnscaler\.runtime\.executor\.(backward_input|backward_weight)\(',
    )
    assert split_actions[:2] == ['backward_input', 'backward_weight']


@replace_all_device_with('cpu')
def test_explicit_fbw_schedule_requires_use_fbw(tmp_path):
    model = PPModule1(dim=64, nlayers=4)
    model.train()

    with pytest.raises(RuntimeError, match='Code generation failed') as exc_info:
        parallelize(
            model,
            {'data': torch.randn(8, 64)},
            pas_policy=lambda graph, cfg: pp_pas(
                graph, cfg, nlayers_per_stage=2
            ),
            compute_config=ComputeConfig(
                2,
                2,
                constant_folding=False,
                use_end2end=True,
                use_fbw=False,
                pas_config={
                    'pipeline_nmicros': 2,
                    'pipeline_nstages': 2,
                    'pipeline_scheduler': (
                        '1f1b_interleaved_zero_bubble'
                    ),
                },
            ),
            gen_savedir=tmp_path,
            load_module=False,
            reuse='override',
        )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert 'use_fbw=True' in str(exc_info.value.__cause__)
