from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer

from tests.cli.common import MLP
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close, assert_equal


def trainer_worker_pipeline(save_dir, config_file, run_name=None, additional_args=None):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = run_name or config_path.stem
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{run_name}'

    additional_args = additional_args or []

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        *additional_args
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()


def trainer_worker_pipeline_multiple_stream(save_dir, config_file, run_name=None, additional_args=None):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = run_name or f'{config_path.stem}_multi_stream'
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{run_name}'

    additional_args = additional_args or []

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.pas_config.pipeline_scheduler', 'tests.test_policies.sched_1f1b_multi_stream',
        '--checkpoint.save_dir', str(ckpt_savedir),
        *additional_args
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline(tmp_path):
    launch_torchrun(4, trainer_worker_pipeline_multiple_stream, tmp_path, 'trainer_args_pipeline.yaml')
    launch_torchrun(4, trainer_worker_pipeline_multiple_stream, tmp_path, 'trainer_args_pipeline.yaml',
        'fbw',
        ['--compute_config.use_fbw', 'True',]
    )
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline_autodist.yaml')
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml')
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml',
        'async_comm',
        [
            '--compute_config.use_async_comm', 'True',
            '--compute_config.use_fbw', 'True'
        ]
    )

    merged_files = list((tmp_path).glob('merged_*.pt'))
    assert len(merged_files) == 5
    merged_state_dicts = [torch.load(merged_file, weights_only=False) for merged_file in merged_files]

    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[1]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[1]['optimizer'])
    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[2]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[2]['optimizer'])
    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[3]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[3]['optimizer'])
    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[4]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[4]['optimizer'])

    # when compute_config.use_async_comm is True, and compute_config.use_fbw is True
    # the code will look like:
    #....
    # def adapter95(self):
    #     glinear_8_131 = nnscaler.runtime.adapter.move((), shape=(2, 16), dtype=torch.float32, src=1, dst=0, async_op=True)
    #     return glinear_8_131
    # ...
    # def _train_step(model, dataloader_112):
    #     ...
    #     _ = nnscaler.runtime.executor.backward_input('segment43', (), (linear_8_184, ), (glinear_8_185, ), model.parameters())
    #     del linear_8_184, glinear_8_185
    #     _ = nnscaler.runtime.executor.backward_input('segment43', (), (linear_8_184, ), (glinear_8_185, ), model.parameters())
    #     del linear_8_184, glinear_8_185
    #     glinear_8_208 = nnscaler.runtime.executor.aexecute(model.adapter95, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     nnscaler.runtime.executor.backward_weight('segment43', model.parameters())
    #     _ = nnscaler.runtime.executor.backward_input('segment43', (), (linear_8_207, ), (glinear_8_208, ), model.parameters())
    #     del linear_8_207, glinear_8_208
    #     binary_cross_entropy_76 = nnscaler.runtime.executor.aexecute(model.adapter102, *(), requires_grad=True)
    #     binary_cross_entropy_176 = nnscaler.runtime.executor.aexecute(model.adapter102, *(), requires_grad=True)
    #     binary_cross_entropy_199 = nnscaler.runtime.executor.aexecute(model.adapter102, *(), requires_grad=True)
    #     binary_cross_entropy_222 = nnscaler.runtime.executor.aexecute(model.adapter102, *(), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     nnscaler.runtime.executor.backward_weight('segment43', model.parameters())
    #     _ = nnscaler.runtime.executor.aexecute(model.reducer293, *(), requires_grad=False)
    #     nnscaler.runtime.executor.AsyncCommHandler().drain_sends()
    #     (binary_cross_entropy_76, binary_cross_entropy_176, binary_cross_entropy_199, binary_cross_entropy_222, ) = nnscaler.runtime.executor.sync_tensors((binary_cross_entropy_76, binary_cross_entropy_176, binary_cross_entropy_199, binary_cross_entropy_222, ))
    #     return binary_cross_entropy_76, binary_cross_entropy_176, binary_cross_entropy_199, binary_cross_entropy_222
    #     ...


def pp_obj_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops

    last_stage_id = 0
    for node in get_pas_ops(graph):
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            yield OpPlan(node, stage_id=layer_idx // 2)
            last_stage_id = layer_idx // 2
        else:
            yield OpPlan(node, stage_id=last_stage_id)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline_obj(tmp_path):
    launch_torchrun(4, trainer_worker_pipeline, tmp_path,
        'trainer_args_pipeline_obj.yaml',
        'no_constant_folding',
        ['--compute_config.constant_folding', False]
    )
    launch_torchrun(4, trainer_worker_pipeline, tmp_path,
        'trainer_args_pipeline_obj.yaml',
        'constant_folding',
    )
    merged_files = list((tmp_path).glob('merged_*.pt'))
    assert len(merged_files) == 2
    state_dict0 = torch.load(merged_files[0], weights_only=False)
    state_dict1 = torch.load(merged_files[1], weights_only=False)

    assert_equal(state_dict0['model'], state_dict1['model'])
    assert_equal(state_dict0['optimizer'], state_dict1['optimizer'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline_async(tmp_path):
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml',
        'async_comm_async_reducer',
        [
            '--compute_config.use_async_comm', 'True',
            '--compute_config.use_async_reducer', 'True'
        ]
    )
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml',
        'async_comm',
        ['--compute_config.use_async_comm', 'True']
    )
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml',
        'no_async_comm',
        ['--compute_config.use_async_comm', 'False']
    )

    merged_files = list((tmp_path).glob('merged_*.pt'))
    assert len(merged_files) == 3
    merged_state_dicts = [torch.load(merged_file, weights_only=False) for merged_file in merged_files]

    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[1]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[1]['optimizer'])
    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[2]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[2]['optimizer'])


class ColocatedSharedMLP(MLP):
    def forward(self, data):
        x = data['data']
        for layer in (0, 0, 1, 0, 1):
            x = self.layers[layer](x)
        return self.loss_fn(torch.sigmoid(x), data['target'])


def colocated_shared_pipeline_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops

    # With 4 ranks and pipeline_size=2, each physical pipeline group has 2 ranks:
    # segment 0: layer 0 twice, on ranks (0, 1)
    # segment 1: layer 1 once,  on ranks (2, 3)
    # segment 2: layer 0 once,  on ranks (0, 1)
    # segment 3: layer 1 once,  on ranks (2, 3), followed by the loss.
    # Each weight produces 2 gradient hooks per microbatch, not its total use count.
    linear_stages = iter((0, 0, 1, 2, 3))
    stage = 0
    for node in get_pas_ops(graph):
        if node.fn == torch.nn.functional.linear:
            stage = next(linear_stages)
            partition = OpPartition(0, 0)
        else:
            partition = 'auto'
        yield OpPlan(node, stage_id=stage, partition=partition)


def trainer_worker_colocated_shared_pipeline(save_dir, run_name, async_reducer, use_fbw):
    save_dir = Path(save_dir)
    ckpt_savedir = save_dir / run_name / 'ckpt'
    trainer = Trainer([
        '-f', Path(__file__).with_name('trainer_args_pipeline.yaml').resolve(),
        '--model.type', 'tests.cli.test_trainer_pipeline.ColocatedSharedMLP',
        '--model.args.nlayers', '2',
        '--pas_policy', 'tests.cli.test_trainer_pipeline.colocated_shared_pipeline_pas',
        '--compute_config.plan_ngpus', '4',
        '--compute_config.pas_config.pipeline_size', '2',
        '--compute_config.pas_config.pipeline_scheduler', '1f1b_interleaved',
        '--compute_config.use_async_reducer', str(async_reducer),
        '--compute_config.use_fbw', str(use_fbw),
        '--global_batch_size', '8',
        '--max_train_steps', '3',
        '--enable_progress_bar', 'False',
        '--instance_name', f'instance_{run_name}',
        '--gen_savedir', str(save_dir / run_name / 'gen'),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    assert trainer.train_status.finished_train_steps == 3
    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4
    expected_name = f'layers.{trainer.rank // 2}.weight'
    assert {meta.orig_name for meta in trainer.model.fullmap.values()} == {expected_name}
    assert len(trainer.model.reducers) == 1
    reducer = trainer.model.reducers[0]
    assert set(reducer._param_num_segments.values()) == {2}
    assert reducer.buckets
    for bucket in reducer.buckets:
        assert bucket._async == async_reducer
        assert bucket._num_grads_per_microbatch == 2

    if trainer.rank == 0:
        checkpoints = list((ckpt_savedir / 'last').glob('*.ckpt'))
        assert len(checkpoints) == 4
        Trainer.merge_checkpoint(checkpoints, save_dir / f'merged_{run_name}.pt')
    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline_colocated_shared_params_async(tmp_path):
    runs = [('sync', False, False), ('async', True, False), ('async_fbw', True, True)]
    for run_name, async_reducer, use_fbw in runs:
        launch_torchrun(
            4, trainer_worker_colocated_shared_pipeline,
            tmp_path, run_name, async_reducer, use_fbw,
        )

    baseline = torch.load(tmp_path / 'merged_sync.pt', weights_only=False)
    for run_name in ('async', 'async_fbw'):
        actual = torch.load(tmp_path / f'merged_{run_name}.pt', weights_only=False)
        assert_close(baseline['model'], actual['model'])
        assert_close(baseline['optimizer'], actual['optimizer'])


def trainer_worker_multi_sched(save_dir, config_file):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = config_path.stem
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{run_name}'

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_train_steps', '12',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    # step-dependent update_freq {0: 2, 5: 4} compiles one scheduler per nmicros
    assert tuple(sorted(trainer.model.nmicros_per_scheduler_step)) == (2, 4)

    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    # step-dependent update_freq {0: 2, 5: 4} compiles one scheduler per nmicros
    assert tuple(sorted(trainer.model.nmicros_per_scheduler_step)) == (2, 4)
    torch.distributed.barrier()

    ckpt1_savedir = save_dir / run_name / 'ckpt1'
    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
    ])
    trainer.run()

    if trainer.rank == 0:
        ckpt_files = list((ckpt_savedir / 'last').glob('*.ckpt'))
        ckpt1_files = list((ckpt1_savedir / 'last').glob('*.ckpt'))
        assert len(ckpt_files) == len(ckpt1_files) == 4
        ckpt_state_dicts = [torch.load(ckpt_file, weights_only=False) for ckpt_file in ckpt_files]
        ckpt1_state_dicts = [torch.load(ckpt1_file, weights_only=False) for ckpt1_file in ckpt1_files]
        for ckpt_state_dict, ckpt1_state_dict in zip(ckpt_state_dicts, ckpt1_state_dicts):
            assert_equal(ckpt_state_dict['model'], ckpt1_state_dict['model'])
            assert_equal(ckpt_state_dict['optimizer'], ckpt1_state_dict['optimizer'])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline_multi_sched(tmp_path):
    launch_torchrun(4, trainer_worker_multi_sched, tmp_path, 'trainer_args_pipeline_multi_sched.yaml')
