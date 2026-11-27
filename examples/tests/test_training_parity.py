"""Qwen/DeepSeek compilation and multi-step training parity.

Requires Transformers with Qwen3.5 and DeepSeek-V3.2 support (validated with
Transformers 5.20.2). CPU tests compare original HF, rewritten eager, and generated
training. GPU tests require two or four GPUs and compare against rewritten eager.
"""

import ast
import builtins
import copy
import importlib
import inspect
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch


CASES = [
    ('Deepseek', 'micro-v2', 'v2_lite_ep4_tp4', 2, 32),
    ('Deepseek', 'micro-v3', 'v3_expanded_ep4', 2, 32),
    ('Deepseek', 'micro-v3', 'v3_expanded_etp4', 2, 32),
    ('Deepseek', 'deepseek-v2-lite', 'v2_lite_ep4_tp4', 2, 64),
    ('qwen3p5', 'micro-moe', 'moe_ep4_tp1', 4, 32),
    ('qwen3p5', 'micro-moe', 'moe_expanded_ep2', 2, 32),
    ('qwen3p5', 'micro-vl', 'vl_tp2', 2, 32),
]


def _training_batch(template, step, vocab_size, image_token_id=None):
    generator = torch.Generator().manual_seed(20261127 + step)
    samples = {name: tensor.clone() for name, tensor in template.items()}
    tokens = torch.randint(min(vocab_size, 128), samples['input_ids'].shape, generator=generator)
    if image_token_id is not None:
        tokens[template['input_ids'] == image_token_id] = image_token_id
    samples['input_ids'] = tokens
    labels = torch.randint(vocab_size, samples['labels'].shape, generator=generator)
    labels[template['labels'] == -100] = -100
    labels[0, -(2 + step % 3):] = -100
    samples['labels'] = labels
    if 'valid_token_count' in samples:
        samples['valid_token_count'] = (labels[:, 1:] != -100).sum(dim=1, keepdim=True).float()
    if 'pixel_values' in samples:
        samples['pixel_values'] = torch.randn(samples['pixel_values'].shape, generator=generator)
    return samples


def _hf_reference_context(family):
    stack = ExitStack()
    if family == 'qwen3p5':
        for model_name in ('qwen3_5', 'qwen3_5_moe'):
            module = importlib.import_module(f'transformers.models.{model_name}.modeling_{model_name}')
            for name in ('causal_conv1d_fn', 'torch_chunk_gated_delta_rule'):
                stack.enter_context(patch.object(module, name, inspect.unwrap(getattr(module, name))))
    return stack


def _hf_parameter_permutations(reference, family):
    permutations = {}
    if family == 'qwen3p5':
        rewrites = importlib.import_module('examples.qwen3p5.rewrites.model_rewrites')
        for name, module in reference.named_modules():
            if isinstance(module, (rewrites.Qwen3_5GatedDeltaNet, rewrites.Qwen3_5MoeGatedDeltaNet)):
                permutation = rewrites._gdn_head_group_perm(
                    module.num_k_heads, module.head_k_dim, module.num_v_heads, module.head_v_dim,
                    module.in_proj_qkv.weight.device,
                )
                for suffix in ('in_proj_qkv.weight', 'conv1d.weight', 'conv1d.bias'):
                    permutations[f'{name}.{suffix}'] = permutation
    return permutations


def assert_train_step_interfaces(path):
    tree = ast.parse(path.read_text())
    module_names = set(dir(builtins))
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            module_names.update(alias.asname or alias.name.split('.')[0] for alias in statement.names)
        elif isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(statement.name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            module_names.update(node.id for node in ast.walk(statement)
                                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    step = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_train_step')
    local_names = {argument.arg for argument in step.args.args}
    local_names.update(node.id for node in ast.walk(step)
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    referenced = {node.id for node in ast.walk(step)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    assert not referenced - local_names - module_names, (path, referenced - local_names - module_names)
    backward_calls = [node for node in ast.walk(step) if isinstance(node, ast.Call)
                      and ast.unparse(node.func).endswith('executor.backward')]
    assert backward_calls, path
    for call in backward_calls:
        assert len(call.args[2].elts) == len(call.args[3].elts), (path, ast.unparse(call))


@pytest.mark.parametrize('family,preset,strategy_name,batch_size,seq_len', CASES)
@pytest.mark.parametrize('single_rank', [True, False])
def test_nonpipeline_train_step(tmp_path, family, preset, strategy_name, batch_size, seq_len, single_rank):
    import nnscaler

    if family == 'Deepseek':
        from examples.Deepseek.compile import compile_deepseek as compile_model
        from examples.Deepseek.policy import load_strategy_config
        kwargs = {}
    else:
        from examples.qwen3p5.compile import compile_qwen3_5 as compile_model
        from examples.qwen3p5.policy.json_pas import load_strategy_config
        kwargs = {'multimodal': preset == 'micro-vl'}
    repo_root = Path(nnscaler.__file__).resolve().parent.parent
    strategy = load_strategy_config(repo_root / 'examples' / family / 'strategies' / f'{strategy_name}.json')
    if single_rank:
        strategy.mesh.plan_ngpus = 1
        strategy.mesh.runtime_ngpus = 1
    manifest = compile_model(preset, strategy, tmp_path, batch_size=batch_size, seq_len=seq_len,
                             weightless=True, **kwargs)
    assert manifest['codegen_validation']['all_ranks_present']
    generated_files = [path for path in tmp_path.rglob('gencode*.py') if '_parallel_modules' in path.parts]
    assert len(generated_files) == strategy.mesh.plan_ngpus
    for generated_file in generated_files:
        assert_train_step_interfaces(generated_file)


@pytest.mark.parametrize('family,preset,strategy_name,batch_size,seq_len',
                         [case for case in CASES if case[1].startswith('micro')])
@pytest.mark.parametrize('optimizer_name', ['sgd', 'adamw', 'adamw_eps1e6'])
def test_micro_generated_training(tmp_path, monkeypatch, family, preset, strategy_name, batch_size, seq_len, optimizer_name):
    import nnscaler
    from nnscaler.parallel import _load_parallel_module_class

    driver = importlib.import_module(f'examples.{family}.compile')
    compile_model = driver.compile_deepseek if family == 'Deepseek' else driver.compile_qwen3_5
    repo_root = Path(nnscaler.__file__).resolve().parent.parent
    strategy = driver.load_strategy_config(repo_root / 'examples' / family / 'strategies' / f'{strategy_name}.json')
    strategy.mesh.plan_ngpus = strategy.mesh.runtime_ngpus = 1
    instance_name = f'{family}_{preset}_{strategy_name}_{optimizer_name}'.replace('-', '_')
    captured = {}
    original_parallelize = driver.parallelize
    torch.manual_seed(20261127)
    builder = importlib.import_module(
        'examples.Deepseek.configs.model_configs' if family == 'Deepseek'
        else 'examples.qwen3p5.rewrites.model_rewrites'
    )
    rewrite_name = 'apply_deepseek_rewrites' if family == 'Deepseek' else 'apply_qwen3_5_rewrites'
    original_rewrite = getattr(builder, rewrite_name)

    def capture_hf_model(model, *args, **kwargs):
        captured['hf_model'] = copy.deepcopy(model)
        return original_rewrite(model, *args, **kwargs)

    monkeypatch.setattr(builder, rewrite_name, capture_hf_model)

    def capture_parallelize(model, dummy_args, **kwargs):
        captured.update(model=model, samples=dummy_args['samples'], gen_savedir=kwargs['gen_savedir'])
        kwargs['instance_name'] = instance_name
        return original_parallelize(model, dummy_args, **kwargs)

    monkeypatch.setattr(driver, 'parallelize', capture_parallelize)
    kwargs = {} if family == 'Deepseek' else {'multimodal': preset == 'micro-vl'}
    compile_model(preset, strategy, tmp_path, batch_size=2, seq_len=16,
                  weightless=False, dtype=torch.float32, **kwargs)
    reference = captured['model']
    template = captured['samples']
    hf_model = captured['hf_model']
    if family == 'Deepseek':
        from examples.Deepseek.rewrites.model_rewrites import DeepseekCausalLMTaskWrapper
        golden = DeepseekCausalLMTaskWrapper(hf_model, sum_ce_loss=False)
    else:
        from examples.qwen3p5.rewrites.model_rewrites import Qwen3_5CompileWrapper
        golden = Qwen3_5CompileWrapper(hf_model, loss_reduction='mean')
    permutations = _hf_parameter_permutations(reference, family)
    generated_class = _load_parallel_module_class(
        type(reference), gen_savedir=captured['gen_savedir'], instance_name=instance_name, rank=0,
    )
    generated = torch.nn.Module()
    metadata = generated_class.get_attr_meta_map(0)
    for name, attr in metadata.items():
        if attr.is_param:
            parameter = reference.get_parameter(attr.orig_name)
            generated.register_parameter(name, torch.nn.Parameter(parameter.detach().clone(),
                                                                  requires_grad=parameter.requires_grad))
        else:
            generated.register_buffer(name, reference.get_buffer(attr.orig_name).detach().clone())
    for name, method in vars(generated_class).items():
        if name.startswith('segment'):
            setattr(generated, name, MethodType(method, generated))
    assert {attr.orig_name for attr in metadata.values() if attr.is_param} == set(dict(reference.named_parameters()))
    assert set(dict(golden.named_parameters())) == set(dict(reference.named_parameters()))
    optimizer_class = torch.optim.SGD if optimizer_name == 'sgd' else torch.optim.AdamW
    optimizer_kwargs = {'lr': 0.001} if optimizer_name == 'sgd' else {
        'lr': 0.0003, 'eps': 1e-6 if optimizer_name == 'adamw_eps1e6' else 1e-8,
    }
    reference_optimizer = optimizer_class(reference.parameters(), **optimizer_kwargs)
    generated_optimizer = optimizer_class(generated.parameters(), **optimizer_kwargs)
    golden_optimizer = optimizer_class(golden.parameters(), **optimizer_kwargs)
    results = []
    for step in range(10):
        samples = _training_batch(template, step, hf_model.get_input_embeddings().num_embeddings,
                                  getattr(hf_model.config, 'image_token_id', None))
        golden_samples = {name: value for name, value in samples.items() if not name.startswith('vision_')}
        reference.zero_grad()
        golden.zero_grad()
        with _hf_reference_context(family):
            golden_loss = golden(golden_samples)
            golden_loss.backward()
        expected_loss = reference(samples)
        expected_loss.backward()
        with monkeypatch.context() as device_context:
            device_context.setattr(torch.cuda, 'current_device', lambda: 'cpu')
            actual_loss = generated_class._train_step(generated, iter([samples]))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual_loss, golden_loss, rtol=1e-4, atol=1e-5)
        errors = {'step': step, 'hf_loss': golden_loss.item(), 'eager_loss': expected_loss.item(),
                  'nnscaler_loss': actual_loss.item(), 'eager_grad_max_abs': 0.0, 'hf_grad_max_abs': 0.0,
                  'eager_weight_max_abs': 0.0, 'hf_weight_max_abs': 0.0}
        grad_diff_sq = grad_norm_sq = weight_diff_sq = weight_norm_sq = 0.0
        for name, attr in metadata.items():
            if not attr.is_param:
                continue
            actual = generated.get_parameter(name).grad
            expected = reference.get_parameter(attr.orig_name).grad
            hf_grad = golden.get_parameter(attr.orig_name).grad
            assert (actual is None) == (expected is None), attr.orig_name
            assert (actual is None) == (hf_grad is None), attr.orig_name
            if expected is not None:
                if attr.orig_name in permutations:
                    hf_grad = hf_grad.index_select(0, permutations[attr.orig_name])
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, msg=attr.orig_name)
                if optimizer_name == 'sgd':
                    torch.testing.assert_close(actual, hf_grad, rtol=1e-4, atol=1e-5, msg=attr.orig_name)
                errors['eager_grad_max_abs'] = max(errors['eager_grad_max_abs'], (actual - expected).abs().max().item())
                errors['hf_grad_max_abs'] = max(errors['hf_grad_max_abs'], (actual - hf_grad).abs().max().item())
                grad_diff_sq += (actual.double() - hf_grad.double()).square().sum().item()
                grad_norm_sq += hf_grad.double().square().sum().item()
        reference_optimizer.step()
        generated_optimizer.step()
        golden_optimizer.step()
        for name, attr in metadata.items():
            if attr.is_param:
                actual = generated.get_parameter(name)
                expected = reference.get_parameter(attr.orig_name)
                hf_parameter = golden.get_parameter(attr.orig_name)
                if attr.orig_name in permutations:
                    hf_parameter = hf_parameter.index_select(0, permutations[attr.orig_name])
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, msg=attr.orig_name)
                if optimizer_name == 'sgd':
                    torch.testing.assert_close(actual, hf_parameter, rtol=1e-4, atol=1e-5, msg=attr.orig_name)
                errors['eager_weight_max_abs'] = max(errors['eager_weight_max_abs'], (actual - expected).abs().max().item())
                errors['hf_weight_max_abs'] = max(errors['hf_weight_max_abs'], (actual - hf_parameter).abs().max().item())
                weight_diff_sq += (actual.double() - hf_parameter.double()).square().sum().item()
                weight_norm_sq += hf_parameter.double().square().sum().item()
        errors['hf_grad_relative_l2'] = (grad_diff_sq / max(grad_norm_sq, 1e-30)) ** 0.5
        errors['hf_weight_relative_l2'] = (weight_diff_sq / max(weight_norm_sq, 1e-30)) ** 0.5
        assert errors['hf_grad_relative_l2'] < 1e-3, errors
        assert errors['hf_weight_relative_l2'] < 1e-3, errors
        results.append(errors)
    report = {'family': family, 'preset': preset, 'strategy': strategy_name, 'device': 'cpu',
              'dtype': 'float32', 'optimizer': optimizer_name, 'optimizer_kwargs': optimizer_kwargs, 'results': results}
    (tmp_path / 'training_metrics.json').write_text(json.dumps(report, indent=2))
    print('TRAINING_METRICS ' + json.dumps(report))


def _distributed_training(family, preset, strategy_name, output_dir):
    import nnscaler
    from nnscaler.parallel import parallelize, build_optimizer

    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=90))
    nnscaler.init()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(20261127)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    driver = importlib.import_module(f'examples.{family}.compile')
    repo_root = Path(nnscaler.__file__).resolve().parent.parent
    strategy = driver.load_strategy_config(repo_root / 'examples' / family / 'strategies' / f'{strategy_name}.json')
    config_dict = driver.load_model_config_dict(preset)
    if family == 'Deepseek':
        driver.register_all_deepseek_ops()
        source, _ = driver.build_deepseek_model(config_dict, rewrite_cfg=strategy.rewrites,
                                               weightless=False, dtype=torch.float32)
        template = driver.build_dummy_samples(config_dict, batch_size=2, seq_len=16)
    else:
        driver.register_all_qwen3_5_ops()
        source, _ = driver.build_qwen3_5_model(config_dict, rewrite_cfg=strategy.rewrites,
                                              multimodal=preset == 'micro-vl', weightless=False, dtype=torch.float32)
        template = driver.build_dummy_samples(config_dict, batch_size=2, seq_len=16,
                                              multimodal=preset == 'micro-vl', dtype=torch.float32)
    compute_config = driver.build_compute_config(strategy, inference_only=False, trace_strategy='cpu', weightless=False)
    generated = parallelize(source, {'samples': template}, driver.build_pas_policy(strategy), compute_config,
                            gen_savedir=output_dir, instance_name='distributed', reuse='override',
                            broadcast_strategy='none')
    print(f'PARITY rank={rank} compiled', flush=True)
    reference = source.cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    generated_optimizer = build_optimizer(generated, torch.optim.SGD, lr=0.001)
    parameters = dict(reference.named_parameters())
    for name, attr in generated.fullmap.items():
        if attr.is_param:
            assert attr.val_chunks == 1
            torch.testing.assert_close(generated.get_parameter(name), parameters[attr.orig_name][attr.slicers])
    results = []
    for step in range(10):
        samples = _training_batch(template, step, source.model.get_input_embeddings().num_embeddings,
                                  getattr(source.model.config, 'image_token_id', None))
        samples = {name: value.cuda() for name, value in samples.items()}
        reference_optimizer.zero_grad()
        generated_optimizer.zero_grad()
        expected_loss = reference(samples)
        expected_loss.backward()
        print(f'PARITY rank={rank} step={step} eager_backward_done', flush=True)
        actual_loss = generated.train_step([samples])[0]
        print(f'PARITY rank={rank} step={step} generated_backward_done', flush=True)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-5)
        generated_optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        weight_max_abs = 0.0
        for name, attr in generated.fullmap.items():
            if attr.is_param:
                actual = generated.get_parameter(name)
                expected = parameters[attr.orig_name][attr.slicers]
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5,
                                           msg=lambda detail: f'rank={rank}, step={step}, {attr.orig_name}\n{detail}')
                weight_max_abs = max(weight_max_abs, (actual - expected).abs().max().item())
        results.append({'step': step, 'nnscaler_loss': actual_loss.item(), 'eager_loss': expected_loss.item(),
                        'weight_max_abs': weight_max_abs})
    report = {'family': family, 'preset': preset, 'strategy': strategy_name, 'rank': rank,
              'world_size': torch.distributed.get_world_size(), 'dtype': 'float32', 'optimizer': 'SGD', 'results': results}
    (Path(output_dir) / f'distributed_metrics_{rank}.json').write_text(json.dumps(report, indent=2))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize('family,preset,strategy_name,batch_size,seq_len',
                         [case for case in CASES if case[1].startswith('micro')])
def test_distributed_micro_training(tmp_path, family, preset, strategy_name, batch_size, seq_len):
    import nnscaler

    driver = importlib.import_module(f'examples.{family}.compile')
    repo_root = Path(nnscaler.__file__).resolve().parent.parent
    strategy = driver.load_strategy_config(repo_root / 'examples' / family / 'strategies' / f'{strategy_name}.json')
    ranks = strategy.mesh.plan_ngpus
    if torch.cuda.device_count() < ranks:
        pytest.skip(f'requires {ranks} GPUs')
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ranks}',
         str(Path(__file__).resolve()), '--distributed', family, preset, strategy_name, str(tmp_path)],
        cwd=repo_root, env=dict(os.environ, OMP_NUM_THREADS='1', TORCH_DISTRIBUTED_DEBUG='DETAIL'),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-16000:]
    reports = list(tmp_path.glob('distributed_metrics_*.json'))
    assert len(reports) == ranks
    for path in reports:
        assert len(json.loads(path.read_text())['results']) == 10


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--distributed':
        _distributed_training(*sys.argv[2:])
    else:
        raise SystemExit('Use pytest, or --distributed FAMILY PRESET STRATEGY OUTPUT_DIR')