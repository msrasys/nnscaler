"""Run the production VL entry point with a PP3/CP2/EP2 15B surrogate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=Path, required=True)
    parser.add_argument('--nnscaler', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--sequence', type=int, default=16384)
    parser.add_argument('--port', type=int, default=29861)
    parser.add_argument('--pattern', default='ref_kl')
    args = parser.parse_args()
    llm, nn = args.llm.resolve(), args.nnscaler.resolve()
    run = ROOT / 'runs' / args.name
    run.mkdir(parents=True, exist_ok=False)
    for folder in ['audit', 'gencode', 'shared', 'tmp', 'cache', 'control']:
        (run / folder).mkdir()
    for name in ['vl_trajectory_observer.py', 'vl_resume_observer_v2.py']:
        shutil.copyfile(ROOT / 'scripts' / name, run / 'control' / name)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5', PYTHONPATH=f'{nn}:/tmp/dion:{llm}/llm',
               NNSCALER_ROOT=str(nn), NNSCALER_TEST_ROOT=str(nn),
               VL_BENCH_LLM_ROOT=str(llm), VL_BENCH_AUDIT_DIR=str(run / 'audit'),
               VL_BENCH_EXPECTED_SHA=sha(nn), VL_BENCH_EXPECTED_LLM_SHA=sha(llm),
               MAX_UPDATES=str(args.steps), GEN_SAVEDIR=str(run / 'gencode'),
               WANDB_MODE='disabled', WANDB_API_KEY='', WANDB_API_KEY_FILE='/dev/null',
               DETERMINISTIC='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               ATTR_SAVE_WORKERS='4', TORCHINDUCTOR_COMPILE_THREADS='2',
               TORCHINDUCTOR_WORKER_START='spawn', PYTHONUNBUFFERED='1',
               TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/inductor'),
               FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(ROOT / 'cache/cute'),
               DG_JIT_CACHE_DIR=str(ROOT / 'cache/deepgemm'),
               CUDA_CACHE_PATH=str(ROOT / 'cache/cuda'), TMPDIR=str(run / 'tmp'))
    if args.reference:
        env['VL_TRAJECTORY_REFERENCE'] = str(args.reference.resolve())
    else:
        env.pop('VL_TRAJECTORY_REFERENCE', None)
    training = [
        '--model', 'YOCO-MoE-15B-A6B-L4', '--data', 'phase1-config-11T-260720-nfs',
        '--data_path', '/data/tmp/ref-kl-opt-20261103/data/mixture.json', '--data_weight', '1.0',
        '--hyperparams', 'warmup_constant', '--num_workers', '4',
        '--batch_size', '1', '--update_freq', '12', '--max_updates', str(args.steps),
        '--max_seq_len', str(args.sequence), '--universal_loop', '3',
        '--adam_eps', '1e-10', '--learning_rate', '3e-5', '--min_lr', '3e-5',
        '--lr_decay_style', 'linear', '--warmup_iters', '5',
        '--plan_ngpus', '6', '--runtime_ngpus', '6', '--zero_group_size', '6',
        '--gpu_mem_constraint', '160', '--disable_shared_param_constraint', '--use_fixed_policy',
        '--fixed_stage_block_counts', '3,3,3|3,3,3|3,3,3|3,4,2',
        '--pipeline_nstages', '12', '--pp_size', '3', '--pipeline_scheduler', '1f1b_interleaved',
        '--max_partition_degree', '2', '--partition_constraints_path', './pcs/all2all_moe.yaml',
        '--cp_size', '2', '--ep_size', '2', '--use_cute', '--precision', 'bf16',
        '--quant_mode', 'mxfp8', '--xentropy_chunk_size', '8192',
        '--moe_loss_scale', '10' if args.pattern == 'ref_kl' else '0.1', '--moe_loss_pattern', args.pattern,
        '--balance_mode', 'subset', '--vl_balance_mode', 'subset',
        '--weight_decay', '0.1', '--enable_muon_optimizer', '--muon_weight_decay', '0.12',
        '--mtp_n', '1', '--mtp_loss_scale', '0.2', '--recompute_modules', 'Block',
        '--reducer_bucket_cap_mb', '200', '--broadcast_strategy', 'none',
        '--vision_encoder_path', '/mnt/pvc/zhiliang/vl/exp/Kimi-K3-ViT-init-3072',
        '--vision_pipeline_mode', 'first_stage', '--no_vision_merge_microbatches', '--vision_freeze',
        '--vision_max_image_tokens', '4096', '--vision_align_mode', 'resize',
        '--vl_dataloader_worker_queue_size', '8', '--vl_dataloader_prefetch_queue_size', '24',
        '--vl_shared_array_dir', str(run / 'shared'), '--vl_image_normalize_in_patch_layout',
        '--diagnostic_interval', '100', '--detail_log_freq', '1', '--codegen_workers', '4',
        '--gen_savedir', str(run / 'gencode'), '--gen_reuse', 'moo',
        '--parallel_mmap_checkpoint_load', '--skip_empty_vision', '--per_token_clip_grad', '2e4',
        '--pr-data-chunk-from-tail', '--cross_file_packing', '--mask_media_base64_loss',
        '--seed', '2611012040', '--save_path', str(run / 'checkpoints'), '--no_save',
        '--no_async_save_ckpt', '--wandb_project', 'ref-kl-local-validation', '--name', 'ref-kl-local',
    ]
    plan = dict(llm_sha=sha(llm), nnscaler_sha=sha(nn), arguments=training,
                production='stage-1.2.256k.pp3ep32cp32dp16.mbs1.192nodes.no-cross.vpp4-342.yaml',
                differences=['15B surrogate', 'PP3/CP2/EP2/DP1 on six local GPUs',
                             '16K global / 8K local tokens instead of 256K / CP32',
                             'fixed real 90:10 nemotron text / grounded-caption VL mixture (four readable shards each)',
                             'reference snapshots initial student; no 150B checkpoint loading',
                             '5-step warmup; smaller 8/24 data prefetch queues; deterministic verification; no checkpoint writes'])
    (run / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    command = [sys.executable, str(ROOT / 'scripts/resource_guard_v2.py'), '--gpu', '--timeout', '7200',
               '--log', str(run / 'train.log'), '--', 'bash', '-c',
               'source "$1"\nshift\nexec "$@"', 'ref-kl-run', str(llm / 'scripts/k8s_env.sh'),
               'torchrun', '--standalone', '--nproc-per-node=6', '--master-port', str(args.port),
               str(run / 'control/vl_trajectory_observer.py'), *training]
    print(json.dumps({'run': str(run), 'plan': plan}), flush=True)
    result = subprocess.run(command, cwd=llm / 'llm', env=env)
    if result.returncode == 0:
        saved = run / 'generated-source'
        for path in (run / 'gencode').rglob('*'):
            if not path.is_file() or not (path.suffix == '.py' or path.name in ('graph.ckp', 'compute_config.pt', 'dist_param_map.pt')):
                continue
            target = saved / path.relative_to(run / 'gencode')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.rmtree(run / 'gencode')
        shutil.rmtree(run / 'shared')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
