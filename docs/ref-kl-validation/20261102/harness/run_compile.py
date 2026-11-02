"""Compile the actual 150B production YAML locally; never execute its setup."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=Path, required=True)
    parser.add_argument('--nnscaler', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    llm, nn = args.llm.resolve(), args.nnscaler.resolve()
    run = ROOT / 'compile' / args.name
    run.mkdir(parents=True, exist_ok=False)
    artifact = Path('/mnt/pvc/yileiyang/codex-ref-kl-opt-20261103') / args.name
    artifact.mkdir(parents=True, exist_ok=False)
    config = llm / 'yamls/moe-150A9B-260923/stage-1.2.256k.pp3ep32cp32dp16.mbs1.192nodes.no-cross.vpp4-342.yaml'
    job = yaml.safe_load(config.read_text())['search']['job_template']
    commands = job['command']
    command, = [c for c in commands if 'nnscaler_train_vl.py' in c]
    tokens = shlex.split(command.split('nnscaler_train_vl.py', 1)[1], comments=True)
    replacements = {
        '--runtime_ngpus': '1536', '--save_path': str(run / 'save'),
        '--gen_savedir': str(artifact), '--wandb_project': 'agens', '--name': 'agens',
        '--ref_checkpoint_path': '/mnt/pvc/lidong1/exp/agens/150A9B-52M/merged-0000-303000/merged.ckpt',
        '--vision_encoder_path': '/mnt/pvc/zhiliang/vl/exp/Kimi-K3-ViT-init-3072',
        '--vl_shared_array_dir': str(run / 'shared'),
    }
    for key, value in replacements.items():
        if key in tokens:
            tokens[tokens.index(key) + 1] = value
    tokens += ['--run_mode', 'compile', '--gen_reuse', 'moo', '--no_save']
    if any('$' in token or '{' in token for token in tokens):
        raise RuntimeError('Unresolved YAML command variable')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=f'{nn}:/tmp/dion:{llm}/llm',
               NNSCALER_ROOT=str(nn), NNSCALER_TEST_ROOT=str(nn),
               REF_KL_LLM_ROOT=str(llm), REF_KL_PROFILE_DIR=str(run),
               REF_KL_EXTERNAL_ARTIFACT_ROOT=str(artifact),
               WANDB_MODE='disabled', WANDB_API_KEY='', WANDB_API_KEY_FILE='/dev/null',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               ATTR_SAVE_WORKERS='8', TORCHINDUCTOR_COMPILE_THREADS='4',
               PYTHONUNBUFFERED='1',
               TRITON_CACHE_DIR=str(run / 'cache/triton'),
               TORCHINDUCTOR_CACHE_DIR=str(run / 'cache/inductor'),
               FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(run / 'cache/cute'),
               DG_JIT_CACHE_DIR=str(run / 'cache/deepgemm'), CUDA_CACHE_PATH=str(run / 'cache/cuda'),
               TORCH_EXTENSIONS_DIR=str(run / 'cache/torch_extensions'))
    (run / 'plan.json').write_text(json.dumps(dict(source=str(config), arguments=tokens,
        artifacts=str(artifact), note='Original model/sequence/topology preserved; single local compile launcher; task-owned output paths'), indent=2)+'\n')
    command = [sys.executable, str(ROOT / 'scripts/resource_guard_v2.py'), '--gpu', '--timeout', '10800',
               '--log', str(run / 'compile.log'), '--', 'bash', '-c',
               'source "$1"\nshift\nexec "$@"', 'ref-kl-compile', str(llm / 'scripts/k8s_env.sh'),
               sys.executable, str(ROOT / 'scripts/profile_compile.py'), *tokens]
    print(json.dumps({'run':str(run),'artifacts':str(artifact)}),flush=True)
    raise SystemExit(subprocess.run(command,cwd=llm/'llm',env=env).returncode)


if __name__ == '__main__':
    main()
