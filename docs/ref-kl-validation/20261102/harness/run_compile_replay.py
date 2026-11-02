"""Regenerate target ranks from the same captured IR, without retracing/weight copies."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--nnscaler',type=Path,required=True);p.add_argument('--name',required=True);args=p.parse_args()
source=ROOT/'compile'/args.source
plan=json.loads((source/'plan.json').read_text())
run=ROOT/'compile'/args.name;run.mkdir(parents=True,exist_ok=False)
artifact=Path('/mnt/pvc/yileiyang/codex-ref-kl-opt-20261103')/args.name;artifact.mkdir(exist_ok=False)
origin=Path(plan['artifacts'])
for src in origin.rglob('*'):
    if not src.is_file():continue
    dst=artifact/src.relative_to(origin)
    if src.name.startswith('fullmodel.pt.') or src.name=='npbuffer.pt':
        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
    elif src.name in ('graph.ckp','forward_args.pkl','origin_module_metadata.pt','compute_config.pt','dist_param_map.pt'):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
a=plan['arguments']
for key,value in {'--gen_savedir':str(artifact),'--save_path':str(run/'save'),'--vl_shared_array_dir':str(run/'shared'),'--gen_reuse':'graph'}.items():a[a.index(key)+1]=value
llm=ROOT/'llm-train-base';nn=(ROOT/'nnscaler') if args.name == 'production-replay-candidate' else args.nnscaler.resolve()
e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=f'{nn}:/tmp/dion:{llm}/llm',NNSCALER_ROOT=str(nn),NNSCALER_TEST_ROOT=str(nn),REF_KL_LLM_ROOT=str(llm),REF_KL_PROFILE_DIR=str(run),REF_KL_EXTERNAL_ARTIFACT_ROOT=str(artifact),WANDB_MODE='disabled',WANDB_API_KEY='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TORCHINDUCTOR_COMPILE_THREADS='4',PYTHONUNBUFFERED='1')
plan.update(arguments=a,artifacts=str(artifact),replay_source=str(origin),note='Reused the same complete pre-partition IR; excludes tracing and attribute serialization')
(run/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cmd=[sys.executable,str(ROOT/'scripts/resource_guard_v2.py'),'--gpu','--timeout','3600','--log',str(run/'compile.log'),'--','bash','-c','source "$1"\nshift\nexec "$@"','ref-kl-replay',str(llm/'scripts/k8s_env.sh'),sys.executable,str(ROOT/'scripts/profile_compile_replay.py'),*a]
raise SystemExit(subprocess.run(cmd,cwd=llm/'llm',env=e).returncode)
