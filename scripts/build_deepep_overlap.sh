#!/usr/bin/env bash
# Build the optional B200 EP2 transport variant in a new, isolated directory.
# Usage: build_deepep_overlap.sh /opt/DeepEP /absolute/path/to/new/output
# NNSCALER_DEEPEP_COMPACT_WARPS=8 selects the optional narrow-payload variant.
# NNSCALER_DEEPEP_PEER_EXCHANGE=1 adds the optional same-node peer copy API.
set -euo pipefail
if (( $# != 2 )); then
    echo 'Usage: build_deepep_overlap.sh SOURCE NEW_OUTPUT' >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CUDA_HOME:?Initialize the CUDA build environment before running this script}"
: "${NVSHMEM_DIR:?Set NVSHMEM_DIR to the installed NVSHMEM package}"
python - "$1" "$2" "$script_dir/patches" "${NNSCALER_DEEPEP_COMPACT_WARPS:-4}" "${NNSCALER_DEEPEP_PEER_EXCHANGE:-0}" <<'PY'
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
source=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();patches=Path(sys.argv[3])
if sys.argv[4] not in ('4', '8'):
    raise SystemExit('NNSCALER_DEEPEP_COMPACT_WARPS must be 4 or 8.')
warps=int(sys.argv[4])
if sys.argv[5] not in ('0','1'):
    raise SystemExit('NNSCALER_DEEPEP_PEER_EXCHANGE must be 0 or 1.')
peer_exchange=sys.argv[5]=='1'
expected=(patches/'deepep_ep2_compact.sha256').read_text().strip()
file=source/'csrc/kernels/intranode.cu'
if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
    raise SystemExit('DeepEP source does not match the reviewed transport patch revision.')
if output.exists() or source==output or source in output.parents:
    raise SystemExit('NEW_OUTPUT must be absent and outside the source checkout.')
patch=patches/'deepep_ep2_compact.patch'
if peer_exchange:
    for name,digest in json.loads((patches/'deepep_peer_exchange.sha256.json').read_text()).items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('DeepEP host source does not match the reviewed peer exchange revision: '+name)
shutil.copytree(source,output,ignore=shutil.ignore_patterns('.git','build','*.so','__pycache__','*.egg-info'))
selected=[patch] + ([patches/'deepep_ep2_warp8.patch'] if warps == 8 else [])
if peer_exchange:selected.append(patches/'deepep_peer_exchange.patch')
for selected_patch in selected:
    subprocess.run(['patch','--batch','--forward','--dry-run','-p1'],input=selected_patch.read_bytes(),cwd=output,check=True)
    subprocess.run(['patch','--batch','--forward','-p1'],input=selected_patch.read_bytes(),cwd=output,check=True)
(output/'overlap-source.json').write_text(json.dumps(dict(source=str(source),source_sha256=expected,
    patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),compact_warps=warps,peer_exchange=peer_exchange,
    patches_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}),indent=2)+'\n')
PY
export CUDA_VISIBLE_DEVICES=''
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0}
export MAX_JOBS=${MAX_JOBS:-2}
export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
cd -- "$2"
python setup.py build_ext --inplace
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('deep_ep_cpp*.so')}
if not files:raise SystemExit('DeepEP extension was not produced.')
(root/'overlap-build.json').write_text(json.dumps(files,indent=2)+'\n')
print('Built isolated overlap backend in',root)
PY
