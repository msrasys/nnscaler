#!/bin/bash

# For parity check.
# Example:
#   bash parity_alert.sh <A_folder_to_put_all_source_code> <data_folder> [<parity_check_config_folder>]
# <A_folder_to_put_all_source_code>: the workspace where all codes are stored.
# <data_folder>: the folder when the train data for torchscale is stored.
# <parity_check_config_folder>: the definition of parity check.
#    Default value is ${the dir of the current script}/test_cases/
# Options:
#       --cube-branch-gt <cube_branch_for_ground_truth>: default is main
#       --cube-branch <the new cube_branch to check>: default is main
#       --conda-base <the env from which the new env will clone>: default is base
#       --test-cases <the test cases to run, split with comma>: default is all
#           The test cases are listed under <parity_check_config_folder> (`test_cases/`) folder, e.g., pasdata, dp2, tp2, hybrid2.
#
# Currently the workspace is not cleared after execution, so it can help fix the parity problem if any.
# To clean the workspace
#   1. run `rm -rf <A_folder_to_put_all_source_code>` to clean the cloned source code.
#   2. run `conda env remove -n parity` to remove conda env.

set -e

export NCCL_DEBUG=WARN

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
POSITIONAL_ARGS=()

CUBE_BRANCH_GT=main

CUBE_BRANCH_NEW=main

CONDA_ENV_BASE=base
TEST_CASES=

while [[ $# -gt 0 ]]; do
  case $1 in
    --cube-branch-gt)
      CUBE_BRANCH_GT="$2"
      shift # past argument
      shift # past value
      ;;
    --cube-branch)
      CUBE_BRANCH_NEW="$2"
      shift # past argument
      shift # past value
      ;;
    --conda-base)
      CONDA_ENV_BASE="$2"
      shift # past argument
      shift # past value
      ;;
    --test-cases)
      TEST_CASES="$2"
      shift # past argument
      shift # past value
      ;;
        -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

OPERATION=$1

if [[ $# -ne 2 ]] && [[ $# -ne 3 ]]; then
    echo "usage: $0 WORKSPACE TRAIN_DATA_DIR [PARITY_CHECK_DATA_DIR]"
    echo "  [--cube-branch-gt <cube_branch_for_ground_truth>]"
    echo "  [--cube-branch <the new cube_branch to check>]"
    echo "  [--conda-base <the env from which the new env will clone>]"
    echo "  [--test-cases <the test cases to run, split with comma>]"
    exit 1
fi


WORKSPACE=$1
TRAIN_DATA_DIR=$2
PARITY_CHECK_DATA_DIR=${3:-${SCRIPT_DIR}/test_cases}

if [[ -d $WORKSPACE ]]; then
    echo "Error: $WORKSPACE has existed, please remove the folder before running the test(s)."
    exit 2
fi


ENV_NAME=parity_$(echo $RANDOM | md5sum | head -c 10)
TMP_SETUP_ENV_SH=tmp_setup_env.sh
TMP_SWITCH_BRANCH_SH=tmp_switch_branch.sh
TMP_MODEL_DIR=result_models  # will not be removed after execution
# get an unused port
UNUSED_PORT=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`

conda create -y -n ${ENV_NAME} --clone ${CONDA_ENV_BASE}

LIBSTDC_PATH=$(conda env list | grep ${ENV_NAME} | awk '{print $NF}')/lib/libstdc++.so.6
rm -f ${LIBSTDC_PATH}

trap "rm -rf tmp_* && conda env remove -n ${ENV_NAME} -y" EXIT

cat > ${TMP_SETUP_ENV_SH} << EOF
#!/bin/bash

set -e

# init python env
pip install build

mkdir -p ${WORKSPACE}
cd ${WORKSPACE}

git clone --recursive "https://github.com/msrasys/nnscaler.git" -b $CUBE_BRANCH_GT
cd nnscaler
# Rename directory to match expected 'MagicCube' or just adapt strict usage. 
# The original script used 'MagicCube' directory name. Let's stick effectively to cloning nnscaler.
# However, train.py and others might expect import structure. 

pip install -e .

python -c 'import os,sys,nnscaler,cppimport.import_hook ; sys.path.append(os.path.dirname(nnscaler.__path__[0])) ; import nnscaler.autodist.dp_solver'
cd ..

# verify installation
python -c 'import torch; import nnscaler; print(torch.__path__, nnscaler.__path__)'

EOF

cat > ${TMP_SWITCH_BRANCH_SH} << EOF
#!/bin/bash

set -e

cd ${WORKSPACE}

cd nnscaler
git checkout $CUBE_BRANCH_NEW

pip install -e .

python -c 'import os,sys,nnscaler,cppimport.import_hook ; sys.path.append(os.path.dirname(nnscaler.__path__[0])) ; import nnscaler.autodist.dp_solver'

cd ..
EOF

export TEST_CASES="$TEST_CASES"
export TRAIN_DATA_DIR="$TRAIN_DATA_DIR"
export UNUSED_PORT="$UNUSED_PORT"
export DETERMINISTIC=1

conda run --no-capture-output -n ${ENV_NAME} bash ${TMP_SETUP_ENV_SH}

conda run --no-capture-output -n ${ENV_NAME} python ${SCRIPT_DIR}/train.py ${WORKSPACE} ${PARITY_CHECK_DATA_DIR} ${TMP_MODEL_DIR}/gt

conda run --no-capture-output -n ${ENV_NAME} bash ${TMP_SWITCH_BRANCH_SH}

conda run --no-capture-output -n ${ENV_NAME} python ${SCRIPT_DIR}/train.py ${WORKSPACE} ${PARITY_CHECK_DATA_DIR} ${TMP_MODEL_DIR}/new

conda run --no-capture-output -n ${ENV_NAME} python ${SCRIPT_DIR}/parity_check.py ${TMP_MODEL_DIR}/gt ${TMP_MODEL_DIR}/new
