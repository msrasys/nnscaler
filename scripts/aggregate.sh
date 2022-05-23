# ============= ITP Variables ============
# NODE_RANK
# MASTER_IP
# MASTER_PORT
# ============= ITP Variables ============

node_num=$1

if [ ${node_num} == 4 ]
then
    mkdir -p /workspace/MagicCube/eval/worker-1
    scp -r worker-1:/workspace/MagicCube/eval/ /workspace/MagicCube/eval/worker-1
    mkdir -p /workspace/MagicCube/eval/worker-2
    scp -r worker-2:/workspace/MagicCube/eval/ /workspace/MagicCube/eval/worker-2
    mkdir -p /workspace/MagicCube/eval/worker-3
    scp -r worker-3:/workspace/MagicCube/eval/ /workspace/MagicCube/eval/worker-3
fi

if [ ${node_num} == 2 ]
then
    mkdir -p /workspace/MagicCube/eval/worker-1
    scp -r worker-1:/workspace/MagicCube/eval/ workspace/MagicCube/eval/worker-1
fi
