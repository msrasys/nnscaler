
# ============= Singularity Variables ============
# NODE_RANK
# MASTER_ADDR
# MASTER_PORT
# ============= Singularity Variables ============

node_num=$1

if [ ${node_num} == 4 ]
then
    scp -r /workspace/MagicCube/handcraft node-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube node-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/handcraft node-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube node-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/handcraft node-3:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube node-3:/workspace/MagicCube/
fi

if [ ${node_num} == 2 ]
then
    scp -r /workspace/MagicCube/handcraft node-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube node-1:/workspace/MagicCube/
fi

