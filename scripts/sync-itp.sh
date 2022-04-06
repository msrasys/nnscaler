# ============= ITP Variables ============
# NODE_RANK
# MASTER_IP
# MASTER_PORT
# ============= ITP Variables ============

node_num=$1

if [ ${node_num} == 4 ]
then
    scp -r /workspace/MagicCube/handcraft worker-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube worker-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/handcraft worker-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube worker-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/handcraft worker-3:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube worker-3:/workspace/MagicCube/
fi

if [ ${node_num} == 2 ]
then
    scp -r /workspace/MagicCube/handcraft worker-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/cube worker-1:/workspace/MagicCube/
fi