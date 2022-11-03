# ============= ITP Variables ============
# NODE_RANK
# MASTER_IP
# MASTER_PORT
# ============= ITP Variables ============

node_num=$1
folder=$2

host=worker

if [ ${node_num} == 4 ]
then
    scp -r /workspace/MagicCube/$folder $host-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-3:/workspace/MagicCube/
fi

if [ ${node_num} == 2 ]
then
    scp -r /workspace/MagicCube/$folder $host-1:/workspace/MagicCube/
fi