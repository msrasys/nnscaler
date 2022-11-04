# ============= ITP Variables ============
# NODE_RANK
# MASTER_IP
# MASTER_PORT
# ============= ITP Variables ============

node_num=$1
folder=$2

host=worker

if [ ${node_num} == 8 ]
then
    scp -r /workspace/MagicCube/$folder $host-1:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-2:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-3:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-4:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-5:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-6:/workspace/MagicCube/
    scp -r /workspace/MagicCube/$folder $host-7:/workspace/MagicCube/
fi

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


# rm -f notify.py
# wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/email/notify.py
# python notify.py --sender zhiqi.0@qq.com --code uyakwgslumknbfgg --recver zhiqi.0@outlook.com \
#     --msg "Test Results Swin Coshard | 32 GPU" \
#     --file logs/e2e-swin-32gpu-coshard-${NODE_RANK}.txt