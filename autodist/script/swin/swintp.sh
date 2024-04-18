#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=35 # in GB
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="swin"
topk=1
cache_folder1="autodist/cost_model/comm/__pycache__"
cache_folder2="autodist/cost_model/__pycache__"
comm_dev=(2 4)

for ((i=0; i<${#comm_dev[*]}; i=i+1)); do
    torchrun --nproc_per_node=${comm_dev[i]} comm_profile.py --connect_type=$connect_type
done

if [ ! -d $save_folder ]
then
    mkdir $save_folder
fi

# We run all cases in the machine with 4 gpus.

bs=(1 2 4 8 16 32)

mesh_cols=(1 2 4)
mesh_rows=(1 1 1)
setting=('toy' '355M' '1.8B')

for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do

        echo "start runtime Swin setting=${setting[k]} bs=${bs[j]}"
        if [ -d $cache_folder1 ]
        then
            echo "Removing $cache_folder1 directory..."
            rm -r $cache_folder1
            rm -r $cache_folder2
        else
            echo "$cache_folder1 directory not found"
        fi

        LOG_TRANSFORM=1 SINGLE_DEV_MODE=1 python main.py --is_train \
                --memory_constraint=$memory_constraint  \
                --memory_granularity=$memory_granularity \
                --save_folder=$save_folder --connect_type=$connect_type \
                --mesh_row=1 --mesh_col=${mesh_cols[k]} --compile \
                --topk=$topk \
                --verbose --swin_setting=${setting[k]} --swin \
                --micro_batch_size=${bs[j]} \
                --global_batch_size=32  --recompute --adaptive_recom

        torchrun --master_port=30001 --nnodes=${mesh_rows[k]} \
                --nproc_per_node=${mesh_cols[k]} main.py --is_train \
                --memory_constraint=$memory_constraint  \
                --memory_granularity=$memory_granularity --save_folder=$save_folder \
                --connect_type=$connect_type --mesh_row=${mesh_rows[k]} --mesh_col=${mesh_cols[k]} \
                --plan_idx=0   --iter_num=2 --warm_num=1 --micro_batch_size=${bs[j]} \
                --global_batch_size=32  --swin_setting=${setting[k]} --swin  \
                --recompute
    done
done

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "allRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
