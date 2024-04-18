#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=38 # in GB
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="alphafold"
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

mesh_rows=(1 1 1)
mesh_cols=(1 2 4)
setting=(1 2 3)
layer=48

for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#setting[*]}; j=j+1)); do

        echo "start runtime Alphafold2 setting=${setting[j]} gpus=${mesh_cols[k]}"
        if [ -d $cache_folder1 ]
        then
            echo "Removing $cache_folder1 directory..."
            rm -r $cache_folder1
            rm -r $cache_folder2
        else
            echo "$cache_folder1 directory not found"
        fi

        SINGLE_DEV_MODE=1 python main.py --is_train \
                    --memory_constraint=$memory_constraint  \
                    --memory_granularity=$memory_granularity \
                    --save_folder=$save_folder --connect_type=$connect_type \
                    --mesh_row=${mesh_rows[k]} --mesh_col=${mesh_cols[k]} --compile \
                    --topk=$topk  --ignore_small_tensor_threshold=2048 \
                    --verbose --alphafold_setting=${setting[j]} --alphafold \
                    --alphafold_layer=$layer --recompute --adaptive_recom

        torchrun --master_port=30001 --nnodes=${mesh_rows[k]} \
                --nproc_per_node=${mesh_cols[k]} main.py --is_train \
                --memory_constraint=$memory_constraint  \
                --memory_granularity=$memory_granularity --save_folder=$save_folder \
                --connect_type=$connect_type --mesh_row=${mesh_rows[k]} --mesh_col=${mesh_cols[k]} \
                --plan_idx=0   --iter_num=4 --warm_num=2  \
                --global_batch_size=1  --alphafold  --ignore_small_tensor_threshold=2048 \
                --alphafold_setting=${setting[j]} --alphafold_layer=$layer --recompute
    done
done

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "allRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
