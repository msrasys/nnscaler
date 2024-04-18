#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=30
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="pp_data"
topk=1

comm_dev=(2 4 8 16)

for ((i=0; i<${#comm_dev[*]}; i=i+1)); do
    torchrun --nproc_per_node=${comm_dev[i]} comm_profile.py --connect_type=$connect_type
done

if [ ! -d $save_folder ]
then
    mkdir $save_folder
fi

# We run all cases in the machine with 4 gpus.

bs=(1 2 4 8 16 32)
mesh_rows=(1 1 1 1)
mesh_cols=(2 4 8 16)
model_config=('760M' '1.3B' '2.6B' '6.7B')

for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do
        echo "start runtime ${bs[j]} ${model_config[k]}"
        SINGLE_DEV_MODE=1 python main.py --GPT_setting=${model_config[k]} --is_train --recompute \
                 --micro_batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                 --memory_granularity=$memory_granularity --save_folder=$save_folder \
                 --connect_type=$connect_type --mesh_row=${mesh_rows[k]} --mesh_col=${mesh_cols[k]} --compile --pipeline --topk=1

        torchrun --nnodes=${mesh_rows[k]} --nproc_per_node=${mesh_cols[k]} main.py --GPT_setting=${model_config[k]} --is_train --recompute \
                 --micro_batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                 --memory_granularity=$memory_granularity --save_folder=$save_folder --connect_type=$connect_type --mesh_row=${mesh_rows[k]} --mesh_col=${mesh_cols[k]} --pipeline --plan_idx=0

    done
done

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "allRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
