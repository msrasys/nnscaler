#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=30.5
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="exp_data_test"

comm_dev=(2 4)

for ((i=0; i<${#comm_dev[*]}; i=i+1)); do
    torchrun --nproc_per_node=${comm_dev[i]} comm_profile.py --connect_type=$connect_type
done

if [ ! -d $save_folder ]
then
    mkdir $save_folder
fi

# spmd for a simple case (with 1 gpu) and a complex case (with 4 gpus).

bs=(32)
mesh_cols=(1 4)
model_config=('350M' '1.3B')

for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do

        SINGLE_DEV_MODE=1 python main.py --GPT_setting=${model_config[k]} --recompute \
                 --batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                 --memory_granularity=$memory_granularity --save_folder=$save_folder --connect_type=$connect_type --mesh_col=${mesh_cols[k]} --compile

        torchrun --nproc_per_node=${mesh_cols[k]} main.py --GPT_setting=${model_config[k]} --recompute \
                 --batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                 --memory_granularity=$memory_granularity --save_folder=$save_folder --connect_type=$connect_type --mesh_col=${mesh_cols[k]}

    done
done

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "checkRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
