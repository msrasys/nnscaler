#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=30
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="tp_data"
topk=20

comm_dev=(2 4 8 16)

for ((i=0; i<${#comm_dev[*]}; i=i+1)); do
    torchrun --nproc_per_node=${comm_dev[i]} comm_profile.py --connect_type=$connect_type
done

if [ ! -d $save_folder ]
then
    mkdir $save_folder
fi

# Use nvidia-smi to get a list of GPUs
gpus=$(nvidia-smi -L)

# Count the number of lines of output
num_gpus=$(echo "$gpus" | wc -l)

bs=(1 2 4 8 16 32)
mesh_cols=(1 2 4 8 16)
model_config=('350M' '760M' '1.3B' '2.6B' '6.7B')


for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do

        count=$((k * ${#bs[*]} + j))
        q=$(expr $count % $num_gpus)

        # bash profile.sh to profile the coarse-gained GPT
        # bash profile.sh * to profile the fine-gained GPT
        if [ $# -eq 0  ]; then
            CUDA_VISIBLE_DEVICES=$q SINGLE_DEV_MODE=1 python main.py --GPT_setting=${model_config[k]} --is_train --recompute \
                    --micro_batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                    --memory_granularity=$memory_granularity --save_folder=$save_folder \
                    --connect_type=$connect_type --mesh_col=${mesh_cols[k]} --compile \
                    --topk=$topk  &

        else
            CUDA_VISIBLE_DEVICES=$q SINGLE_DEV_MODE=1 python main.py --GPT_setting=${model_config[k]} --is_train --recompute \
                    --micro_batch_size=${bs[j]} --memory_constraint=$memory_constraint  \
                    --memory_granularity=$memory_granularity --save_folder=$save_folder \
                    --connect_type=$connect_type --mesh_col=${mesh_cols[k]} --compile \
                    --topk=$topk --fine_grained_GPT &
        fi

        if  [ "$q" -eq "$((num_gpus-1))" ]; then
           wait
        fi

    done
done

wait

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "allRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
