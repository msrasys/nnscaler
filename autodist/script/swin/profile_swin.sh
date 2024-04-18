#!/bin/bash --login
start_time=$(date +%s)

memory_constraint=30
memory_granularity=1 # in byte
connect_type='NV2'
save_folder="tp_data"
topk=20

comm_dev=(2 4)

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

bs=(1 2 4 8 16)
mesh_cols=(1 2 4)
setting=('toy' '355M' '1.8B')


for ((k=0; k<${#mesh_cols[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do

        count=$((k * ${#bs[*]} + j))
        q=$(expr $count % $num_gpus)

        echo "start runtime Swin setting=${setting[k]} bs=${bs[j]}"

        CUDA_VISIBLE_DEVICES=$q  LOG_TRANSFORM=1 SINGLE_DEV_MODE=1 python main.py --is_train \
                --memory_constraint=$memory_constraint  \
                --memory_granularity=$memory_granularity \
                --save_folder=$save_folder --connect_type=$connect_type \
                --mesh_row=1 --mesh_col=${mesh_cols[k]} --compile \
                --topk=$topk \
                --verbose --swin_setting=${setting[k]} --swin \
                --recompute --micro_batch_size=${bs[j]} --global_batch_size=32 &

        if  [ "$q" -eq "$((num_gpus-1))" ]; then
           wait
        fi

    done
done

wait

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "allRun.sh spends $(($cost_time/60))min $(($cost_time%60))s"
