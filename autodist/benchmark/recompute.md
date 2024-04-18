# Continuous Recompute

We implement the continuous recompute search algorithm, which outperforms the alpa and the manual strategy of Megatron[1] on GPT. On the following cases, [Alpa](https://github.com/alpa-projects/alpa) is OOM and autodist has a 5.4% gain over megatron.

## Experimental Config

The model config is GPT3 760M, on 2 GPUs, num_layer increased from 24 into 48, global_batch_size = 1024 and micro_batch_size = 8.

## Results

| Search algorithm               | runtime (time/s, memory/GB) | compile time/s  | Remark   |
|:-------------------------------|:----------------------------|:----------------|:---------|
| Megatron(selective recompute)  | (137.10, 17.22)             | 8.40            | OOM      |
| Megatron(full recompute)       | (171.78,  7.09)             | 9.23            |          |
| Megatron(search)               | (154.22, 15.85)             | 9.60            |          |
| Alpa                           | (149.53, 16.15)             | 129.96          | OOM      |
| Autodist(continuous recompute) | **(145.90, 15.67)**         | 2069.80         |          |
| Autodist(single recompute)     | (146.65, 15.99)             | 130.31          | Multiref |

## Details

**Megatron:** Megatron using selective recompute(as well as using full recompute) represents selective recompute(as well as full recompute) for all layers. Megatron(search) is the optimal solution searched manually according to [args.recompute-method](https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/arguments.py#L375:~:text=group.add_argument(%27%2D%2Drecompute%2Dmethod%27%2C%20type%3Dstr%2C%20default%3DNone%2C)), [args.recompute-granularity](https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/arguments.py#L375:~:text=group.add_argument(%27%2D%2Drecompute%2Dgranularity%27%2C%20type%3Dstr%2C%20default%3DNone%2C)) and [args.recompute-num-layers](https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/arguments.py#L375:~:text=group.add_argument(%27%2D%2Drecompute%2Dnum%2Dlayers%27%2C%20type%3Dint%2C%20default%3D1%2C)) in Megatron.  Depending on the permutation of these switches, without  pipeline, we can specify that some layers recompute and others don't, and that the recompute strategy can be selective recompute or full recompute. But we can not obtain the strategy that different layers take different recompute strategies. The optimal solution of our artificial search is that Megatron(search) fully recomputes the first 31 layers.

**Alpa:** The Alpa solution is OOM with the config(dp=1, op=2, use_remat=True).

**Autodist:** The Autodist(continuous recompute) searches for a solution, where the first 23 layers fully recompute and the left 25 layers selective recompute. The compile time of Autodist(continuous recompute) is up to 15x of that of Autodist(single recompute), whose search solution has multiref BUG.

**Remark**: We use whether the memory exceeds 16G to determine if it is OOM. Because heavy fragmentation arises in the above case, memory utilization is less than 70% and difficults the memory estimation.
