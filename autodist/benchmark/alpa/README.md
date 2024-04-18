# Benchmark Alpa

## GPT-3

### Usage

For the 3d setting, the config is the same with Table 4 in [1]. For the 2d setting, we test the GPT-3 6.7B with only 4 layers. Details of the model config can be found in the `benchmark.py`, `gpt_alpa_3d.sh`, `gpt_alpa_2d_table1.sh` and `gpt_alpa_2d_table2.sh`.

You can cd the analyse_strategy folder for more specific analysis.

### Experimental Config

The benchmarks are implemented on a server runing on Ubuntu 20.04 system, which is equipped with an Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz and 16 NVIDIA V100-SXM2 32GB GPUs, each having a theoretical TFLOPS of 120 for FP16. The 16 GPUs are connected via NVLink and the interconnect bandwidth is 300GB/s (details seeing [NVIDIA TESLA V100 GPU ACCELERATOR](https://images.nvidia.com/content/technologies/volta/pdf/437317-Volta-V100-DS-NV-US-WEB.pdf). The version of CUDA is 11.3.

**w/ pipeline parallelism (i.e. 3d)**

We follow alpa's GPT-3 benchmark code (seeing Fig. 7a in [1]) on our testbed and results are in table 1.
In this case you can choose to overwrite the `benchmark.py` or not and run:

```bash
bash gpt_alpa_3d.sh
```

**w/o pipeline parallelism (i.e. 2d)**

We follow alpa's GPT-3 benchmark code under shard parallel (i.e. only intra-opeartor parallelism, no pipeline parallelism).
The results with 8 V100s are in table2.1 and those with 4 V100s in table2.2.
In this case you need to overwrite the `benchmark.py` and run:

```bash
bash gpt_alpa_2d_table1.sh

bash gpt_alpa_2d_table2.sh
```

**Description of parameters in alpa**

- `shard-only` : Only profile the 2D case. No pipeline parallelism, default=`False`
- `num_micro_batches`: The number of micro batches, equal to batch size/micro batches. When `num_micro_batches>1`, the grad function will apply `alpa.grad`, which adds the gradient accumulation mechanism to `jax.grad`. The default is `1`
- `num_gpt_layer` : The number of the gpt layer, other config parameters can be seen in `benchmark.py`.
- `dp`: The number of channel for data parallelism, an `int` from [1,2,4,……,gpus].
- `op`: The number of channel for operator parallelism, an `int` from [1,2,4,……,gpus].
- `reduce-scatter`:  If this is True, alpa will use **reduce-scatter** and  **all-gather** to replace **all-reduce**. It will achieve a sppedup in execute for reduce-scatter-friendly system, but burden the optimization time.
- `parallel mode`. It can be selected from `uniform` and `zero-3`.
- `shard`. Not using the ray cluster, default=`False`.
- `profile driven time`. Profile the execution time on the driver instead of the workers, default=`False`.
- `recomputation`. This switch determines whether recomputation is turned on, default=`False`. If recomputation is open, the memory cost will increase during the backward and the communication overhead (**all-reduce** during the recomputation) saves.

### Results

**Table 1**

| gpus | TFLOPs  | Peak Mem/GB | Execute time/s (Mean, Std) | Complie + optimize time/s |
|:----:|:------: |:-----------:|:--------------------------:|:-------------------------:|
| 1    | 40.67   | 7.053       | (80.787, 0.004)            | 50.19                     |
| 2    | 119.04  | 10.376      | (57.36,  0.080)            | 49.94                     |
| 4    | 240.76  | 8.575       | (48.337, 0.023)            | 57.52                     |
| 8    | 511.36  | 11.66       | (45.646, 0.010)            | 110.69                    |
| 16   | 1110.72 | 11.346      | (51.868, 0.019)            | 117.46                    |

**Table 2.1**

Details for Table 2.1:

```bash
--num-devices-per-host 8 --num_gpt_layer 4 --num_batch_size 32 --num_micro_batches 1 --reduce_scatter
```

| (dp,op) | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:-------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| (1,8)   | 462.56  | 6.478       | (0.565, 0.000)             | 28    | 28        | 0         | 0             | 0       | 5.69                      |
| (2,4)   | 538.88  | 5.098       | (0.485, 0.001)             | 33    | 29        | 1         | 3             | 0       | 7.98                      |
| (4,2)   | 571.20  | 5.449       | (0.457, 0.000)             | 33    | 29        | 1         | 3             | 0       | 7.96                      |
| (8,1)   | 587.44  | 6.924       | (0.445, 0.003)             | 4     | 1         | 1         | 2             | 0       | 4.00                      |

**Table 2.2** w/ recompute float16

Details for table 2.2 w/ recompute float16:

```bash
--num-devices-per-host 4 --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1 --recomputation
```

| (dp,op) | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:-------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| (1,4)   | 229.80  | 5.076       | (0.142, 0.000)             | 28    | 28        | 0         | 0             | 0       | 5.98                      |
| (2,2)   | 179.44  | 10.287      | (0.182, 0.000)             | 31    | 31        | 0         | 0             | 0       | 8.33                      |
| (4,1)   | 161.92  | 20.571      | (0.202, 0.001)             | 3     | 3         | 0         | 0             | 0       | 4.08                      |

**Table 2.2** w/o recompute float16

Details for table 2.2 w/o recompute float16

```bash
--num-devices-per-host 4 --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1
```

| (dp,op) | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:-------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| (1,4)   | 220.48  | 6.288       | (0.117, 0.000)             | 20    | 20        | 0         | 0             | 0       | 4.47                      |
| (2,2)   | 164.64  | 10.287      | (0.157, 0.000)             | 23    | 23        | 0         | 0             | 0       | 6.45                      |
| (4,1)   | 143.00  | 20.571      | (0.180, 0.001)             | 3     | 3         | 0         | 0             | 0       | 2.80                      |

**Table 2.2** w/ recompute float32

Details for table 2.2 w/ recompute float32:

```bash
--num-devices-per-host 4 --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1 --recomputation
```

| (dp,op) | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:-------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| (1,4)   | 48.12   | 5.485       | (0.679, 0.001)             | 29    | 27        | 2         | 0             | 0       | 5.62                      |
| (2,2)   | 43.96   | 11.429      | (0.743, 0.000)             | 30    | 30        | 0         | 0             | 0       | 8.59                      |
| (4,1)   | 43.20   | 22.857      | (0.756, 0.001)             | 2     | 2         | 0         | 0             | 0       | 3.59                      |

**Table 2.2** w/o recompute float32

Details for table 2.2 w/o recompute float32:

```bash
--num-devices-per-host 4 --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1
```

| (dp,op) | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:-------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| (1,4)   | 47.28   | 7.704       | (0.545, 0.000)             | 21    | 19        | 2         | 0             | 0       | 4.44                      |
| (2,2)   | 42.08   | 11.429      | (0.613, 0.000)             | 22    | 22        | 0         | 0             | 0       | 5.89                      |
| (4,1)   | 40.64   | 22.857      | (0.634, 0.001)             | 2     | 2         | 0         | 0             | 0       | 2.81                      |

**Table 2.3** w/o recompute float16

Details for table 2.3 w/o recompute float16:

```bash
--num-devices-per-host 4  --num_batch_size 4 --num_micro_batches 1 --dp 1 --op 4
```

| layers | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| 8      | 228.12  | 10.791      | (0.203, 0.000)             | 36    | 36        | 0         | 0             | 0       | 8.72                      |
| 12     | 228.60  | 15.340      | (0.293, 0.000)             | 52    | 52        | 0         | 0             | 0       | 13.14                     |
| 16     | 231.32  | 19.843      | (0.379, 0.000)             | 68    | 68        | 0         | 0             | 0       | 18.05                     |

**Table 2.3** w/ recompute float16

Details for table 2.3 w/o recompute float16:

```bash
--num-devices-per-host 4  --num_batch_size 4 --num_micro_batches 1 --dp 1 --op 4
```

| layers | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| 8      | 236.68  | 8.163       | (0.254, 0.000)             | 52    | 52        | 0         | 0             | 0       | 11.50                     |
| 12     | 237.52  | 11.290      | (0.369, 0.000)             | 76    | 76        | 0         | 0             | 0       | 18.95                     |
| 16     | 237.76  | 14.448      | (0.484, 0.001)             | 100   | 100       | 0         | 0             | 0       | 23.82                     |

**Table 2.3** w/o recompute float32

Details for table 2.3 w/o recompute float32:

```bash
--num-devices-per-host 4  --num_batch_size 4 --num_micro_batches 1 --dp 1 --op 4
```

| layers | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| 8      | 49.32   | 12.707      | (0.940, 0.000)             | 37    | 35        | 2         | 0             | 0       | 7.70                      |
| 12     | 50.40   | 17.710      | (1.330, 0.001)             | 53    | 51        | 2         | 0             | 0       | 13.46                     |
| 16     | 50.68   | 22.744      | (1.729, 0.001)             | 69    | 67        | 2         | 0             | 0       | 16.34                     |

**Table 2.3** w/ recompute float32

Details for table 2.3 w/ recompute float32:

```bash
--num-devices-per-host 4  --num_batch_size 4 --num_micro_batches 1 --dp 1 --op 4
```

| layers | TFLOPS  | Peak Mem/GB | Execute time/s (Mean, Std) | #comm | allreduce | allgather | reducescatter | all2all | Complie + optimize time/s |
|:------:|:-------:|:-----------:|:--------------------------:|:-----:|:---------:|:---------:|:-------------:|:-------:|:-------------------------:|
| 8      | 50.12   | 8.564       | (1.200, 0.001)             | 53    | 51        | 2         | 0             | 0       | 12.01                     |
| 12     | 50.72   | 11.644      | (1.727, 0.000)             | 77    | 75        | 2         | 0             | 0       | 16.54                     |
| 16     | 51.16   | 14.801      | (2.251, 0.002)             | 101   | 99        | 2         | 0             | 0       | 22.59                     |

Remark 1: When `Prefer_reduce_scatter=False` and `recomputation=False`, the tensor parallelism strategy generated by *alpa* is consistent with that of *megatron-lm*.

## Q&A

    Q1: Why the mean time for data parallelism (dp=8,op=1 in Table 2.1) is faster than tensor parallelism (dp=1,op=8 in Table 2.1)?

    A1: Because the communication volume of the former is 12 *hidden_size*hidden_size and that of the later is 4*batch size*hidden_size (2 all-reduce in the feedfoward and 2 in the backward).
        And the mean time in Table 2.2 (both w/ and w/o recomputation) supports this view. When we reduce batch size from 32 to 4, then the data parallelism (dp=4,op=1 in Table 2.2) is slower than tensor parallelism (dp=1,op=4 in Table 2.2).

    Q2: Why the TFLOPs are reduced to 1/4 of the precision of 16 bits when the precision is 32 bits?

    A2: Because it uses the tensor core technique, which boosts the TFLOPS.

## Reference

\[1\] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
