# Alphafold2

## Model Config

We focus on evoformer like structures during training currently. Data type is *float16*.

**Evoformer Stack**

| Case             | s    | r    | cm   | cz   |
| ---------------- | ---- | ---- | ---- | ---- |
| initial training | 128  | 256  | 256  | 128  |
| 1st fine-tuning  | 512  | 256  | 256  | 128  |
| 2nd fine-tuning  | 512  | 384  | 256  | 128  |

**Extra Msa Stack**

| Case             | s    | r    | cm   | cz   |
| ---------------- | ---- | ---- | ---- | ---- |
| initial training | 1024 | 256  | 64   | 128  |
| 2.1 fine-tuning  | 1024 | 384  | 64   | 128  |
| 2.2 fine-tuning  | 5120 | 384  | 64   | 128  |

## Baselines

**Deepmind's plan**

data parallelism (each accelerator with exact 1 sample) + recompute. Since the parameter size is relatively small in Alphafold2, the latency can be approximately by single device execution. Hyperparameter setting is listed in the following table.

| Case            | evo_num | use_chunk |
| --------------- | ------- | --------- |
| Evoformer Stack | 48      | False     |
| Extra Msa Stack | 4       | True      |

**Dynamic Axial Parallelism (DAP)**

The end-to-end time is bounded by the computation. In other words, given input tensors with a fixed batch size, it is possible to reduce the time by introducing more devices (partition a operator into parallelizable sub-operators). Here are possible experiment dimensions.

| batch size | #gpus |
| ---------- | ----- |
| 1          | 2     |
| 2          | 4     |
| 4          | 8     |
| 8          | 16    |

**Table 1: Evoformer Stack & Training**

| Case             | batch size | #gpus | latency/ms | peak mem/MB |
| ---------------- | ---------- | ----- | ---------- | ----------- |
| initial training | 1          | 1     | 3521.98    | 4414        |
| initial training | 1          | 2     | 2430.38    | 2531        |
| initial training | 1          | 4     | 1497.77    | 1574        |
| initial training | 2          | 4     | 2485.53    | 2647        |
| 1st fine-tuning  | 1          | 1     | 7696.62    | 10729       |
| 1st fine-tuning  | 1          | 2     | 4663.32    | 5744        |
| 1st fine-tuning  | 1          | 4     | 2620.09    | 3211        |
| 1st fine-tuning  | 2          | 4     | 4717.36    | 5921        |
| 2nd fine-tuning  | 1          | 1     | 16632.06   | 17810       |
| 2nd fine-tuning  | 1          | 2     | 9377.98    | 9417        |
| 2nd fine-tuning  | 1          | 4     | 5099.72    | 5157        |
| 2nd fine-tuning  | 2          | 4     | 9422.99    | 9804        |

**Table 2: Extra Msa Stack & Training**

| Case             | batch size | #gpus | latency/ms | peak mem/MB |
| ---------------- | ---------- | ----- | ---------- | ----------- |
| initial training | 1          | 1     | x          | x           |
| initial training | 1          | 2     | x          | x           |
| initial training | 1          | 4     | x          | x           |
| initial training | 2          | 4     | x          | x           |
| 2.1 fine-tuning  | 1          | 1     | x          | x           |
| 2.1 fine-tuning  | 1          | 2     | x          | x           |
| 2.1 fine-tuning  | 1          | 4     | x          | x           |
| 2.1 fine-tuning  | 2          | 4     | x          | x           |
| 2.2 fine-tuning  | 1          | 1     | x          | x           |
| 2.2 fine-tuning  | 1          | 2     | x          | x           |
| 2.2 fine-tuning  | 1          | 4     | x          | x           |
| 2.2 fine-tuning  | 2          | 4     | x          | x           |

## End-to-end evaluation results (DAP vs Autodist)

### Model Config

Evoformer Stack
    - shape config
        - bs, s, r, cm, cz = 1, 128, 256, 256, 128
        - bs, s, r, cm, cz = 1, 512, 256, 256, 128
        - bs, s, r, cm, cz = 1, 512, 384, 256, 128
    - other config: dtype, use_chunk, is_train, is_extra = torch.float16, False, True, False

*note*: results organized in (estimate time/ms, execution time/ms, device mem/GB)

**Table 1: tensor parallelism(2gpu) w/o recompute**

evo_num = 4

| s, r          | DAP             | Autodist                 | compile time/s |
| ------------- | --------------- | ------------------------ | ---------------|
| 128, 256      | (139.15, 4.58)  | (127.13, 156.15, 5.35)   | 0.77           |
| 512, 256      | (293.11, 11.02) | (286.04, 307.54, 12.86)  | 0.77           |
| 512, 384      | (596.41, 20.91) | (568.72, 595.00, 24.44)  | 0.77           |

*note*: results organized in (estimate time/ms, execution time/ms, device mem/GB)

**Table 2: tensor parallelism(2gpu) w/ adaptive recompute**

evo_num = 48
memory constraint = 40GB

| s, r          | DAP             | Autodist                  | compile time/s |
| ------------- | --------------- | ------------------------- | ---------------|
| 128, 256      | (2250.27, 2.53) | (1690.71, 1915.13, 38.33) | 43.57          |
| 512, 256      | (4733.89, 5.74) | (4273.40, 4525.81, 39.06) | 45.39          |
| 512, 384      | (9673.10, 9.42) | (8911.85, 10042.22, 39.70)| 43.88          |

**Table 3: tensor parallelism(4gpu) w/ adaptive recompute**

evo_num = 48
memory constraint = 40GB

| s, r          | DAP             | Autodist                  | compile time/s |
| ------------- | --------------- | ------------------------- | ---------------|
| 128, 256      | (1874.73, 1.54) | (1083.93, 1400.29, 29.13) | 4650.48        |
| 512, 256      | (3350.06, 3.13) | (2388.69, 2965.40, 36.50) | 4483.49        |
| 512, 384      | (6724.48, 5.04) | (4932.62, 6450.42, 41.80) | 4427.15          |
