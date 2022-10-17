# Introduction

Benchmark different schedule plans of Alphafold2 based on MagicCube.

# Results

## Training

### Evoformer Stack

**s, r = 128, 256**

| device num | policy | peak mem (MB) | activation mem (MB) | time (ms) |
|:-----------|:-------|:--------------|:--------------------|:----------|
| 1          | small  | 8119          | 7462                | 3656.61   |
| 1          | large  | 4414          | 2070                | 3635.38   |
| 2          | small  | 4351          | 4014                | 2539.56   |
| 2          | large  | 2531          | 1318                | 2506.10   |

**s, r = 512, 256**

| device num | policy | peak mem (MB) | activation mem (MB) | time (ms) |
|:-----------|:-------|:--------------|:--------------------|:----------|
| 1          | small  | 18952         | 14471               | 7949.96   |
| 1          | large  | 10729         | 4423                | 7914.68   |
| 2          | small  | 9839          | 7567                | 4839.22   |
| 2          | large  | 5744          | 2543                | 4793.78   |

**s, r = 512, 384**

| device num | policy | peak mem (MB) | activation mem (MB) | time (ms) |
|:-----------|:-------|:--------------|:--------------------|:----------|
| 1          | small  | OOM           | OOM                 | OOM       |
| 1          | large  | 17810         | 7104                | 17063.41  |
| 2          | small  | 16230         | 12847               | 9659.66   |
| 2          | large  | 9416          | 3870                | 9629.48   |

### Extra Msa Stack

**device num = 1**

| Config           | peak mem (MB) | activation mem (MB) | time (ms) |
|:-----------------|:--------------|:--------------------|:----------|
| s, r = 1024, 256 | 3236          | 1166                | 2306      |
| s, r = 1024, 384 | 6976          | 1805                | 3749.43   |
| s, r = 5120, 384 | 16168         | 8210                | 58393.83  |

## Inference

### T1044

**s, r = 128, 2048**

| device num | policy      | peak mem (MB) | time (ms) |
|:-----------|:------------|:--------------|:----------|
| 1          | direct      | OOM           | OOM       |
| 1          | chunk       | 23374         | 339742.02 |
| 2          | DAP         | OOM           | OOM       |
| 2          | DAP + chunk | 13006         | 192577.34 |
| 4          | DAP         | OOM           | OOM       |
| 4          | DAP + chunk | 9358          | 101993    |