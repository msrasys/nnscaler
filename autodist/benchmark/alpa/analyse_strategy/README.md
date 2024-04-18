# Analyse alpa strategy

In this part, we write the gen_str.py to generate the partition strategy from the log in Alpa. The best spmd results with 760M and 1.3B are in **strategy.zip**.

## Usage

```bash
python gen_str.sh
```

The default load_file is log.txt and the save_file is test.txt, you can  specific the load_file and save_file by adding **--load_file** and **--save_file**. If you want to see more information about the Alpa partition, you can add **--whole_strategy --detailed_partition_strs**

## Comparsion with Autodist

<!-- Todo: Peiran -->
