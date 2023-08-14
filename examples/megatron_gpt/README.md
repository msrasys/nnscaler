# Train Megatron-GPT with Cube

This example demonstrates how to train a GPT model from Megatron-ML using Cube. The process consists of three main steps:
1. Instantiate the model and trace it to an fx.Graph. Then, convert the fx.Graph to a Cube graph.
2. Compile the Cube graph into Python code by **data parallel** on 2 devices.
3. Train the GPT model using the compiled code in Fairseq.

At first, clone the Megatron-LM and checkpoint to the devcube branch, gpt model in this branch is a single device version.

```console
git clone https://msrasrg.visualstudio.com/SuperScaler/_git/Megatron-LM
cd Megatron-LM
git checkout devcube
# cd MagicCube dir
cd ../MagicCube/examples/megatron_gpt
# download gpt2-vocab.json and gpt2-merges.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

The following three commands correspond to the above three steps:

```console
bash run.sh trace
bash run.sh compile
bash run.sh run
```
