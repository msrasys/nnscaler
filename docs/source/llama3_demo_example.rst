###############
Llama 3 Example
###############

This is an example demostrating how to train Llama 3 8B with nnScaler's :doc:`trainer <trainer>`.

The example contains one single script, ``train.py``.

***********
Get Started
***********

Installation
============

0. Get your `Hugging Face token <https://huggingface.co/docs/hub/security-tokens>`_ to access Llama 3 model ::

    export HF_TOKEN=...

1. Install nnScaler ::

    pip install nnscaler

2. Clone nnScaler repo to get the example ::

    git clone --recursive https://msrasrg.visualstudio.com/SuperScaler/_git/MagicCube
    cd MagicCube/examples/llama3_demo

3. Install Llama 3 dependencies ::

    pip install -r requirements.txt

   Note: The requirements file has pinned ``torch``, ``transformers``, and ``datasets`` versions
   to ensure their compatibility with each others.

4. Prepare dataset ::

    # To run Llama 3 8B:
    python train.py --prepare_data

    # Or to run a shrinked Llama for debug:
    python train.py --prepare_data --mini

Train a Mini-model
==================

This examples requires 8 × 80GB GPU memory to train a full 8B model.
If your have adequate GPUs, you can skip to :ref:`the next section <Finetune Llama 3 8B>`.

Alternatively, you can start from a smaller model for verification: ::

    python train.py --prepare_data --mini
    torchrun --nproc_per_node=2 train.py --mini

This will resize Llama 3 to 4 hidden layers and reduce max sequence length to 4K.
We have tested it with 2 × 48GB memory.

If the model is still too large, you can shrink it further: ::

    python train.py --prepare_data --max_seq_len=1024
    torchrun --nproc_per_node=2 train.py --max_seq_len=1024 --num_hidden_layers=2 --from_scratch

With the default mini config (4 layers, 4K sequence length), the loss curve will be like following:

.. image:: ./images/llama3-curves-mini.png

Finetune Llama 3 8B
===================

Use the following commands to finetune `Meta-Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`: ::

    python train.py --prepare_data
    torchrun --nproc_per_node=8 train.py

.. image:: ./images/llama3-curves-8b.png

********
Resuming
********

The example will save a checkpoint on finish.
To continue training from the checkpoint: ::

    torchrun --nproc_per_node=8 train.py --resume_from=last --max_train_steps=2000

Please note that the checkpoint is sharded according to the distribution strategy.
If you want to resume a checkpoint in a different environment, you need to merge it into an ordinal checkpoint first: ::

    python train.py --merge_checkpoint=./checkpoints/last
    torchrun --nproc_per_node=8 train.py --resume_from=./checkpoints/merged.ckpt --max_train_steps=3000
