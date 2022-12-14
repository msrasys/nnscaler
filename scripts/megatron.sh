#!/bin/bash --login

torchrun --nproc_per_node=2  --nnodes=1    \
        examples/nlp/gpt/train.py --policy=PASMegatronWSRTP --lrw --fp16 | tee -a LogForMegatronRecompute.txt