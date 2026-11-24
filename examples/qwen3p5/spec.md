## Target for this Dir
I wanna make an organized and complete dir here for qwen3.5's almost all models.
We can use human-defined or agent-defined parallelization strategy rather than searched by Autodist in nnScaler. So that we can mainly focus on our search.
The core is to implemente an agent-based auto adaptation for various frontier models, the main part is auto adaptating the novel operators.
And our ultimate goal is to produce FX and IR graph for consequent verfication process, so the training iteration is not the most important.

### The first version of auto loop of auto adapting novel features of frontier models
1. Graph Capture and 1-GPU golden parity establishment.
2. Default 4-GPU plan execution
3. Trace Diagnosis and Equivalent Algebraic Rewrite
4. Custom-op Registration
5. Rerun and jump to step.3 for another round
But since it is under the condition that we will still use autodist, which might be not so important. So we might need refine the loop later.

### The Adaptation Stages
The adaptation work includes 3 stages:
1. Dense Text (0.8B-27B),
2. MoE Text (35B-A3B to 397B-A17B)
3. Multimodal (additional support )

## Regulation for this Dir
1. Reflecting Qwen3p5_0p8B's dir, I think we might not need truly run models' training iterations.
2. Kick off Autodist and actual training iteration to avoid OOM, it should only capture each model's operator flow graph and to generate FX graph and IR graph under given GPUs.

### The obvious MUST DOs:
1. Since much more models are too large to be executed on 4 or even 16 GPUs, we must separate compilation configurations from actual hardware support and we might need lossen the memory budget as well?

2. We need to support statically assig parallelization strategy in the nnScaler framework, like use a json file to explicitly define each operators strategy (I think Agent can be good at this?) And we should be able to dump organizedly these artifacts(I think we are able already?)

3. Previous I clone the wrong outdated repo, newer versions should all base on this repo https://github.com/msrasys/nnscaler.git. I have cloned the newest repo at /home/v-xuasong/nnscaler, the older version has been changed to the /home/v-xuasong/nnscaler_outdated

4. 



