from graph_manipulation import *



# general transformations
'''
Op := I -> Op (pre-identity)
Op := Op -> I (post-identity)
Op := Op, Op (replicate)
'''

# batch transformation (due to operator sample-wise)
'''
DataLoader
    split (output)activation

OperatorForward
    split (input)activation
    replica (input)weight
    split (output)activation*

OperatorBackward-(activation's gradient)
    split (input)d-activation*
    replica (input)weight
    split (output)d-activation

OperatorBackward-(weight's gradient)
    split (input)d-activation*
    split (input)activation
    value-split (to-reduce) (output)d-weight
'''

# non-batch transformation (operator semantic aware)
'''
elementwise operators (including optimizers)
    arbitrary same split on inputs and outputs
    
MatMul [M, K]*[K, N] => [M, N]
    1. split M or N (e.g., cases with M or N as batch-dim)
    2. split reducing dim K: [M, K/2]*[K/2, N] => value-split [M, N]
    
Conv2D
    1. split (input) image H, W => halo exchange then local Conv2D, split (output) image 
    2. split (input) filter out-channel-dim => Conv2D on replicated image with partial filter, value-split (output) image

(more cases) ...
'''


def trans(node, )->Node:
    pass