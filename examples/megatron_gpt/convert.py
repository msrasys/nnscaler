# 1. build model
from gpt_model import build_model, GeLUFunction
model = build_model()

# 2. register customized op
from cube.graph.parser.register import register
register('* h, h -> * h')(GeLUFunction.apply)

# 3. build semantic model
from cube import SemanticModel
smodel = SemanticModel(model)

# 4. set dummy input
import torch
batch_size = 16
seq_len = 128
dict_len = 50000
smodel.dummy_input={
    'src_tokens': torch.randint(0, dict_len, (batch_size, seq_len)),
    'target': torch.randint(0, dict_len, (batch_size, seq_len)),
    'ntokens': 128,
}

from cube.graph.function import IRObject
from cube.ir import IRFullTensor

src_tokens = IRFullTensor(shape=[batch_size, seq_len],
                          name='src_tokens',
                          dtype=torch.int).tosub()

target = IRFullTensor(shape=[batch_size, seq_len],
                      name='target',
                      dtype=torch.int).tosub()

ntokens = IRObject(name='ntokens')

# 5. convert to graph
from cube.graph.segment import IRSegment
from cube.program import Program

from torch.autograd.graph import saved_tensors_hooks

class no_save_tensor_hook(saved_tensors_hooks):
    def __init__(self):

        def pack(x):
            return None

        def unpack(x):
            raise RuntimeError("not expecting backward to be called on this tensor")

        super().__init__(pack, unpack)

Program().clear()

with no_save_tensor_hook():
    outputs = smodel(src_tokens, target, ntokens)
outputs[0].backward()

Program().finalize()
Program().set_input([src_tokens, target, ntokens])

if outputs is None:
    outputs = []
elif not (isinstance(outputs, tuple) or isinstance(outputs, list)):
    outputs = [outputs]
Program().set_output(outputs)

graph = Program().get_graph()

# 6. save graph
graph.dump('megatron_gpt2.cube')

for node in graph._nodes:
    if isinstance(node, IRSegment):
        print(node.debug_tensor_map_str())
