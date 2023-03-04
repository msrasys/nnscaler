import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "bigscience/bloom-560m"
model_path = "/home/quzha/bloom560m"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, cache_dir=model_path)
print(type(model), '; is nn.Module? ', isinstance(model, nn.Module))
print("Model's generation config which does not list default values: ", model.generation_config)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("Loading Done!")
prompt = "If I want to travel to a new city, I should plan my trip as follows:"
#input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
inputs = tokenizer(prompt, return_tensors="pt")

# Cube
# from cube.graph import parser
# ir_graph = parser.convert_model(model, input_shapes=[1, 17], save_content=False)

print("concrete tracing model...")
from nni.common.concrete_trace_utils import concrete_trace
traced_graph = concrete_trace(model, inputs, use_operator_patch=True,
        autowrap_leaf_class={torch.finfo: ((), False)})
print("tracing model done.")

print("parsing fx graph to cube graph...")
from cube.graph.parser import FxModuleParser
FxModuleParser.parse(traced_graph, dummy_inputs=inputs)
print("parsing done.")
