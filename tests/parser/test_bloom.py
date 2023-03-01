import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# model_name = "bigscience/bloom-7b1"
# model_path = "/home/quzha/bloom7b1"
model_name = "bigscience/bloom-560m"
model_path = "/home/quzha/bloom560m"
# model_name = "facebook/opt-66b"
# model_name = "facebook/opt-iml-30b"
# model_name = "facebook/optiml30b"
# model_name = "facebook/opt-iml-1.3b"
# model_name = "facebook/opt-13b"
# model_path = "/home/quzha/opt13b"

print("Loading model...") #device_map="balanced", 
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, cache_dir=model_path)#.cuda()
print(type(model), '; is nn.Module? ', isinstance(model, nn.Module))
print("Model's generation config which does not list default values: ", model.generation_config)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("Loading Done!")
prompt = "If I want to travel to a new city, I should plan my trip as follows:"
#input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
inputs = tokenizer(prompt, return_tensors="pt")#.to('cuda:0')

# Cube
# from cube.graph import parser
# ir_graph = parser.convert_model(model, input_shapes=[1, 17], save_content=False)

# model(input_ids, None, None, None, None, None, None, None, None, None)

print("concrete tracing model...")
from nni.common.concrete_trace_utils import concrete_trace
#traced_graph = concrete_trace(model, (input_ids, None, None, None, None, None, None, None, None, None), use_function_patch=True,
#        autowrap_leaf_class={torch.finfo: ((), False)})
#traced_graph = concrete_trace(model, inputs, use_function_patch=True,
traced_graph = concrete_trace(model, inputs, use_operator_patch=True,
        autowrap_leaf_class={torch.finfo: ((), False)})
# traced_graph.graph.print_tabular()
print("tracing model done.")

print("parsing fx graph to cube graph...")
from cube.graph.parser import FxModuleParser
# dummy_inputs = [inputs.input_ids, None, inputs.attention_mask, None, None, None, None, None, None, None, {}]
# FxModuleParser.parse(traced_graph, dummy_inputs)
FxModuleParser.parse(traced_graph, inputs)
print("parsing done.")

# AutoDist
# from autodist.apis import compile
# from cube.runtime.resource import EnvResource
# resource = EnvResource()
# graph = compile(ir_graph, resource)


# print(type(model), '; is nn.Module? ', isinstance(model, nn.Module))
# print("Model's generation config which does not list default values: ", model.generation_config)
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# print("Loading Done!")

# #prompt = "If you are a calculator, please tell me the results of 32 x 23 ="
# # prompt = "what is the english word that means little modification and starts with character "t"? the english word is"
# # prompt = "If I want to travel to USA, I need to apply for a"
# prompt = "If I want to travel to a new city, I should plan my trip as follows:"
# # prompt = "I look forward to"
# # prompt = "Today was an amazing day because"
# # prompt = "What is the color of a carrot?\nA:"


# # Some of the commonly adjusted parameters: max_new_tokens, num_beams, do_sample, num_return_sequences
# # https://huggingface.co/blog/how-to-generate
# # https://huggingface.co/docs/transformers/v4.26.1/en/generation_strategies#text-generation-strategies
# # Beam-search decoding
# generation_config_beam = GenerationConfig(
#     num_beams=4,
#     do_sample=False,
#     early_stopping=True,
#     decoder_start_token_id=0,
#     eos_token_id=model.config.eos_token_id,
#     pad_token=model.config.pad_token_id,
# )
# # Beam-search decoding without early stopping
# generation_config_beam_fixed_len = GenerationConfig(
#     num_beams=4,
#     do_sample=False,
#     early_stopping=False,
#     max_new_tokens=20,
#     decoder_start_token_id=0,
#     eos_token_id=model.config.eos_token_id,
#     pad_token=model.config.pad_token_id,
# )
# # Contrastive search
# generation_config_contrastive = GenerationConfig(
#     penalty_alpha=0.6,
#     top_k=4,
#     max_new_tokens=100,
# )

# print("Tokenizing prompt...")
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
# print("input_ids shape: ", input_ids.size())
# print("Generating sequence ids...")
# generated_ids = model.generate(input_ids, generation_config=generation_config_beam_fixed_len)
# print("Decoding sequence ids...")
# output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(output)
