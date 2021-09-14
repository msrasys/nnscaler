"""
Convert PyTorch nn.Module to our IRGraph
"""
import torch

class Node:

    def __init__(self, name, type_name):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            name (str): the var name of the module
            type_name: the type name of the module

        Example:
            init code:
                self.linear1 = torch.nn.Linear(input_feats, output_feats)
            forward code:
                output = self.linear1(input)
            =>
                name = linear1; type_name = torch.nn.Linear
        """
        pass


class IRGraph:

    def __init__(self, module, example_inputs=None):

        self.module = module
        self.script_module = torch.jit.script(module)
        # model info
        self.module_name = None


    def _convert(self):
        self.module_name = self.script_module.original_name
