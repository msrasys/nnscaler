
from typing import List

from cube.graph.operator.operator import IRFwOperation
from cube.ir.cten import IRTensor

from einops.einops import _apply_recipe

import torch

class IRScriptEinOps(IRFwOperation):

    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        signature = 'cube.runtime.function.einops'
        assert len(inputs) == 1, "Expected only input"
        assert len(kwargs) == 2, "Expected 2 kwargs: recipe_str, reduction_type"
        super().__init__(name, signature, 1, 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update(kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.inputs(0).shape) == 0:
            return False

        recipe_str = self.kwargs['recipe_str']
        import pickle
        recipe = pickle.loads(recipe_str)

        reduction_type = self.kwargs['reduction_type']
        tmp_tensor = torch.zeros(self.inputs(0).shape)
        tmp_output = _apply_recipe(recipe, tmp_tensor, reduction_type)
        self.outputs(0).shape = list(tmp_output.shape)
        return True

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        recipe_str = self.kwargs['recipe_str']
        reduction_type = self.kwargs['reduction_type']
        op = IRScriptEinOps(self.signature, inputs, self.name,
                            recipe_str=recipe_str, reduction_type=reduction_type)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op

