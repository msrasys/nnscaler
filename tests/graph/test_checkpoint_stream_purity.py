# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict

import pytest
import torch

from nnscaler import ModuleStreamRule
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.schedule.module_streams import checkpoint_stream
from nnscaler.ir.cten import IRObject
from nnscaler.ir.tensor import IRFullTensor


def operation(signature, inputs, output):
    node = IRPyFunc(signature, inputs, [output])
    node.module_stack = OrderedDict(block=torch.nn.Module)
    return node


def update(target, key='key', value=1):
    return operation('nnscaler.runtime.function.setitem', [target, key, value],
                     IRObject('updated', value={}))


RULE = ModuleStreamRule('block', stream='recompute', recompute_only=True)


def test_fresh_checkpoint_kwargs_can_be_updated_repeatedly():
    first = update({'causal': True}, 'max_length', IRObject('length', value=8192))
    second = update(first.output(0), 'deterministic', True)
    assert checkpoint_stream([first, second], [RULE]) == 'recompute'


@pytest.mark.parametrize('constant', [False, True])
def test_external_dictionary_mutation_is_rejected_even_when_constant(constant):
    node = update(IRObject('caller_state', value={}, is_constant=constant))
    with pytest.raises(ValueError, match='pure operations.*setitem'):
        checkpoint_stream([node], [RULE])


def test_nested_external_dictionary_alias_is_not_owned():
    external = IRObject('caller_state', value={})
    fresh = update({'nested': external})
    alias = operation('nnscaler.runtime.function.getitem', [fresh.output(0), 'nested'],
                      IRObject('nested', value={}))
    mutation = update(alias.output(0))
    with pytest.raises(ValueError, match='pure operations.*setitem'):
        checkpoint_stream([fresh, alias, mutation], [RULE])


def test_dictionary_created_outside_checkpoint_is_not_owned():
    first = update({})
    second = update(first.output(0))
    with pytest.raises(ValueError, match='pure operations.*setitem'):
        checkpoint_stream([second], [RULE])


def test_tensor_setitem_is_still_rejected():
    node = update(IRFullTensor((4,), dtype=torch.float32), 0)
    with pytest.raises(ValueError, match='pure operations.*setitem'):
        checkpoint_stream([node], [RULE])
