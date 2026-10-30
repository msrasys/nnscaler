#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from nnscaler.codegen.emit import CodeEmission, IRValue
from nnscaler.ir.cten import IRObject
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function import Dropout
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.device import _group_batches


def test_tensor_name():
    repr_expr = CodeEmission().tensor_name
    assert repr_expr(1, 'model.') == '1'
    assert repr_expr('1') == "'1'"

    assert repr_expr(IRObject('name', 111, 'value'), 'model.') == 'name_111'
    assert repr_expr(IRObject('name', 111, 'value').as_attr(), 'model.') == 'model.name_111'
    assert repr_expr((IRObject('name', 111, 'value').as_attr(),), 'model.') == '(model.name_111,)'

    assert repr_expr(slice(1, None, IRObject('name', 111, 'value').as_attr()), 'model.') == 'slice(1, None, model.name_111)'
    assert repr_expr({'a': 1, 'b': IRObject('name', 111, 'value')}, 'model.') == "{'a': 1, 'b': name_111}"
    assert repr_expr([1], 'model.') == '[1]'
    assert repr_expr((1,), 'model.') == '(1,)'

    assert repr_expr((1,...), ) == '(1, Ellipsis)'

    with pytest.raises(ValueError):
        from datetime import datetime
        repr_expr(datetime.now())



def test_emit_module_attr():
    dropout = Dropout(IRFullTensor([1024, 1024], requires_grad=True), p=0.5, training=IRValue('self.training'), signature='torch.nn.functional.dropout')
    code = FuncEmission().emit_fnode(dropout, runtime_devid=0, plan_ndevs=1, runtime_ndevs=1)
    print(code)
    assert 'training=self.training' in code[0]


def test_group_batches_tp_pp():
    # TP=2, PP=2, DP=4: each replica occupies four consecutive ranks.
    singletons = [[rank] for rank in range(16)]
    tp_groups = [[rank, rank + 1] for rank in range(0, 16, 2)]
    pp_groups = [[base + lane, base + lane + 2] for base in range(0, 16, 4) for lane in range(2)]
    dp_groups = [list(range(rank, 16, 4)) for rank in range(4)]
    # ZeRO-1 with zero_ngroups=2: split each DP group and reduce across matching shard owners.
    zero_subgroups = [ranks[start:start + 2] for ranks in dp_groups for start in (0, 2)]
    zero_crossgroups = [ranks[offset::2] for ranks in dp_groups for offset in range(2)]
    groups = (
        singletons + tp_groups + pp_groups + dp_groups + zero_subgroups + zero_crossgroups
        + [[1, 0], [2, 0], list(range(16))]
    )
    expected = [
        singletons,
        tp_groups,
        # P2P between matching TP lanes in adjacent pipeline stages.
        [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]],
        # ZeRO subgroups for optimizer-state sharding within each DP group.
        [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
        # Cross-ZeRO all-reduce between matching shard owners in the two subgroups.
        [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]],
        dp_groups,
    ]

    assert _group_batches(groups, 16) == expected
    assert _group_batches(groups[::-1], 16) == expected
