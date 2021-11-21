from typing import Dict

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import CubeComplexToQKV
from cube.graph.operator.function import CubeComplexTrilMask
from cube.graph.operator.function import CubeComplexAttnView


_kWaitDecision = None


class CubeToQKVDataParallel(GenericDistAlgo):
    """
    Inputs:
        hidden_state: [L, N, E]
        weight: [3 * (num_head * dim_head), E]
        num_head: int

    where L = sequence length, N = batch size, E = num_head * dim_head

    Returns:
        Q: [L, N * num_head, dim_head]
        K: [L, N * num_head, dim_head]
        V: [L, N * num_head, dim_head]
    """
    def __init__(self, node: CubeComplexToQKV):

        if not isinstance(node, CubeComplexToQKV):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[1]

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.bs % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_size, weight = node.inputs()
        q, k, v = node.outputs()

        ins = split_axis(hidden_size, 1, self.chunk_num)
        qs = split_axis(q, 1, self.chunk_num)
        ks = split_axis(k, 1, self.chunk_num)
        vs = split_axis(v, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [ins[idx], weight, self.num_head]
            node = CubeComplexToQKV(
                signature = 'cube.runtime.function.complex.toqkv',
                inputs = inputs,
                name = 'toqkv'
            )
            node.set_output(0, qs[idx])
            node.set_output(1, ks[idx])
            node.set_output(2, vs[idx])
            nodes.append(node)
        return nodes


class CubeToQKVHeadParallel(GenericDistAlgo):
    """
    Inputs:
        hidden_state: [L, N, E] (seqlen, batch size, num_head * dim_head)
        weight: [E * 3, E]
        num_head: int

    Returns:
        Q: [L, N * num_head, dim_head]
        K: [L, N * num_head, dim_head]
        V: [L, N * num_head, dim_head]
    """
    def __init__(self, node: CubeComplexToQKV):

        if not isinstance(node, CubeComplexToQKV):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[1]

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.num_head % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_state, weight = node.inputs()
        q, k, v = node.outputs()

        ws = split_axis(weight, 0, self.chunk_num)
        qs = split_axis(q, 1, self.chunk_num)
        ks = split_axis(k, 1, self.chunk_num)
        vs = split_axis(v, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [hidden_state, ws[idx], self.num_head // self.chunk_num]
            node = CubeComplexToQKV(
                signature = 'cube.runtime.function.complex.toqkv',
                inputs = inputs,
                name = 'toqkv'
            )
            node.set_output(0, qs[idx])
            node.set_output(1, ks[idx])
            node.set_output(2, vs[idx])
            nodes.append(node)
        return nodes


class CubeTrilMaskDataParallel(GenericDistAlgo):
    """
    Inputs:
        input: [N * num_head, L, L]
        num_head: int
    
    Returns:
        output: [N * num_head, L, L]
    """
    def __init__(self, node: CubeComplexTrilMask):

        if not isinstance(node, CubeComplexTrilMask):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[0] // self.num_head

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.bs % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_size = node.inputs(0)
        masked_out = node.outputs(0)

        ins = split_axis(hidden_size, 0, self.chunk_num)
        ous = split_axis(masked_out, 0, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [ins[idx], self.num_head]
            node = CubeComplexTrilMask(
                signature = 'cube.runtime.function.complex.tril_mask',
                inputs = inputs,
                name = 'tril_mask'
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes


class CubeTrilMaskHeadParallel(GenericDistAlgo):
    """
    Inputs:
        input: [N * num_head, L, L]
        num_head: int
    
    Returns:
        output: [N * num_head, L, L]
    """
    def __init__(self, node: CubeComplexTrilMask):

        if not isinstance(node, CubeComplexTrilMask):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[0] // self.num_head

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.num_head % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_size = node.inputs(0)
        masked_out = node.outputs(0)

        ins = split_axis(hidden_size, 0, self.chunk_num)
        ous = split_axis(masked_out, 0, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [ins[idx], self.num_head // self.chunk_num]
            node = CubeComplexTrilMask(
                signature = 'cube.runtime.function.complex.tril_mask',
                inputs = inputs,
                name = 'tril_mask'
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes


class CubeAttnViewDataParallel(GenericDistAlgo):
    """
    Inputs:
        [N * num_head, L, dim_head]

    Outputs:
        [L, N, num_head * dim_head]
    """
    def __init__(self, node: CubeComplexAttnView):
        if not isinstance(node, CubeComplexAttnView):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[0] // self.num_head

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.bs % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        attn = node.inputs(0)
        out = node.outputs(0)

        ins = split_axis(attn, 0, self.chunk_num)
        ous = split_axis(out, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [ins[idx], self.num_head]
            node = CubeComplexAttnView(
                signature = 'cube.runtime.function.complex.attn_view',
                inputs = inputs,
                name = 'attn_view'
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes


class CubeAttnViewHeadParallel(GenericDistAlgo):
    """
    Inputs:
        [N * num_head, L, dim_head]

    Outputs:
        [L, N, num_head * dim_head]
    """
    def __init__(self, node: CubeComplexAttnView):
        if not isinstance(node, CubeComplexAttnView):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']
        self.bs = node.inputs(0).shape[0] // self.num_head

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.num_head % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        attn = node.inputs(0)
        out = node.outputs(0)

        ins = split_axis(attn, 0, self.chunk_num)
        ous = split_axis(out, 2, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [ins[idx], self.num_head // self.chunk_num]
            node = CubeComplexAttnView(
                signature = 'cube.runtime.function.complex.attn_view',
                inputs = inputs,
                name = 'attn_view'
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes
