from typing import Dict

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import CubeComplexToQKV
from cube.graph.operator.function import CubeComplexTrilMask
from cube.graph.operator.function import CubeComplexAttnView
from cube.graph.operator.function import CubeComplexSelfAttention
from cube.graph.operator.function import CubeComplexFeedForward


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


class CubeSelfAttentionHeadParallel(GenericDistAlgo):
    """
    Multi-Head Self-Attention.

    L: sequence length
    N: batch size
    E: embedding size
    
    Inputs:
        hidden_state: [L, N, E]
        w_qkv       : [3 * num_head * dim_head, E]
        w_out       : [E, E]
        num_head: int
        dim_head: int
        dropout_p: float

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, node: CubeComplexSelfAttention):
        if not isinstance(node, CubeComplexSelfAttention):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.num_head = node.kwargs['num_head']

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.num_head % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: CubeComplexSelfAttention, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_state = node.inputs(0)
        w_qkv = node.inputs(1)
        w_out = node.inputs(2)
        num_head = node.kwargs['num_head']
        dim_head = node.kwargs['dim_head']
        dropout_p = node.kwargs['dropout_p']
        out = node.outputs(0)

        
        w_qkvs = split_axis(w_qkv, 0, self.chunk_num)
        w_outs = split_axis(w_out, 1, self.chunk_num)
        ous = split_value(out, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [
                hidden_state, w_qkvs[idx], w_outs[idx],
                num_head // self.chunk_num, dim_head, dropout_p
            ]
            node = CubeComplexSelfAttention(
                signature = 'cube.runtime.function.complex.self_attn',
                inputs = inputs,
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes


class CubeSelfAttentionDataParallel(GenericDistAlgo):
    """
    Multi-Head Self-Attention.

    L: sequence length
    N: batch size
    E: embedding size
    
    Inputs:
        hidden_state: [L, N, E]
        w_qkv       : [3 * num_head * dim_head, E]
        w_out       : [E, E]
        num_head: int
        dim_head: int
        dropout_p: float

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, node: CubeComplexSelfAttention):
        if not isinstance(node, CubeComplexSelfAttention):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.bs = node.inputs(0).shape[1]

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.bs % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: CubeComplexSelfAttention, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_state = node.inputs(0)
        w_qkv = node.inputs(1)
        w_out = node.inputs(2)
        num_head = node.kwargs['num_head']
        dim_head = node.kwargs['dim_head']
        dropout_p = node.kwargs['dropout_p']
        out = node.outputs(0)

        ins = split_axis(hidden_state, 1, self.chunk_num)
        ous = split_axis(out, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [
                ins[idx], w_qkv, w_out,
                num_head, dim_head, dropout_p
            ]
            node = CubeComplexSelfAttention(
                signature = 'cube.runtime.function.complex.self_attn',
                inputs = inputs,
            )
            node.set_output(0, ous[idx])
            nodes.append(node)
        return nodes


class CubeFeedForwardTensorParallel(GenericDistAlgo):
    """
    FeedForward

    Inputs:
        hidden_state: [L, N, E]
        w_proj1: [4 * E, E]
        w_bias1: [4 * E,]
        w_porj2: [E, 4 * E]
        w_bias2: [E,]

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, node: CubeComplexFeedForward):
        if not isinstance(node, CubeComplexFeedForward):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.embed_size = node.inputs(1).shape[0]

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.embed_size % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: CubeComplexFeedForward, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_state = node.inputs(0)
        w_proj1 = node.inputs(1)
        w_bias1 = node.inputs(2)
        w_proj2 = node.inputs(3)
        w_bias2 = node.inputs(4)

        out = node.outputs(0)

        w_proj1s = split_axis(w_proj1, 0, self.chunk_num)
        w_bias1s = split_axis(w_bias1, 0, self.chunk_num)
        w_proj2s = split_axis(w_proj2, 1, self.chunk_num)
        w_bias2s = split_value(w_bias2, self.chunk_num)

        outs = split_value(out, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [
                hidden_state, 
                w_proj1s[idx], w_bias1s[idx],
                w_proj2s[idx], w_bias2s[idx]
            ]
            node = CubeComplexFeedForward(
                signature = 'cube.runtime.function.complex.feedforward',
                inputs = inputs,
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes


class CubeFeedForwardDataParallel(GenericDistAlgo):
    """
    FeedForward

    Inputs:
        hidden_state: [L, N, E]
        w_proj1: [4 * E, E]
        w_bias1: [4 * E,]
        w_porj2: [E, 4 * E]
        w_bias2: [E,]

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, node: CubeComplexFeedForward):
        if not isinstance(node, CubeComplexFeedForward):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision
        self.bs = node.inputs(0).shape[1]

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        if chunk_num > 0 and self.bs % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: CubeComplexFeedForward, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        hidden_state = node.inputs(0)
        w_proj1 = node.inputs(1)
        w_bias1 = node.inputs(2)
        w_proj2 = node.inputs(3)
        w_bias2 = node.inputs(4)
        out = node.outputs(0)

        ins = split_axis(hidden_state, 1, self.chunk_num)
        outs = split_axis(out, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            inputs = [
                ins[idx], 
                w_proj1, w_bias1,
                w_proj2, w_bias2,
            ]
            node = CubeComplexFeedForward(
                signature = 'cube.runtime.function.complex.feedforward',
                inputs = inputs,
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes
