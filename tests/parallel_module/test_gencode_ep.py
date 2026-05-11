from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import nnscaler
from nnscaler.graph.function import DimopSplit, TransformRule
from nnscaler.graph.schedule.schedplan import SchedulePlan, StreamConfig, StreamContext
from nnscaler.graph.segment import IRSegment
from nnscaler.parallel import IRGraph
from nnscaler.policies import get_pas_ops

from .test_gencode import print_gencode, _gencode_contains, replace_all_device_with


@nnscaler.register_op('l m^ -> ?, ?,?, ?')
def step1_preprocess(routing_map, num_local_experts, num_experts, ep_size):
    """Gather per-expert token counts across EP ranks and compute split sizes.
    """
    # shape: num_tokens_per_local_expert: self.num_local_experts
    num_local_tokens_per_expert = routing_map.sum(dim=0).long()

    # preprocess
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1)
    num_global_tokens_per_expert = torch.zeros(ep_size, num_experts, dtype=num_local_tokens_per_expert.dtype, device=num_local_tokens_per_expert.device)
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, :num_local_experts].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(axis=1)
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

    if num_local_experts > 1:
        num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts)

    return (
        num_tokens_per_local_expert,
        num_global_tokens_per_local_expert,
        input_splits.tolist(),
        output_splits.tolist(),
    )

@nnscaler.register_op('l h^, l m^, l m^ -> /, /, /')
def step2_permute(hidden_states, routing_map, routing_probs, topk):
    """Permute tokens so same-expert tokens are contiguous.
    """
    return hidden_states.repeat_interleave(topk, dim=0).clone(), \
        routing_map.int().topk(topk, dim=1)[0].flatten().clone(), \
        routing_probs.topk(topk, dim=1)[0].flatten().clone()


@nnscaler.register_op('/, / -> /, /')
def step3_dispatch(permuted_local_input_tokens, permuted_probs, input_splits, output_splits):
    """All-to-all to send tokens to expert-owning ranks.
    """
    return permuted_local_input_tokens.clone(), \
        permuted_probs.clone()


@nnscaler.register_op('/, /, /, / -> /, /')
def step4_dispatch_postprocess(global_input_tokens, global_probs, num_global_tokens_per_local_expert, num_experts, num_local_experts):
    """Re-sort received tokens so each local expert's tokens are contiguous.
    """
    return global_input_tokens.clone(), \
        global_probs.clone()


@nnscaler.register_op('/, E t^ h^, E h^ d^, /, / -> /')
def step5_compute(global_input_tokens, w13, w2, num_tokens_per_local_expert, global_probs):
    """Run fused FFN on tokens grouped by local expert.
    """
    return global_input_tokens.clone()


@nnscaler.register_op('/, /, /, / -> /')
def step6_compute_postprocess(expert_outs, num_global_tokens_per_local_expert, num_experts, num_local_experts):
    """Re-sort expert outputs back to per-rank order for all-to-all return.
    """
    return expert_outs.clone()


@nnscaler.register_op('/, /, / -> /')
def step7_combine(expert_outs, input_splits, output_splits):
    """All-to-all to return expert outputs back to source ranks.
    """
    return expert_outs.clone()


@nnscaler.register_op('l h^, l m^, /, / -> l h^')
def step8_postprocess(hidden_state, routing_map, permuted_local_input_tokens, reversed_local_input_permutation_mapping):
    """Unpermute tokens back to original order.
    """
    return hidden_state.clone()


@nnscaler.register_op('l e^ -> l e^, l e^, l e^')
def topk_routing(logits, top_k):
    gate_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=top_k, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs.to(logits.dtype))
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map, gate_scores


def build_ep_transform_rule():
    itransform = [
        DimopSplit.D(0),
        DimopSplit.D(0),
        DimopSplit.D(0),
        DimopSplit.D(0),
        DimopSplit.D(0),
    ]

    otransform = [
        DimopSplit.D(0),
    ]

    return TransformRule(itransform, otransform)


# @nnscaler.register_op(f'l h^, l m^, l m^, E t^ h^, E h^ d^ -> l h^', transform_rules=(build_ep_transform_rule(),))
def moe_forward(x, routing_map, routing_probs, w13, w2, top_k, num_local_experts, num_experts, ep_size):
    """
    It is really tricky to break a function into multiple segments
    and register them as separate ops.
    We need to check the generated code to make sure everything is correct.

    We need to be very careful to avoid generating additional adapters or missing necessary adapters
    Here are some experiences we learned when doing this:
    1. For sub function annotations, only annotate the inputs and outputs of the whole function.
       for example, inputs are used in
        `step1_preprocess`, `step2_permute`, `step5_compute`, `step8_postprocess`
        and output is returned from `step8_postprocess`
       All intermediate tensors should be annotated as '/' to make sure no grad reduction adapter is generated.

    2. `y=x.view(*x.shape)` is important if x is splitted in different ways.
       Here x is used in `self.gate`, which is outside of `moe_forward`.
       if `self.gate` doesn't split x, but `moe_forward` splits x,
       then `multiref` will be inserted,
       and `step2_permute` and `step8_postprocess` will have different version of x,
       which will cause additional `split_allgather` to be generated for x.

    """
    y = x.view(*x.shape) # shallow copy to break the autograd graph for x

    # shape: num_tokens_per_local_expert: num_local_experts
    # num_global_tokens_per_local_expert: (ep_size, num_local_experts)
    (
        num_tokens_per_local_expert,
        num_global_tokens_per_local_expert,
        input_splits,
        output_splits,
    ) = step1_preprocess(
        routing_map,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        ep_size=ep_size
    )

    # shape:
    # permuted_local_input_tokens: (l topk) h^,
    # permuted_probs: (l topk),
    # reversed_local_input_permutation_mapping: (l topk)
    (
        permuted_local_input_tokens,
        permuted_probs,
        reversed_local_input_permutation_mapping
    ) = step2_permute(y, routing_map, routing_probs, top_k)

    # shape
    # global_input_tokens: (l topk) h^,
    # global_probs: (l topk)
    (
        global_input_tokens,
        global_probs
    ) = step3_dispatch(permuted_local_input_tokens, permuted_probs, input_splits, output_splits)

    # shape: this shape is fake.
    # we don't know the exact global shape
    # global_input_tokens: (l topk) h^,
    # global_probs: (l topk)
    (
        global_input_tokens,
        global_probs,
    ) = step4_dispatch_postprocess(global_input_tokens, global_probs, num_global_tokens_per_local_expert, num_experts, 2)

    # shape
    # global_input_tokens: (l topk) h^,
    global_input_tokens = step5_compute(global_input_tokens, w13, w2, num_tokens_per_local_expert, global_probs)

    # shape:
    # expert_outs: (l topk) h^
    expert_outs = step6_compute_postprocess(global_input_tokens, num_global_tokens_per_local_expert, num_experts, 2)

    # shape:
    # expert_outs: (l topk) h^
    expert_outs = step7_combine(expert_outs, input_splits, output_splits)

    # shape:
    # output: l h^
    output = step8_postprocess(y, routing_map, expert_outs, reversed_local_input_permutation_mapping)

    return output


class MoE(nn.Module):
    def __init__(
        self,
        embed_dim,
        moe_ffn_dim,
        num_experts,
        top_k,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.moe_ffn_dim = moe_ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = 2
        self.num_local_experts = num_experts // self.ep_size

        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False, dtype=torch.float32)
        self.w13 = torch.nn.Parameter(torch.empty((self.num_experts, 2 * self.moe_ffn_dim, self.embed_dim)))
        self.w2 = torch.nn.Parameter(torch.empty((self.num_experts, self.embed_dim, self.moe_ffn_dim)))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5), mode='fan_in')
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w13[i], a=math.sqrt(5), mode='fan_in')
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5), mode='fan_out')

    def forward(self, x):
        # shape: x: l h^, gate: h^ m^ -> l m^
        logits = self.gate(x.float())
        # shape: routing_probs: l m^, routing_map: l m^, gate_scores: l m^
        routing_probs, routing_map, gate_scores = topk_routing(logits, self.top_k)

        return moe_forward(x, routing_map, routing_probs, self.w13, self.w2,
                           self.top_k, self.num_local_experts, self.num_experts, self.ep_size)


def ep_policy(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition

    for node in get_pas_ops(graph):
        if node.fn == step8_postprocess:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))
        elif node.fn == step5_compute:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
        elif node.fn == step2_permute:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))
        elif node.fn == step1_preprocess:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))
        elif node.fn == moe_forward:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0))


@replace_all_device_with('cpu')
def test_moe_whole(tmp_path):
    nnscaler.register_op(
        f'l h^, l m^, l m^, E t^ h^, E h^ d^ -> l h^',
        transform_rules=(build_ep_transform_rule(),)
    )(moe_forward)
    ep = MoE(embed_dim=32, moe_ffn_dim=64, num_experts=8, top_k=2)
    h = torch.randn(16, 32)

    nnscaler.parallelize(
        ep,
        {'x': h},
        ep_policy,
        nnscaler.ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=4,
        ),
        gen_savedir=tmp_path,
        load_module=False
    )
    # only x will be multiref'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.function.multiref')) == 1
    # only x and routing_probs needs to be split_allgather'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.nn.split_allgather')) == 2
    # only routing_map needs to be chunk'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.chunk')) == 1
    # only output needs to be allgather_split'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.nn.allgather_split')) == 1

    assert True
    # def segment107(self, x_25):
    #     # activation
    #     x_62, x_63 = nnscaler.runtime.function.multiref(x_25, times=2)
    #     del x_25
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_register_op.py", line 251, in forward,  logits = self.gate(x.float())
    #     float_1_27 = torch.Tensor.float(x_62)
    #     del x_62
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_register_op.py", line 251, in forward,  logits = self.gate(x.float())
    #     linear_29 = torch.nn.functional.linear(float_1_27, self.gate_weight_28, bias=None)
    #     del float_1_27
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_register_op.py", line 253, in forward,  routing_probs, routing_map, gate_scores = topk_routing(logits, self.top_k)
    #     topk_routing_30, topk_routing_31, topk_routing_32 = tests.parallel_module.test_gencode_register_op.topk_routing(linear_29, top_k=2)
    #     del linear_29, topk_routing_32
    #     topk_routing_50 = nnscaler.runtime.adapter.nn.split_allgather(topk_routing_30, dim=0, ranks=[0, 1])
    #     del topk_routing_30
    #     topk_routing_48 = nnscaler.runtime.adapter.chunk(topk_routing_31, dim=0, ranks=[0, 1])
    #     del topk_routing_31
    #     x_66 = nnscaler.runtime.adapter.nn.split_allgather(x_63, dim=0, ranks=[0, 1])
    #     del x_63
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_register_op.py", line 255, in forward,  return moe_forward(x, routing_map, routing_probs, self.w13, self.w2,
    #     moe_forward_56 = tests.parallel_module.test_gencode_register_op.moe_forward(x_66, topk_routing_48, topk_routing_50, self.w13_52, self.w2_54, top_k=2, num_local_experts=4, num_experts=8, ep_size=2)
    #     del topk_routing_50, topk_routing_48, x_66
    #     moe_forward_26 = nnscaler.runtime.adapter.nn.allgather_split(moe_forward_56, dim=0, ranks=[0, 1])
    #     del moe_forward_56
    #     return moe_forward_26


@replace_all_device_with('cpu')
def test_moe_breakdown(tmp_path):
    from nnscaler.graph.parser.register import CustomizedOps
    CustomizedOps.kOpMap.pop('tests.parallel_module.test_gencode_ep.moe_forward', None)

    ep = MoE(embed_dim=32, moe_ffn_dim=64, num_experts=8, top_k=2)
    h = torch.randn(16, 32)

    nnscaler.parallelize(
        ep,
        {'x': h},
        ep_policy,
        nnscaler.ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=4,
        ),
        gen_savedir=tmp_path,
        load_module=False
    )
    # x / routing_map will be multiref'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.function.multiref')) == 2
    # only x and routing_probs needs to be split_allgather'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.nn.split_allgather')) == 2
    # only routing_map needs to be chunk'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.chunk')) == 1
    # only output needs to be allgather_split'ed
    assert len(_gencode_contains(tmp_path, MoE, 0, 'nnscaler.runtime.adapter.nn.allgather_split')) == 1

    # def segment211(self, x_78):
    #     # created at IRAdapterGener:local_consumer_multiref
    #     x_148, x_152 = nnscaler.runtime.function.multiref(x_78, times=2)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 235, in forward,  logits = self.gate(x.float())
    #     float_1_80 = torch.Tensor.float(x_148)
    #     del x_148
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 235, in forward,  logits = self.gate(x.float())
    #     linear_82 = torch.nn.functional.linear(float_1_80, self.gate_weight_81, bias=None)
    #     del float_1_80
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 237, in forward,  routing_probs, routing_map, gate_scores = topk_routing(logits, self.top_k)
    #     topk_routing_83, topk_routing_84, topk_routing_85 = tests.parallel_module.test_gencode_ep.topk_routing(linear_82, top_k=2)
    #     del linear_82, topk_routing_85
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 144, in moe_forward,  y = x.view(*x.shape) # shallow copy to break the autograd graph for x
    #     im_output_188 = builtins.getattr(x_78, 'shape')
    #     getattr_3_76 = im_output_188[0]
    #     getattr_3_77 = im_output_188[1]
    #     del im_output_188
    #     del x_78
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 144, in moe_forward,  y = x.view(*x.shape) # shallow copy to break the autograd graph for x
    #     view_86 = torch.Tensor.view(x_152, size=(getattr_3_76, getattr_3_77))
    #     del x_152
    #     topk_routing_122 = nnscaler.runtime.adapter.chunk(topk_routing_84, dim=0, ranks=[0, 1])
    #     del topk_routing_84
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 148, in moe_forward,  (
    #     step1_preprocess_87, step1_preprocess_88, step1_preprocess_32, step1_preprocess_35 = tests.parallel_module.test_gencode_ep.step1_preprocess(topk_routing_122, num_local_experts=4, num_experts=8, ep_size=2)
    #     view_124 = nnscaler.runtime.adapter.nn.split_allgather(view_86, dim=0, ranks=[0, 1])
    #     del view_86
    #     # created at IRAdapterGener:local_consumer_multiref
    #     view_165, view_169 = nnscaler.runtime.function.multiref(view_124, times=2)
    #     del view_124
    #     topk_routing_126 = nnscaler.runtime.adapter.nn.split_allgather(topk_routing_83, dim=0, ranks=[0, 1])
    #     del topk_routing_83
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 164, in moe_forward,  (
    #     step2_permute_89, step2_permute_90, step2_permute_91 = tests.parallel_module.test_gencode_ep.step2_permute(view_169, topk_routing_122, topk_routing_126, topk=2)
    #     del view_169, topk_routing_126
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 173, in moe_forward,  (
    #     step3_dispatch_92, step3_dispatch_93 = tests.parallel_module.test_gencode_ep.step3_dispatch(step2_permute_89, step2_permute_90, input_splits=step1_preprocess_32, output_splits=step1_preprocess_35)
    #     del step2_permute_89, step2_permute_90
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 182, in moe_forward,  (
    #     step4_dispatch_postprocess_94, step4_dispatch_postprocess_95 = tests.parallel_module.test_gencode_ep.step4_dispatch_postprocess(step3_dispatch_92, step3_dispatch_93, step1_preprocess_88, 8, num_local_experts=2)
    #     del step3_dispatch_92, step3_dispatch_93
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 189, in moe_forward,  global_input_tokens = step5_compute(global_input_tokens, w13, w2, num_tokens_per_local_expert, global_probs)
    #     step5_compute_98 = tests.parallel_module.test_gencode_ep.step5_compute(step4_dispatch_postprocess_94, self.w13_128, self.w2_130, step1_preprocess_87, step4_dispatch_postprocess_95)
    #     del step1_preprocess_87, step4_dispatch_postprocess_94, step4_dispatch_postprocess_95
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 193, in moe_forward,  expert_outs = step6_compute_postprocess(global_input_tokens, num_global_tokens_per_local_expert, num_experts, 2)
    #     step6_compute_postprocess_99 = tests.parallel_module.test_gencode_ep.step6_compute_postprocess(step5_compute_98, step1_preprocess_88, 8, 2)
    #     del step1_preprocess_88, step5_compute_98
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 197, in moe_forward,  expert_outs = step7_combine(expert_outs, input_splits, output_splits)
    #     step7_combine_100 = tests.parallel_module.test_gencode_ep.step7_combine(step6_compute_postprocess_99, step1_preprocess_32, step1_preprocess_35)
    #     del step6_compute_postprocess_99
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 201, in moe_forward,  output = step8_postprocess(y, routing_map, expert_outs, reversed_local_input_permutation_mapping)
    #     step8_postprocess_132 = tests.parallel_module.test_gencode_ep.step8_postprocess(view_165, topk_routing_122, step7_combine_100, step2_permute_91)
    #     del topk_routing_122, view_165, step2_permute_91, step7_combine_100
    #     step8_postprocess_79 = nnscaler.runtime.adapter.nn.allgather_split(step8_postprocess_132, dim=0, ranks=[0, 1])
    #     del step8_postprocess_132
    #     return step8_postprocess_79


def ep_policy_overlap(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition

    stage = 0
    # step0 ~ step2: stage=0
    # dispatch(step 3) stage=1
    # dispatch_postprocess(step 4) ~ step6: stage=2
    # combine(step 7) stage=3
    # step 8 ~ end: stage=4
    for node in get_pas_ops(graph):
        if node.fn == step8_postprocess:
            stage += 1
            yield OpPlan(node, partition=OpPartition(input=0, dim=0), stage_id=stage)
        elif node.fn == step7_combine:
            stage += 1
            yield OpPlan(node, stage_id=stage)
        elif node.fn == step5_compute:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0), stage_id=stage)
        elif node.fn == step4_dispatch_postprocess:
            stage += 1
            yield OpPlan(node, stage_id=stage)
        elif node.fn == step3_dispatch:
            stage += 1
            yield OpPlan(node, stage_id=stage)
        elif node.fn == step2_permute:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0), stage_id=stage)
        elif node.fn == step1_preprocess:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0), stage_id=stage)
        elif node.fn == moe_forward:
            yield OpPlan(node, partition=OpPartition(input=0, dim=0), stage_id=stage)
        else:
            yield OpPlan(node, stage_id=stage)


def sched_overlap(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
    """
    1F1B overlapped schedule with two CUDA streams.

    Warmup:   full forward  mb 0
    1F1B:     for i in 1..N-1  →  forward(mb_i) interleaved with backward(mb_{i-1})
    Cooldown: full backward  mb N-1

    Within each 1F1B step the interleaving pairs forward and backward segments
    that land on *different* streams so they truly execute concurrently:

        comp_stream: fwd0 │ bwd4 │ fwd2 │ bwd2 │ fwd4 │ bwd0
        comm_stream:      │ fwd1 │ bwd3 │ fwd3 │ bwd1 │
                          ↕      ↕      ↕      ↕
                       overlap overlap overlap overlap

    Each stream waits for the other stream's previous event at pair boundaries.
    """

    if num_microbatches != 4:
        raise ValueError("This toy schedule is hard-coded for num_microbatches=4")
    if num_stages != 5:
        raise ValueError("This toy schedule is hard-coded for num_stages=5")

    segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]
    assert len(fsegs) == num_stages, f"Mismatch of forward segment number ({len(fsegs)}) with num_stages ({num_stages})"

    # describe schedule
    sched = SchedulePlan(graph, num_microbatches)
    comm_sc = StreamContext(stream='comm',wait_streams=['comp'])
    comp_sc = StreamContext(stream='comp', wait_streams=['comm'])
    forward_event = 'forward'
    backward_event = 'backward'

    fseg_stream_name = ['comp', 'comm', 'comp', 'comm', 'comp']
    sched.stream_config = StreamConfig(
        dataloader=comp_sc,
        zero_grad=comp_sc,
        inter_segment_move=comm_sc,
        result_broadcast=comm_sc,
        grad_reduce=comm_sc,
    )
    # warm up: full forward of mb 0
    for sid in range(num_stages):
        sched.add_segment(fsegs[sid], 0, step=sid,
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_events=[forward_event])
            if sid == 0 else
            StreamContext(stream=fseg_stream_name[sid], wait_events=[forward_event], record_events=[forward_event])
        )

    # 1F1B: forward(i) + backward(i-1)
    backward_event_recorded = False
    for micro_idx in range(1, num_microbatches):
        # Phase 1: fwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0], micro_idx, step=sid,
            stream_context=StreamContext(stream='comp',wait_streams=['comm'], record_events=[forward_event])
        )

        # Phase 2-5: overlapping pairs
        for fwd_s, bwd_s in [(1, 4), (2, 3), (3, 2), (4, 1)]:
            sid += 1
            sched.add_segment(fsegs[fwd_s], micro_idx, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[fwd_s], record_events=[forward_event], wait_events=[forward_event]
                )
            )
            sid += 1
            sched.add_segment(fsegs[bwd_s].mirror, micro_idx - 1, step=sid,
                    stream_context=StreamContext(
                    stream=fseg_stream_name[bwd_s],
                    record_events=[backward_event],
                    wait_events=[backward_event] if backward_event_recorded else []
                )
            )
            backward_event_recorded = True

        # Phase 6: bwd_seg0 alone on comp_stream
        sid += 1
        sched.add_segment(fsegs[0].mirror, micro_idx - 1, step=sid,
                stream_context=StreamContext(
                stream=fseg_stream_name[0], record_events=[backward_event], wait_events=[backward_event]
            )
        )

    # Cooldown: backward last mb
    for stage in reversed(range(num_stages)):
        sid += 1
        sched.add_segment(fsegs[stage].mirror, num_microbatches - 1, step=sid,
            stream_context=
                StreamContext(stream='comp',wait_streams=['comm'], record_events=[backward_event])
                if stage == num_stages - 1 else
                StreamContext(stream=fseg_stream_name[stage], wait_events=[backward_event], record_events=[backward_event])
        )

    sched.finish()
    return sched


class MoeWithLoss(torch.nn.Module):
    def __init__(self, embed_dim, moe_ffn_dim, num_experts, top_k):
        super().__init__()
        self.moe = MoE(embed_dim, moe_ffn_dim, num_experts, top_k)

    def forward(self, x):
        output = self.moe(x)
        loss = torch.sum(output)
        return loss


@replace_all_device_with('cpu')
def test_moe_breakdown_overlap(tmp_path):
    ep = MoeWithLoss(embed_dim=32, moe_ffn_dim=64, num_experts=8, top_k=2)
    h = torch.randn(16, 32)

    nnscaler.parallelize(
        ep,
        {'x': h},
        ep_policy_overlap,
        nnscaler.ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=4,
            use_end2end=True,
            pas_config={
                'pipeline_nmicros': 4,
                'pipeline_size': 1,
                'pipeline_scheduler': sched_overlap
            }
        ),
        gen_savedir=tmp_path,
        load_module=False
    )
    assert True  # should success

    # def segment54(self, x_65):
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 235, in forward,  logits = self.gate(x.float())
    #     float_1_67 = torch.Tensor.float(x_65)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 235, in forward,  logits = self.gate(x.float())
    #     linear_69 = torch.nn.functional.linear(float_1_67, self.moe_gate_weight_68, bias=None)
    #     del float_1_67
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 237, in forward,  routing_probs, routing_map, gate_scores = topk_routing(logits, self.top_k)
    #     topk_routing_70, topk_routing_71, topk_routing_72 = tests.parallel_module.test_gencode_ep.topk_routing(linear_69, top_k=2)
    #     del linear_69, topk_routing_72
    #     # create at IRAdapterGener:autoref, comment before transformation: activation
    #     topk_routing_161, topk_routing_162 = nnscaler.runtime.function.multiref(topk_routing_71, times=2)
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 144, in moe_forward,  y = x.view(*x.shape) # shallow copy to break the autograd graph for x
    #     im_output_479 = builtins.getattr(x_65, 'shape')
    #     getattr_3_63 = im_output_479[0]
    #     getattr_3_64 = im_output_479[1]
    #     del im_output_479
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 144, in moe_forward,  y = x.view(*x.shape) # shallow copy to break the autograd graph for x
    #     view_73 = torch.Tensor.view(x_65, size=(getattr_3_63, getattr_3_64))
    #     del x_65
    #     # create at IRAdapterGener:autoref, comment before transformation: activation
    #     view_164 = nnscaler.runtime.function.multiref(view_73, times=1)
    #     topk_routing_165 = nnscaler.runtime.adapter.chunk(topk_routing_161, dim=0, ranks=[0, 1])
    #     del topk_routing_161
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 148, in moe_forward,  (
    #     step1_preprocess_74, step1_preprocess_75, step1_preprocess_29, step1_preprocess_32 = tests.parallel_module.test_gencode_ep.step1_preprocess(topk_routing_165, num_local_experts=4, num_experts=8, ep_size=2)
    #     del topk_routing_165
    #     topk_routing_99 = nnscaler.runtime.adapter.nn.split_allgather(topk_routing_70, dim=0, ranks=[0, 1])
    #     del topk_routing_70
    #     topk_routing_169 = nnscaler.runtime.adapter.chunk(topk_routing_162, dim=0, ranks=[0, 1])
    #     del topk_routing_162
    #     view_167 = nnscaler.runtime.adapter.chunk(view_164, dim=0, ranks=[0, 1])
    #     del view_164
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 164, in moe_forward,  (
    #     step2_permute_76, step2_permute_77, step2_permute_78 = tests.parallel_module.test_gencode_ep.step2_permute(view_167, topk_routing_169, topk_routing_99, topk=2)
    #     del topk_routing_99, topk_routing_169, view_167
    #     return topk_routing_71, view_73, step1_preprocess_74, step1_preprocess_75, step1_preprocess_29, step1_preprocess_32, step2_permute_76, step2_permute_77, step2_permute_78

    # def segment59(self, topk_routing_71, view_73, step1_preprocess_74, step1_preprocess_75, step1_preprocess_29, step1_preprocess_32, step2_permute_76, step2_permute_77, step2_permute_78):
    #     step2_permute_137 = nnscaler.runtime.function.identity(step2_permute_78)
    #     del step2_permute_78
    #     step2_permute_134 = nnscaler.runtime.function.identity(step2_permute_77)
    #     del step2_permute_77
    #     step2_permute_132 = nnscaler.runtime.function.identity(step2_permute_76)
    #     del step2_permute_76
    #     step1_preprocess_128 = nnscaler.runtime.function.identity(step1_preprocess_75)
    #     del step1_preprocess_75
    #     step1_preprocess_124 = nnscaler.runtime.function.identity(step1_preprocess_74)
    #     del step1_preprocess_74
    #     view_116 = nnscaler.runtime.function.identity(view_73)
    #     del view_73
    #     topk_routing_108 = nnscaler.runtime.function.identity(topk_routing_71)
    #     del topk_routing_71
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 173, in moe_forward,  (
    #     step3_dispatch_79, step3_dispatch_80 = tests.parallel_module.test_gencode_ep.step3_dispatch(step2_permute_132, step2_permute_134, input_splits=step1_preprocess_29, output_splits=step1_preprocess_32)
    #     del step2_permute_134, step2_permute_132
    #     return step3_dispatch_79, step3_dispatch_80, topk_routing_108, view_116, step1_preprocess_124, step1_preprocess_128, step2_permute_137

    # def segment62(self, step3_dispatch_79, step3_dispatch_80, topk_routing_108, view_116, step1_preprocess_124, step1_preprocess_128, step2_permute_137):
    #     step3_dispatch_154 = nnscaler.runtime.function.identity(step3_dispatch_80)
    #     del step3_dispatch_80
    #     step3_dispatch_152 = nnscaler.runtime.function.identity(step3_dispatch_79)
    #     del step3_dispatch_79
    #     step2_permute_141 = nnscaler.runtime.function.identity(step2_permute_137)
    #     del step2_permute_137
    #     step1_preprocess_130 = nnscaler.runtime.function.identity(step1_preprocess_128)
    #     del step1_preprocess_128
    #     step1_preprocess_126 = nnscaler.runtime.function.identity(step1_preprocess_124)
    #     del step1_preprocess_124
    #     view_118 = nnscaler.runtime.function.identity(view_116)
    #     del view_116
    #     topk_routing_110 = nnscaler.runtime.function.identity(topk_routing_108)
    #     del topk_routing_108
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 182, in moe_forward,  (
    #     step4_dispatch_postprocess_81, step4_dispatch_postprocess_82 = tests.parallel_module.test_gencode_ep.step4_dispatch_postprocess(step3_dispatch_152, step3_dispatch_154, step1_preprocess_130, 8, num_local_experts=2)
    #     del step3_dispatch_154, step3_dispatch_152
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 189, in moe_forward,  global_input_tokens = step5_compute(global_input_tokens, w13, w2, num_tokens_per_local_expert, global_probs)
    #     step5_compute_85 = tests.parallel_module.test_gencode_ep.step5_compute(step4_dispatch_postprocess_81, self.moe_w13_101, self.moe_w2_103, step1_preprocess_126, step4_dispatch_postprocess_82)
    #     del step1_preprocess_126, step4_dispatch_postprocess_81, step4_dispatch_postprocess_82
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 193, in moe_forward,  expert_outs = step6_compute_postprocess(global_input_tokens, num_global_tokens_per_local_expert, num_experts, 2)
    #     step6_compute_postprocess_86 = tests.parallel_module.test_gencode_ep.step6_compute_postprocess(step5_compute_85, step1_preprocess_130, 8, 2)
    #     del step1_preprocess_130, step5_compute_85
    #     return step6_compute_postprocess_86, topk_routing_110, view_118, step2_permute_141

    # def segment65(self, step1_preprocess_29, step1_preprocess_32, step6_compute_postprocess_86, topk_routing_110, view_118, step2_permute_141):
    #     step6_compute_postprocess_156 = nnscaler.runtime.function.identity(step6_compute_postprocess_86)
    #     del step6_compute_postprocess_86
    #     step2_permute_145 = nnscaler.runtime.function.identity(step2_permute_141)
    #     del step2_permute_141
    #     view_120 = nnscaler.runtime.function.identity(view_118)
    #     del view_118
    #     topk_routing_112 = nnscaler.runtime.function.identity(topk_routing_110)
    #     del topk_routing_110
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 197, in moe_forward,  expert_outs = step7_combine(expert_outs, input_splits, output_splits)
    #     step7_combine_87 = tests.parallel_module.test_gencode_ep.step7_combine(step6_compute_postprocess_156, step1_preprocess_29, step1_preprocess_32)
    #     del step6_compute_postprocess_156
    #     return step7_combine_87, topk_routing_112, view_120, step2_permute_145

    # def segment68(self, step7_combine_87, topk_routing_112, view_120, step2_permute_145):
    #     step7_combine_158 = nnscaler.runtime.function.identity(step7_combine_87)
    #     del step7_combine_87
    #     step2_permute_149 = nnscaler.runtime.function.identity(step2_permute_145)
    #     del step2_permute_145
    #     view_122 = nnscaler.runtime.function.identity(view_120)
    #     del view_120
    #     topk_routing_114 = nnscaler.runtime.function.identity(topk_routing_112)
    #     del topk_routing_112
    #     topk_routing_175 = nnscaler.runtime.adapter.chunk(topk_routing_114, dim=0, ranks=[0, 1])
    #     del topk_routing_114
    #     view_173 = nnscaler.runtime.adapter.chunk(view_122, dim=0, ranks=[0, 1])
    #     del view_122
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 201, in moe_forward,  output = step8_postprocess(y, routing_map, expert_outs, reversed_local_input_permutation_mapping)
    #     step8_postprocess_105 = tests.parallel_module.test_gencode_ep.step8_postprocess(view_173, topk_routing_175, step7_combine_158, step2_permute_149)
    #     del step7_combine_158, step2_permute_149, topk_routing_175, view_173
    #     step8_postprocess_88 = nnscaler.runtime.adapter.all_gather(step8_postprocess_105, dim=0, ranks=[0, 1])
    #     del step8_postprocess_105
    #     # File "/data/weijiangxu/nnscaler/tests/parallel_module/test_gencode_ep.py", line 543, in forward,  loss = torch.sum(output)
    #     sum_1_66 = torch.sum(step8_postprocess_88)
    #     del step8_postprocess_88
    #     return sum_1_66

    # def _train_step(model, dataloader_89):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         model.zero_grad()
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_65 = next(*(dataloader_89, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_65.record_stream(torch.cuda.current_stream())
    #         topk_routing_71, view_73, step1_preprocess_74, step1_preprocess_75, step1_preprocess_29, step1_preprocess_32, step2_permute_76, step2_permute_77, step2_permute_78 = nnscaler.runtime.executor.fexecute('segment54', model.segment54, *(x_65, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del x_65

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         topk_routing_71.record_stream(torch.cuda.current_stream())
    #         view_73.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_74.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_75.record_stream(torch.cuda.current_stream())
    #         step2_permute_76.record_stream(torch.cuda.current_stream())
    #         step2_permute_77.record_stream(torch.cuda.current_stream())
    #         step2_permute_78.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_79, step3_dispatch_80, topk_routing_108, view_116, step1_preprocess_124, step1_preprocess_128, step2_permute_137 = nnscaler.runtime.executor.fexecute('segment59', model.segment59, *(topk_routing_71, view_73, step1_preprocess_74, step1_preprocess_75, step1_preprocess_29, step1_preprocess_32, step2_permute_76, step2_permute_77, step2_permute_78, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del topk_routing_71, view_73, step1_preprocess_74, step1_preprocess_75, step2_permute_76, step2_permute_77

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step3_dispatch_79.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_80.record_stream(torch.cuda.current_stream())
    #         topk_routing_108.record_stream(torch.cuda.current_stream())
    #         view_116.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_124.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_128.record_stream(torch.cuda.current_stream())
    #         step2_permute_137.record_stream(torch.cuda.current_stream())
    #         step6_compute_postprocess_86, topk_routing_110, view_118, step2_permute_141 = nnscaler.runtime.executor.fexecute('segment62', model.segment62, *(step3_dispatch_79, step3_dispatch_80, topk_routing_108, view_116, step1_preprocess_124, step1_preprocess_128, step2_permute_137, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step3_dispatch_79, step3_dispatch_80, topk_routing_108, view_116, step1_preprocess_124, step1_preprocess_128

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step6_compute_postprocess_86.record_stream(torch.cuda.current_stream())
    #         topk_routing_110.record_stream(torch.cuda.current_stream())
    #         view_118.record_stream(torch.cuda.current_stream())
    #         step2_permute_141.record_stream(torch.cuda.current_stream())
    #         step7_combine_87, topk_routing_112, view_120, step2_permute_145 = nnscaler.runtime.executor.fexecute('segment65', model.segment65, *(step1_preprocess_29, step1_preprocess_32, step6_compute_postprocess_86, topk_routing_110, view_118, step2_permute_141, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step6_compute_postprocess_86, topk_routing_110, view_118

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step7_combine_87.record_stream(torch.cuda.current_stream())
    #         topk_routing_112.record_stream(torch.cuda.current_stream())
    #         view_120.record_stream(torch.cuda.current_stream())
    #         step2_permute_145.record_stream(torch.cuda.current_stream())
    #         sum_1_66 = nnscaler.runtime.executor.fexecute('segment68', model.segment68, *(step7_combine_87, topk_routing_112, view_120, step2_permute_145, ), requires_grad=False)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step7_combine_87, topk_routing_112, view_120

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_186 = next(*(dataloader_89, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_186.record_stream(torch.cuda.current_stream())
    #         topk_routing_190, view_192, step1_preprocess_194, step1_preprocess_196, step1_preprocess_197, step1_preprocess_198, step2_permute_200, step2_permute_202, step2_permute_205 = nnscaler.runtime.executor.fexecute('segment54', model.segment54, *(x_186, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del x_186

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         topk_routing_190.record_stream(torch.cuda.current_stream())
    #         view_192.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_194.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_196.record_stream(torch.cuda.current_stream())
    #         step2_permute_200.record_stream(torch.cuda.current_stream())
    #         step2_permute_202.record_stream(torch.cuda.current_stream())
    #         step2_permute_205.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_219, step3_dispatch_221, topk_routing_223, view_225, step1_preprocess_227, step1_preprocess_229, step2_permute_232 = nnscaler.runtime.executor.fexecute('segment59', model.segment59, *(topk_routing_190, view_192, step1_preprocess_194, step1_preprocess_196, step1_preprocess_197, step1_preprocess_198, step2_permute_200, step2_permute_202, step2_permute_205, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del topk_routing_190, view_192, step1_preprocess_194, step1_preprocess_196, step2_permute_200, step2_permute_202
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         _ = nnscaler.runtime.executor.backward('segment68', (), (), ())
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step3_dispatch_219.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_221.record_stream(torch.cuda.current_stream())
    #         topk_routing_223.record_stream(torch.cuda.current_stream())
    #         view_225.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_227.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_229.record_stream(torch.cuda.current_stream())
    #         step2_permute_232.record_stream(torch.cuda.current_stream())
    #         step6_compute_postprocess_245, topk_routing_247, view_249, step2_permute_252 = nnscaler.runtime.executor.fexecute('segment62', model.segment62, *(step3_dispatch_219, step3_dispatch_221, topk_routing_223, view_225, step1_preprocess_227, step1_preprocess_229, step2_permute_232, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step3_dispatch_219, step3_dispatch_221, topk_routing_223, view_225, step1_preprocess_227, step1_preprocess_229
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_141.record_stream(torch.cuda.current_stream())
    #         step2_permute_145.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_146.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_142 = nnscaler.runtime.executor.backward('segment65', (step2_permute_141, ), (step2_permute_145, ), (gstep2_permute_146, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_145, gstep2_permute_146

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step6_compute_postprocess_245.record_stream(torch.cuda.current_stream())
    #         topk_routing_247.record_stream(torch.cuda.current_stream())
    #         view_249.record_stream(torch.cuda.current_stream())
    #         step2_permute_252.record_stream(torch.cuda.current_stream())
    #         step7_combine_264, topk_routing_266, view_268, step2_permute_271 = nnscaler.runtime.executor.fexecute('segment65', model.segment65, *(step1_preprocess_197, step1_preprocess_198, step6_compute_postprocess_245, topk_routing_247, view_249, step2_permute_252, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step6_compute_postprocess_245, topk_routing_247, view_249
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_137.record_stream(torch.cuda.current_stream())
    #         step2_permute_141.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_142.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_138 = nnscaler.runtime.executor.backward('segment62', (step2_permute_137, ), (step2_permute_141, ), (gstep2_permute_142, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_141, gstep2_permute_142

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step7_combine_264.record_stream(torch.cuda.current_stream())
    #         topk_routing_266.record_stream(torch.cuda.current_stream())
    #         view_268.record_stream(torch.cuda.current_stream())
    #         step2_permute_271.record_stream(torch.cuda.current_stream())
    #         sum_1_281 = nnscaler.runtime.executor.fexecute('segment68', model.segment68, *(step7_combine_264, topk_routing_266, view_268, step2_permute_271, ), requires_grad=False)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step7_combine_264, topk_routing_266, view_268
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_78.record_stream(torch.cuda.current_stream())
    #         step2_permute_137.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_138.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_94 = nnscaler.runtime.executor.backward('segment59', (step2_permute_78, ), (step2_permute_137, ), (gstep2_permute_138, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_137, gstep2_permute_138
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_78.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_94.record_stream(torch.cuda.current_stream())
    #         _ = nnscaler.runtime.executor.backward('segment54', (), (step2_permute_78, ), (gstep2_permute_94, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_78, gstep2_permute_94

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_283 = next(*(dataloader_89, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_283.record_stream(torch.cuda.current_stream())
    #         topk_routing_287, view_289, step1_preprocess_291, step1_preprocess_293, step1_preprocess_294, step1_preprocess_295, step2_permute_297, step2_permute_299, step2_permute_302 = nnscaler.runtime.executor.fexecute('segment54', model.segment54, *(x_283, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del x_283

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         topk_routing_287.record_stream(torch.cuda.current_stream())
    #         view_289.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_291.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_293.record_stream(torch.cuda.current_stream())
    #         step2_permute_297.record_stream(torch.cuda.current_stream())
    #         step2_permute_299.record_stream(torch.cuda.current_stream())
    #         step2_permute_302.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_316, step3_dispatch_318, topk_routing_320, view_322, step1_preprocess_324, step1_preprocess_326, step2_permute_329 = nnscaler.runtime.executor.fexecute('segment59', model.segment59, *(topk_routing_287, view_289, step1_preprocess_291, step1_preprocess_293, step1_preprocess_294, step1_preprocess_295, step2_permute_297, step2_permute_299, step2_permute_302, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del topk_routing_287, view_289, step1_preprocess_291, step1_preprocess_293, step2_permute_297, step2_permute_299
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('segment68', (), (), ())
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step3_dispatch_316.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_318.record_stream(torch.cuda.current_stream())
    #         topk_routing_320.record_stream(torch.cuda.current_stream())
    #         view_322.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_324.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_326.record_stream(torch.cuda.current_stream())
    #         step2_permute_329.record_stream(torch.cuda.current_stream())
    #         step6_compute_postprocess_342, topk_routing_344, view_346, step2_permute_349 = nnscaler.runtime.executor.fexecute('segment62', model.segment62, *(step3_dispatch_316, step3_dispatch_318, topk_routing_320, view_322, step1_preprocess_324, step1_preprocess_326, step2_permute_329, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step3_dispatch_316, step3_dispatch_318, topk_routing_320, view_322, step1_preprocess_324, step1_preprocess_326
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_252.record_stream(torch.cuda.current_stream())
    #         step2_permute_271.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_272.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_253 = nnscaler.runtime.executor.backward('segment65', (step2_permute_252, ), (step2_permute_271, ), (gstep2_permute_272, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_271, gstep2_permute_272

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step6_compute_postprocess_342.record_stream(torch.cuda.current_stream())
    #         topk_routing_344.record_stream(torch.cuda.current_stream())
    #         view_346.record_stream(torch.cuda.current_stream())
    #         step2_permute_349.record_stream(torch.cuda.current_stream())
    #         step7_combine_361, topk_routing_363, view_365, step2_permute_368 = nnscaler.runtime.executor.fexecute('segment65', model.segment65, *(step1_preprocess_294, step1_preprocess_295, step6_compute_postprocess_342, topk_routing_344, view_346, step2_permute_349, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step6_compute_postprocess_342, topk_routing_344, view_346
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_232.record_stream(torch.cuda.current_stream())
    #         step2_permute_252.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_253.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_233 = nnscaler.runtime.executor.backward('segment62', (step2_permute_232, ), (step2_permute_252, ), (gstep2_permute_253, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_252, gstep2_permute_253

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step7_combine_361.record_stream(torch.cuda.current_stream())
    #         topk_routing_363.record_stream(torch.cuda.current_stream())
    #         view_365.record_stream(torch.cuda.current_stream())
    #         step2_permute_368.record_stream(torch.cuda.current_stream())
    #         sum_1_378 = nnscaler.runtime.executor.fexecute('segment68', model.segment68, *(step7_combine_361, topk_routing_363, view_365, step2_permute_368, ), requires_grad=False)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step7_combine_361, topk_routing_363, view_365
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_205.record_stream(torch.cuda.current_stream())
    #         step2_permute_232.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_233.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_206 = nnscaler.runtime.executor.backward('segment59', (step2_permute_205, ), (step2_permute_232, ), (gstep2_permute_233, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_232, gstep2_permute_233
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_205.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_206.record_stream(torch.cuda.current_stream())
    #         _ = nnscaler.runtime.executor.backward('segment54', (), (step2_permute_205, ), (gstep2_permute_206, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_205, gstep2_permute_206

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_380 = next(*(dataloader_89, ))
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         x_380.record_stream(torch.cuda.current_stream())
    #         topk_routing_384, view_386, step1_preprocess_388, step1_preprocess_390, step1_preprocess_391, step1_preprocess_392, step2_permute_394, step2_permute_396, step2_permute_399 = nnscaler.runtime.executor.fexecute('segment54', model.segment54, *(x_380, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del x_380

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         topk_routing_384.record_stream(torch.cuda.current_stream())
    #         view_386.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_388.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_390.record_stream(torch.cuda.current_stream())
    #         step2_permute_394.record_stream(torch.cuda.current_stream())
    #         step2_permute_396.record_stream(torch.cuda.current_stream())
    #         step2_permute_399.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_413, step3_dispatch_415, topk_routing_417, view_419, step1_preprocess_421, step1_preprocess_423, step2_permute_426 = nnscaler.runtime.executor.fexecute('segment59', model.segment59, *(topk_routing_384, view_386, step1_preprocess_388, step1_preprocess_390, step1_preprocess_391, step1_preprocess_392, step2_permute_394, step2_permute_396, step2_permute_399, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del topk_routing_384, view_386, step1_preprocess_388, step1_preprocess_390, step2_permute_394, step2_permute_396
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         _ = nnscaler.runtime.executor.backward('segment68', (), (), ())
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step3_dispatch_413.record_stream(torch.cuda.current_stream())
    #         step3_dispatch_415.record_stream(torch.cuda.current_stream())
    #         topk_routing_417.record_stream(torch.cuda.current_stream())
    #         view_419.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_421.record_stream(torch.cuda.current_stream())
    #         step1_preprocess_423.record_stream(torch.cuda.current_stream())
    #         step2_permute_426.record_stream(torch.cuda.current_stream())
    #         step6_compute_postprocess_439, topk_routing_441, view_443, step2_permute_446 = nnscaler.runtime.executor.fexecute('segment62', model.segment62, *(step3_dispatch_413, step3_dispatch_415, topk_routing_417, view_419, step1_preprocess_421, step1_preprocess_423, step2_permute_426, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step3_dispatch_413, step3_dispatch_415, topk_routing_417, view_419, step1_preprocess_421, step1_preprocess_423
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_349.record_stream(torch.cuda.current_stream())
    #         step2_permute_368.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_369.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_350 = nnscaler.runtime.executor.backward('segment65', (step2_permute_349, ), (step2_permute_368, ), (gstep2_permute_369, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_368, gstep2_permute_369

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step6_compute_postprocess_439.record_stream(torch.cuda.current_stream())
    #         topk_routing_441.record_stream(torch.cuda.current_stream())
    #         view_443.record_stream(torch.cuda.current_stream())
    #         step2_permute_446.record_stream(torch.cuda.current_stream())
    #         step7_combine_458, topk_routing_460, view_462, step2_permute_465 = nnscaler.runtime.executor.fexecute('segment65', model.segment65, *(step1_preprocess_391, step1_preprocess_392, step6_compute_postprocess_439, topk_routing_441, view_443, step2_permute_446, ), requires_grad=True)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step6_compute_postprocess_439, topk_routing_441, view_443
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_329.record_stream(torch.cuda.current_stream())
    #         step2_permute_349.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_350.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_330 = nnscaler.runtime.executor.backward('segment62', (step2_permute_329, ), (step2_permute_349, ), (gstep2_permute_350, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_349, gstep2_permute_350

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').wait()
    #         step7_combine_458.record_stream(torch.cuda.current_stream())
    #         topk_routing_460.record_stream(torch.cuda.current_stream())
    #         view_462.record_stream(torch.cuda.current_stream())
    #         step2_permute_465.record_stream(torch.cuda.current_stream())
    #         sum_1_475 = nnscaler.runtime.executor.fexecute('segment68', model.segment68, *(step7_combine_458, topk_routing_460, view_462, step2_permute_465, ), requires_grad=False)
    #         nnscaler.runtime.device.DeviceGroup().get_event('forward').record()

    #     del step7_combine_458, topk_routing_460, view_462
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_302.record_stream(torch.cuda.current_stream())
    #         step2_permute_329.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_330.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_303 = nnscaler.runtime.executor.backward('segment59', (step2_permute_302, ), (step2_permute_329, ), (gstep2_permute_330, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_329, gstep2_permute_330
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_302.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_303.record_stream(torch.cuda.current_stream())
    #         _ = nnscaler.runtime.executor.backward('segment54', (), (step2_permute_302, ), (gstep2_permute_303, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_302, gstep2_permute_303
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm'))
    #         _ = nnscaler.runtime.executor.backward('segment68', (), (), ())
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_446.record_stream(torch.cuda.current_stream())
    #         step2_permute_465.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_466.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_447 = nnscaler.runtime.executor.backward('segment65', (step2_permute_446, ), (step2_permute_465, ), (gstep2_permute_466, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_465, gstep2_permute_466
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_426.record_stream(torch.cuda.current_stream())
    #         step2_permute_446.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_447.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_427 = nnscaler.runtime.executor.backward('segment62', (step2_permute_426, ), (step2_permute_446, ), (gstep2_permute_447, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_446, gstep2_permute_447
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_399.record_stream(torch.cuda.current_stream())
    #         step2_permute_426.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_427.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_400 = nnscaler.runtime.executor.backward('segment59', (step2_permute_399, ), (step2_permute_426, ), (gstep2_permute_427, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_426, gstep2_permute_427
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp')):
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').wait()
    #         step2_permute_399.record_stream(torch.cuda.current_stream())
    #         gstep2_permute_400.record_stream(torch.cuda.current_stream())
    #         _ = nnscaler.runtime.executor.backward('segment54', (), (step2_permute_399, ), (gstep2_permute_400, ))
    #         nnscaler.runtime.device.DeviceGroup().get_event('backward').record()

    #     del step2_permute_399, gstep2_permute_400

    #     with torch.cuda.stream(nnscaler.runtime.device.DeviceGroup().get_stream('comm')):
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp'))
    #         _ = nnscaler.runtime.executor.aexecute(model.reducer732, *(), requires_grad=False)
    #         torch.cuda.current_stream().wait_stream(nnscaler.runtime.device.DeviceGroup().get_stream('comp'))
    #         _ = nnscaler.runtime.executor.aexecute(model.reducer733, *(), requires_grad=False)

    #     return sum_1_66, sum_1_281, sum_1_378, sum_1_475
