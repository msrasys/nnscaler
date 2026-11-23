# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepEP configuration and same-node EP2 peer-exchange transport.

PeerBuffer adapts an existing DeepEP Buffer. The caller removes local expert
routes and supplies equal token counts/layouts on the two ranks. Returned tensors
own allocator storage; IPC staging views never escape the transport.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import torch
import os

ENV_NAME = 'NNSCALER_DEEPEP_CONFIG'


def validate_configs(configs, max_sms):
    if not isinstance(configs, dict):
        raise ValueError(f'{ENV_NAME} must map EP sizes to dispatch/combine configs')
    result = {}
    for ranks, pair in configs.items():
        ranks = int(ranks)
        if ranks not in (2, 4, 8):
            raise ValueError('DeepEP default tuning currently supports intranode EP2/4/8')
        if not isinstance(pair, dict) or set(pair) != {'dispatch', 'combine'}:
            raise ValueError('Each EP size requires exactly dispatch and combine')
        converted = {}
        for operation, values in pair.items():
            if not isinstance(values, (list, tuple)) or len(values) != 5:
                raise ValueError('DeepEP configs contain five integers: SMs, NVL send/receive, RDMA send/receive')
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError('DeepEP configuration values must be positive integers')
            sms, nvl_send, nvl_recv, rdma_send, rdma_recv = values
            if sms % 2 or sms > max_sms:
                raise ValueError(f'DeepEP SM count must be even and at most {max_sms}')
            if nvl_send > nvl_recv or rdma_send > rdma_recv:
                raise ValueError('DeepEP send chunks must fit their receive buffers')
            converted[operation] = tuple(values)
        if converted['dispatch'][0] != converted['combine'][0]:
            raise ValueError('Dispatch and combine must use the same channel/SM count')
        if converted['dispatch'][2] != converted['combine'][2]:
            raise ValueError('Dispatch and combine must use the same NVL receive capacity')
        result[ranks] = converted
    return result


@lru_cache(maxsize=8)
def _configs(value, max_sms):
    return validate_configs(json.loads(value), max_sms)


def get_config_override(num_ranks, operation):
    """Return an explicit DeepEP config override, without modifying the library."""
    value = os.environ.get(ENV_NAME)
    if value is None:
        return None
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine configuration')
    from deep_ep import Config

    pair = _configs(
        value,
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
    ).get(num_ranks)
    return Config(*pair[operation]) if pair is not None else None


@dataclass
class _Handle:
    _nnscaler_transport: object


@torch.compile
def _receive_indices(indices, rank, experts):
    start = rank * experts
    valid = (indices >= start) & (indices < start + experts)
    return torch.where(valid, indices - start, -1)


class PeerBuffer:
    @staticmethod
    def required_buffer_bytes(x, topk, num_experts):
        # Backward may exchange FP32 values even when forward inputs are BF16.
        sizes = (
            x.numel() * max(x.element_size(), 4),
            x.shape[0] * topk * 8,
            x.shape[0] * topk * 4,
            num_experts * 4,
        )
        return max(sum((n + 255) // 256 * 256 for n in sizes), 512 * 1024**2)

    def __init__(self, native_buffer, num_tokens):
        self.native = native_buffer
        self.tokens = num_tokens
        if self.native.group_size != 2 or self.native.runtime.get_num_rdma_ranks() != 1:
            raise ValueError('Peer expert exchange requires same-node EP2')
        self.exchange = getattr(self.native.runtime, 'peer_exchange_copy', None)
        if self.exchange is None:
            raise RuntimeError('Build DeepEP with its peer-exchange API before enabling peer copy')

    def __getattr__(self, name):
        return getattr(self.native, name)

    def dispatch(
        self,
        x,
        handle=None,
        *,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_expert=None,
        expert_alignment=1,
        num_worst_tokens=0,
        previous_event=None,
        async_finish=False,
        **kwargs,
    ):
        from deep_ep.utils import EventOverlap, EventHandle

        if (
            x.ndim != 2
            or x.shape[0] != self.tokens
            or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float32)
            or expert_alignment != 1
            or num_worst_tokens
        ):
            raise ValueError("Unsupported tensor layout or padding for peer expert exchange")
        caller = torch.cuda.current_stream()
        stream = self.native.get_comm_stream()
        if previous_event is not None:
            with torch.cuda.stream(stream):
                previous_event.current_stream_wait()
        if handle is not None:
            if handle._nnscaler_transport is not self or topk_idx is not None or topk_weights is not None:
                raise ValueError("Cached peer dispatch must use its original transport handle")
            outputs, event = self.exchange([x], False)
            ready = EventOverlap(event)
            if not async_finish:
                ready.current_stream_wait()
            return outputs[0], None, None, None, None, ready
        if topk_idx is None or topk_weights is None or num_tokens_per_expert is None:
            raise ValueError("Peer dispatch requires indices, probabilities and expert counts")
        if topk_idx.shape != topk_weights.shape or topk_idx.shape[0] != self.tokens:
            raise ValueError("Peer routing tensors must match the declared token count")
        outputs, _ = self.exchange(
            [
                x,
                topk_idx.contiguous(),
                topk_weights.contiguous(),
                num_tokens_per_expert.contiguous(),
            ],
            False,
        )
        experts = num_tokens_per_expert.numel() // 2
        with torch.cuda.stream(stream):
            indices = _receive_indices(outputs[1], self.native.rank, experts)
            counts = outputs[3].cpu().tolist()
            start = self.native.rank * experts
            end = start + experts
            if any(counts[:start] + counts[end:]):
                raise ValueError("Remove local expert routes before peer exchange")
            ready = EventOverlap(EventHandle())
        indices.record_stream(caller)
        if not async_finish:
            ready.current_stream_wait()
        return (
            outputs[0],
            indices,
            outputs[2],
            counts[start:end],
            _Handle(self),
            ready,
        )

    def combine(
        self,
        x,
        handle,
        topk_weights=None,
        bias=None,
        *,
        previous_event=None,
        async_finish=False,
        **kwargs,
    ):
        from deep_ep.utils import EventOverlap

        if (
            handle._nnscaler_transport is not self
            or bias is not None
            or x.ndim != 2
            or x.shape[0] != self.tokens
            or not x.is_contiguous()
        ):
            raise ValueError("Peer combine must use the original layout and transport handle")
        if previous_event is not None:
            with torch.cuda.stream(self.native.get_comm_stream()):
                previous_event.current_stream_wait()
        values = [x]
        if topk_weights is not None:
            if topk_weights.shape[0] != self.tokens:
                raise ValueError("Peer probability gradients must match the token count")
            values.append(topk_weights.contiguous())
        outputs, event = self.exchange(values, False)
        ready = EventOverlap(event)
        if not async_finish:
            ready.current_stream_wait()
        return outputs[0], outputs[1] if topk_weights is not None else None, ready
