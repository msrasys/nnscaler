"""Utility functions for MoE FWD-BWD overlap scheduling.

Gradient sanitization, reducer buffer management, and chunked vocab gradient.
"""

import logging

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_logger = logging.getLogger(__name__)

_VOCAB_GRAD_CHUNK = 25088  # rows per chunk; 25088 * 1024 * 4 = 98 MiB


def find_param_in_reducer(parallel_module, target_param):
    """Find the contiguous grad buffer slice for a parameter in any reducer bucket.
    Returns (grad_buffer, offset) or (None, None) if not found.
    """
    pm = parallel_module
    if hasattr(pm, 'backbone'):
        pm = pm.backbone
    if not hasattr(pm, '_reducers'):
        return None, None
    for reducer in pm._reducers:
        for bucket in reducer._buckets:
            if target_param in bucket._pofset:
                return bucket._contiguous_grads, bucket._pofset[target_param]
    return None, None


def manual_sync_grads(parallel_module):
    """Manually trigger synchronous allreduce after all backward calls.

    The merged scheduler uses skip_reducer=True, so hooks copy grads to the
    contiguous buffer but skip counting/triggering allreduce. This function
    performs the allreduce and sets param.grad from the buffer.
    """
    pm = parallel_module
    if hasattr(pm, 'backbone'):
        pm = pm.backbone
    if not hasattr(pm, '_reducers'):
        _logger.warning("No _reducers found on parallel module, skipping manual sync")
        return

    for reducer in pm._reducers:
        for bucket in reducer._buckets:
            old_async = bucket._async
            bucket._async = False
            try:
                bucket.sync_grads()
            finally:
                bucket._async = old_async
            bucket.reset()


def make_chunked_output_linear(grad_buffer, buffer_offset, vocab_chunk=None):
    """Return an apply()-compatible function that replaces mix_precision_linear
    for the output layer, writing grad_weight to the reducer buffer in chunks.
    """
    if vocab_chunk is None:
        vocab_chunk = _VOCAB_GRAD_CHUNK

    class _CGL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight):
            compute_dtype = input.dtype
            weight_cast = weight.to(compute_dtype)
            ctx.save_for_backward(input, weight_cast)
            return F.linear(input, weight_cast)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_input = F.linear(grad_output, weight.t())

            if ctx.needs_input_grad[1]:
                vocab, d = weight.shape[0], weight.shape[1]
                input_fp32 = input.float()
                for vs in range(0, vocab, vocab_chunk):
                    ve = min(vs + vocab_chunk, vocab)
                    chunk_grad = torch.mm(
                        grad_output[:, vs:ve].t().float(), input_fp32)
                    s = buffer_offset + vs * d
                    e = buffer_offset + ve * d
                    grad_buffer[s:e].add_(chunk_grad.view(-1))
                    del chunk_grad
                del input_fp32

            return grad_input, None

    return _CGL.apply


def chunked_linear_cross_entropy(x, w, y, chunked_linear_fn):
    """Drop-in replacement for linear_cross_entropy using chunked output linear."""
    logits = chunked_linear_fn(x, w).to(torch.float32)
    log_z = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    loss = log_z - target_logits
    z_loss = log_z ** 2
    return loss, z_loss


def merged_chunk_linear_cross_entropy(x, w, y, chunk_size, enable_checkpoint,
                                      chunked_linear_fn):
    """Drop-in replacement for inner_chunk_linear_cross_entropy using chunked grad."""
    token_num, hidden_size = x.size()
    if token_num % chunk_size != 0:
        raise ValueError(
            f"token_num {token_num} is not divisible by chunk_size {chunk_size}")
    chunk_num = token_num // chunk_size
    xs = x.view(chunk_num, chunk_size, hidden_size)
    ys = y.view(chunk_num, chunk_size)
    losses = []
    z_losses = []
    for i in range(chunk_num):
        if enable_checkpoint:
            loss, z_loss = torch_checkpoint(
                chunked_linear_cross_entropy,
                xs[i], w, ys[i], chunked_linear_fn,
                use_reentrant=True)
        else:
            loss, z_loss = chunked_linear_cross_entropy(
                xs[i], w, ys[i], chunked_linear_fn)
        losses.append(loss)
        z_losses.append(z_loss)
    return torch.cat(losses), torch.cat(z_losses)
