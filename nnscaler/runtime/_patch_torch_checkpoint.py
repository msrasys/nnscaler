"""Checkpoint support for delayed dWeight in split backward.

PyTorch's ``GraphExecGroup`` shares a non-reentrant checkpoint replay across
multiple backward GraphTasks, but intentionally rejects a saved activation
used by more than one GraphTask. I/W split backward has exactly that case: an
activation can be read once for dInput and later again for dWeight.

This patch preserves those replayed tensors until the delayed dWeight GraphTask
finishes. Holders that are no longer referenced by the remaining graph still
drop normally, so I-only activations are not kept for W.
"""

import uuid
import weakref

import torch
import torch.utils.checkpoint as checkpoint_module


GraphExecGroup = getattr(checkpoint_module, "GraphExecGroup", None)


if GraphExecGroup is not None:
    class ReusableGraphExecGroup(GraphExecGroup):
        """A graph-execution group whose checkpoint tensors may be reused.

        A delayed dWeight can contain multiple autograd traversals (one per
        parameter group).  Those traversals are allowed to read the checkpoint
        replay produced by dInput, but the replay must be released as soon as
        the whole dWeight phase finishes.
        """

        def __init__(self):
            self._checkpoint_frames = set()

        def _register_checkpoint_frame(self, frame):
            self._checkpoint_frames.add(frame)

        def release(self):
            """Release every replay tensor owned by this execution group."""
            for frame in self._checkpoint_frames:
                for weak_holder in frame.weak_holders:
                    holder = weak_holder()
                    if holder is not None:
                        holder.handles.pop(self, None)
                frame.recomputed.pop(self, None)
                frame.recomp_counter.pop(self, None)
                frame.is_recomputed.pop(self, None)
            self._checkpoint_frames.clear()


    class _ReusableCheckpointHook(torch.autograd.graph.saved_tensors_hooks):
        def __init__(self, frame):
            def pack_hook(tensor):
                holder = checkpoint_module._Holder()
                frame.weak_holders.append(weakref.ref(holder))
                if frame.metadata_fn is not None:
                    with torch.no_grad():
                        frame.x_metadatas.append(frame.metadata_fn(tensor))
                return holder

            def unpack_hook(holder):
                gid = checkpoint_module.GraphExecGroup._get_current_group()
                if gid is None:
                    gid = torch._C._current_graph_task_id()
                    if gid == -1:
                        gid = int(uuid.uuid4())

                if not frame.is_recomputed[gid]:
                    ctx = frame.input_saver.grad_fn
                    args = ctx.get_args(ctx.saved_tensors)
                    try:
                        with (
                            checkpoint_module._recomputation_hook(
                                weakref.ref(frame), gid
                            ),
                            torch.autograd.enable_grad(),
                        ):
                            checkpoint_module._run_fn_with_dynamo_disabled(
                                frame.recompute_fn, *args
                            )
                    except checkpoint_module._StopRecomputationError:
                        pass
                    frame.is_recomputed[gid] = True
                    frame.check_recomputed_tensors_match(gid)

                if isinstance(gid, ReusableGraphExecGroup):
                    gid._register_checkpoint_frame(frame)

                checkpoint_module._internal_assert(gid in holder.handles)
                handle = holder.handles[gid]
                if handle is None:
                    raise checkpoint_module.CheckpointError(
                        "torch.utils.checkpoint: Unpack is being triggered for "
                        "a tensor that was already unpacked once. If you are "
                        "calling ctx.saved_tensors in backward, make sure to do "
                        "so only once. Otherwise please open an issue with "
                        "details on your use case."
                    )
                checkpoint_module._internal_assert(
                    handle in frame.recomputed[gid]
                )
                tensor = frame.recomputed[gid][handle]
                # A single W phase can execute several parameter-group
                # GraphTasks through the same custom/AOT backward. Keep the
                # handle valid for all of them. Executor.backward_weight calls
                # ``release`` after the final group, which avoids retaining
                # checkpoint replay tensors across microbatches.
                if not isinstance(gid, ReusableGraphExecGroup):
                    holder.handles[gid] = None
                return tensor

            if frame.unpack_error_cb is not None:
                def unpack_hook_with_error_cb(holder):
                    try:
                        return unpack_hook(holder)
                    except checkpoint_module.CheckpointError as error:
                        frame.unpack_error_cb(error)

                super().__init__(pack_hook, unpack_hook_with_error_cb)
            else:
                super().__init__(pack_hook, unpack_hook)
else:
    ReusableGraphExecGroup = None
    _ReusableCheckpointHook = None


def configure_reusable_checkpoint() -> None:
    """Enable reusable checkpoint replay only for our execution-group type."""
    if _ReusableCheckpointHook is not None:
        checkpoint_module._checkpoint_hook = _ReusableCheckpointHook
