#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
#
# The gradient-direction context and split-gradient graph traversal are
# adapted from the d9d project for nnScaler.
# Copyright (c) 2026 The d9d Project
# d9d is licensed under the Apache License, Version 2.0.
# https://github.com/d9d-project/d9d

"""d9d-style FBW: preserve the graph and run it twice under a phase context.

Unlike the DeepSeek example, custom operators do not save explicit dW
callbacks. B captures gradients at graph boundary nodes, and W re-enters the
remaining graph from those nodes. The scheduler itself is unchanged.
"""

from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Iterable, Optional

import pytest
import torch
from torch.autograd.graph import GradientEdge, _engine_run_backward

import nnscaler
import nnscaler.runtime.executor as executor
from nnscaler.cli import (
    CheckpointConfig,
    DataloaderConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    Trainer,
    TrainerArgs,
)
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime import _patch_torch_pipelining_backend as pipelining
from nnscaler.runtime.executor import Executor, FBWParamGroup
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


class _GradDirection:
    # These labels are operator semantics, not properties inferred from a
    # Tensor. Each custom Function author must identify activation/weight slots.
    INPUTS = 'inputs'
    WEIGHTS = 'weights'


class _GlobalGradContext:
    """Side channel compensating for stale ctx.needs_input_grad in torch 2.10."""

    # torch 2.10 initializes ctx.needs_input_grad from forward-time
    # requires_grad, so it remains (True, True) in both partial traversals.
    # This process-global context selects the expensive branch explicitly.
    directions = {_GradDirection.INPUTS, _GradDirection.WEIGHTS}

    @classmethod
    def allows(cls, direction: str) -> bool:
        return direction in cls.directions

    @classmethod
    @contextmanager
    def with_directions(cls, *directions: str):
        # try/finally is required because a leaked phase would silently corrupt
        # all later backward calls in this worker process.
        previous = cls.directions
        cls.directions = set(directions)
        try:
            yield
        finally:
            cls.directions = previous

    @classmethod
    def reset(cls) -> None:
        cls.directions = {_GradDirection.INPUTS, _GradDirection.WEIGHTS}


@dataclass
class _D9DBackwardState:
    # param_groups=None means there was no grad-requiring stage input, so no B
    # traversal ran and W must own the full backward. An empty list instead is
    # a paired no-op state (for example, a segment with no output tensors).
    param_groups: Optional[list[FBWParamGroup]] = None
    # This field is intentionally never read. Holding the output tensors keeps
    # Python autograd.Function's underlying C++ graph and saved tensors alive
    # until this state is popped and W finishes.
    graph_owners: tuple[torch.Tensor, ...] = ()
    output_tensors: tuple[torch.Tensor, ...] = ()
    output_tensor_grads: tuple[Optional[torch.Tensor], ...] = ()


class _D9DStateStore:
    """FIFO pairing each generated B call with the corresponding W call."""

    # The same segment is invoked for multiple microbatches. A single state per
    # name would be overwritten before a delayed W consumes it.
    queues: dict[str, deque[_D9DBackwardState]] = defaultdict(deque)

    @classmethod
    def put(cls, name: str, state: _D9DBackwardState) -> None:
        cls.queues[name].append(state)

    @classmethod
    def pop(cls, name: str) -> _D9DBackwardState:
        assert cls.queues[name], f'No deferred weight backward for segment {name}'
        return cls.queues[name].popleft()

    @classmethod
    def clear(cls) -> None:
        cls.queues = defaultdict(deque)

    @classmethod
    def check_clear(cls) -> None:
        assert all(not queue for queue in cls.queues.values())


class _OutputGradStats:
    none_grads = 0
    tensor_grads = 0

    @classmethod
    def record(cls, grads: Iterable[Optional[torch.Tensor]]) -> None:
        for grad in grads:
            if grad is None:
                cls.none_grads += 1
            else:
                assert torch.is_tensor(grad)
                cls.tensor_grads += 1

    @classmethod
    def clear(cls) -> None:
        cls.none_grads = 0
        cls.tensor_grads = 0


@nnscaler.register_op('a b, b c -> a c')
class _D9DLinearFunction(torch.autograd.Function):
    """Opaque Linear that gates dX/dW using the semantic phase context."""

    # Counters prove both phases executed; checkpoint parity proves their values.
    input_grad_computations = 0
    weight_grad_computations = 0

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor, weight)
        return input_tensor @ weight

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        input_tensor, weight = ctx.saved_tensors
        input_grad = weight_grad = None

        if (
            ctx.needs_input_grad[0]
            and _GlobalGradContext.allows(_GradDirection.INPUTS)
        ):
            # B computes dX so it can be sent to the previous pipeline stage.
            input_grad = output_grad @ weight.T
            _D9DLinearFunction.input_grad_computations += 1

        if (
            ctx.needs_input_grad[1]
            and _GlobalGradContext.allows(_GradDirection.WEIGHTS)
        ):
            # W revisits this same PyNode but computes only dW.
            weight_grad = input_tensor.T @ output_grad
            _D9DLinearFunction.weight_grad_computations += 1

        return input_grad, weight_grad

    @classmethod
    def clear_counts(cls) -> None:
        cls.input_grad_computations = 0
        cls.weight_grad_computations = 0


class _D9DLinear(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim, dim))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _D9DLinearFunction.apply(input_tensor, self.weight)


class _D9DModel(torch.nn.Module):
    def __init__(self, dim: int = 8, nlayers: int = 4):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([_D9DLinear(dim) for _ in range(nlayers)])

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        output = sample['input']
        for layer in self.layers:
            output = torch.tanh(layer(output))
        return torch.nn.functional.mse_loss(output, sample['target'])


class _D9DDataset:
    def __init__(self, dim: int = 8, size: int = 16):
        generator = torch.Generator().manual_seed(1)
        self.input = torch.randn(size, dim, generator=generator)
        self.target = torch.randn(size, dim, generator=generator)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {'input': self.input[index], 'target': self.target[index]}

    def __len__(self) -> int:
        return len(self.input)


def _deduplicate_outputs(
    output_tensors: Iterable[torch.Tensor],
    output_tensor_grads: Iterable[Optional[torch.Tensor]],
) -> tuple[list[torch.Tensor], list[Optional[torch.Tensor]]]:
    # Match Executor.backward: generated graphs may expose the same
    # (output, grad) pair more than once, which must not double-count gradients.
    visited = set()
    deduplicated_outputs = []
    deduplicated_grads = []
    for tensor, grad in zip(output_tensors, output_tensor_grads):
        pair = (id(tensor), id(grad))
        if pair in visited:
            continue
        visited.add(pair)
        deduplicated_outputs.append(tensor)
        deduplicated_grads.append(grad)
    return deduplicated_outputs, deduplicated_grads


def _capture_param_groups(
    outputs: list[torch.Tensor],
    inputs: list[torch.Tensor],
    weights: tuple[torch.nn.Parameter, ...],
) -> list[FBWParamGroup]:
    # Parameter entries are AccumulateGrad nodes. Intermediates are the boundary
    # where the input-gradient closure intersects each parameter closure. B
    # captures gradients there; W starts from the same nodes later.
    output_nodes = list(filter(None, map(pipelining._get_grad_fn_or_grad_acc, outputs)))
    input_nodes = list(filter(None, map(pipelining._get_grad_fn_or_grad_acc, inputs)))
    weight_nodes = list(filter(None, map(pipelining._get_grad_fn_or_grad_acc, weights)))
    reverse_graph = pipelining.construct_reverse_graph(output_nodes)
    return cast(
        list[FBWParamGroup],
        pipelining.get_param_groups(input_nodes, weight_nodes, reverse_graph),
    )


def _d9d_backward_input(
    name: str,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor],
    output_tensor_grads: list[Optional[torch.Tensor]],
    weights: Iterable[torch.nn.Parameter],
):
    # Record before sync_tensors to observe exactly what generated code passed:
    # loss contributes None, while an inter-stage edge contributes a Tensor.
    _OutputGradStats.record(output_tensor_grads)
    # Resolve async stage-boundary gradients before autograd consumes them.
    output_tensor_grads = executor.sync_tensors(output_tensor_grads)
    weights = tuple(weights)

    # This example reimplements Executor.backward_input because the stock helper
    # detaches stage outputs and its public torch.autograd.grad W path cannot
    # handle a Python autograd.Function boundary on torch 2.10.
    saved_pairs = Executor._detach[name].pop(0)
    input_ids = {id(tensor) for tensor in input_tensors}
    saved_input_ids = {tensor_id for tensor_id, _ in saved_pairs}
    assert input_ids.issubset(saved_input_ids)

    if not output_tensors:
        # The generated scheduler still emits a matching W, so publish a no-op
        # state instead of omitting the FIFO entry.
        # this is a corner case. No output tensors means no gradients to propagate back.
        _D9DStateStore.put(name, _D9DBackwardState(param_groups=[]))
        return None

    inputs = [
        tensor
        for _, tensor in saved_pairs
        if torch.is_tensor(tensor) and tensor.requires_grad
    ]
    outputs, output_grads = _deduplicate_outputs(
        output_tensors, output_tensor_grads
    )

    if Executor._backward_pre_hook is not None:
        inputs, outputs, output_grads = Executor._backward_pre_hook(
            inputs, outputs, output_grads
        )

    if not inputs:
        # No dX crosses this stage boundary (normally the first stage). Delay the
        # whole backward to W rather than constructing an empty input closure.
        _D9DStateStore.put(
            name,
            _D9DBackwardState(
                output_tensors=tuple(outputs),
                output_tensor_grads=tuple(output_grads),
            ),
        )
        return None

    param_groups = _capture_param_groups(outputs, inputs, weights)
    handles = []
    try:
        for param_group in param_groups:
            intermediates = list(param_group['intermediates'])
            param_group['intermediates'] = intermediates
            for index, intermediate in enumerate(intermediates):
                def capture(grads, group=param_group, grad_index=index):
                    # A Node pre-hook receives one tuple entry per Node output,
                    # after the engine has accumulated all contributions that
                    # reach those outputs and before the Node executes. Returning
                    # None leaves that tuple unchanged, so B only records the
                    # canonical boundary gradient. Preserve every tuple entry:
                    # W maps entry i back to GradientEdge(intermediate, i).
                    captured_grads = group.get('grads')
                    if captured_grads is None:
                        captured_grads = [None] * len(group['intermediates'])
                        group['grads'] = captured_grads
                    captured_grads[grad_index] = grads

                handles.append(intermediate.register_prehook(capture))

        # `inputs=inputs` asks the engine for a partial traversal. Native C++ ops
        # prune dW automatically; the phase context gives Python Functions the
        # equivalent signal. retain_graph is required by the later W traversal.
        with _GlobalGradContext.with_directions(_GradDirection.INPUTS):
            torch.autograd.backward(
                outputs,
                grad_tensors=output_grads,
                inputs=inputs,
                retain_graph=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    grads = tuple(tensor.grad for tensor in inputs)
    assert all(grad is not None for grad in grads)
    _D9DStateStore.put(
        name,
        _D9DBackwardState(
            param_groups=param_groups,
            # Do not detach these outputs. Earlier attempts retained only the
            # Python PyNode, which does not own its underlying C++ graph.
            graph_owners=tuple(outputs),
        ),
    )

    if len(grads) == 1:
        return grads[0]
    return grads


def _run_weight_backward(
    weights: tuple[torch.nn.Parameter, ...],
    param_groups: list[FBWParamGroup],
) -> None:
    # Map the graph's AccumulateGrad nodes back to leaf Parameters. Passing the
    # leaves to the engine with accumulate_grad=True fires reducer hooks.
    grad_acc_to_weight = {
        pipelining._get_grad_fn_or_grad_acc(weight): weight for weight in weights
    }

    for param_group in param_groups:
        captured_grads = param_group.get('grads')
        assert captured_grads is not None
        assert all(grads is not None for grads in captured_grads)
        completed_grads = cast(
            list[tuple[Optional[torch.Tensor], ...]], captured_grads
        )
        valid_edges = []
        valid_grad_outputs = []
        for grads, intermediate in zip(
            completed_grads, param_group['intermediates']
        ):
            for output_index, grad in enumerate(grads):
                if grad is not None:
                    # Do not hard-code output_nr=0 or sum multi-output grads;
                    # doing so silently produces the wrong dW for opaque ops.
                    valid_edges.append(GradientEdge(intermediate, output_index))
                    # grad the output gradient of the intermediate node at this output index.
                    valid_grad_outputs.append(grad)

        handles = []
        try:
            if len(param_group['intermediates']) > 1:
                # Every valid boundary edge above is a root in the same W
                # GraphTask. Boundaries can overlap along one path, for example:
                #
                #     weight -> ancestor -> descendant -> loss
                #
                # If both nodes are boundaries, B's saved ancestor gradient
                # already includes the contribution propagated from descendant.
                # W nevertheless seeds both roots. Without these hooks,
                # descendant propagates to ancestor again, and the engine adds
                # that contribution to ancestor's own seed before executing it.
                # The ancestor path, and therefore dW, is then counted twice.
                #
                # A Node pre-hook receives this already-accumulated gradient
                # tuple. Replacing it with B's saved canonical tuple clamps each
                # boundary and prevents roots from changing one another. The
                # default argument binds this loop iteration's tuple instead of
                # late-binding the final `grads` value. See the standalone
                # `repro_fbw_boundary_clamp.py` for a 6 -> 12 -> 6 example.
                for grads, intermediate in zip(
                    completed_grads,
                    param_group['intermediates'],
                    strict=True,
                ):
                    handles.append(
                        intermediate.register_prehook(
                            lambda grad_outputs, saved_grads=grads: saved_grads
                        )
                    )

            if valid_edges:
                weight_inputs = tuple(
                    grad_acc_to_weight[node]
                    for node in param_group['params']
                    if node in grad_acc_to_weight
                )
                with _GlobalGradContext.with_directions(_GradDirection.WEIGHTS):
                    # The public autograd.grad/backward wrappers inspect
                    # GradientEdge.node._input_metadata. Python _FunctionBase
                    # rejects that legacy access in torch 2.10, so call the
                    # engine after B has already validated/captured these grads.
                    # accumulate_grad=True is deliberate: returning dW and then
                    # assigning weight.grad would bypass AccumulateGrad reducers.
                    # torch 2.10's private engine signature is:
                    #   run_backward(tensors, grad_tensors, keep_graph,
                    #                create_graph, inputs, allow_unreachable,
                    #                accumulate_grad)
                    _engine_run_backward(
                        # Boundary output edges that act as W's backward roots.
                        tuple(valid_edges),
                        # One B-captured VJP seed for each boundary root.
                        tuple(valid_grad_outputs),
                        # W consumes this retained graph; it is not reused.
                        keep_graph=False,
                        # Training needs first-order dW, not a higher-order graph.
                        create_graph=False,
                        # Restrict requested leaf outputs to this parameter group.
                        inputs=weight_inputs,
                        # This flag controls unused-input errors only on the
                        # autograd.grad path. It is ignored here because
                        # accumulate_grad=True selects backward API behavior.
                        allow_unreachable=False,
                        # Execute the leaf AccumulateGrad nodes, writing `.grad`
                        # and firing nnScaler reducer hooks instead of returning dW.
                        accumulate_grad=True,
                    )
        finally:
            for handle in handles:
                handle.remove()


def _d9d_backward_weight(
    name: str,
    weights: Iterable[torch.nn.Parameter],
) -> None:
    weights = tuple(weights)
    # Keeping `state` local also keeps graph_owners alive through this W call.
    state = _D9DStateStore.pop(name)

    if state.param_groups is not None:
        if weights and state.param_groups:
            _run_weight_backward(weights, state.param_groups)
        return

    if weights:
        # With no stage-input gradient to send, W owns the complete backward.
        # Activation gradients are still needed between operators in this stage.
        # A previous W-only implementation left early-stage weights unchanged:
        # later operators could compute dW, but their dX was blocked before it
        # reached earlier operators. Therefore both directions are enabled here.
        with _GlobalGradContext.with_directions(
            _GradDirection.INPUTS,
            _GradDirection.WEIGHTS,
        ):
            torch.autograd.backward(
                state.output_tensors,
                grad_tensors=state.output_tensor_grads,
            )


def _trainer_args(work_dir: Path, mode: str) -> TrainerArgs:
    # d9d_async verifies that synchronizing at B entry is sufficient before
    # graph state is retained for the delayed W traversal.
    use_fbw = mode != 'baseline'
    return TrainerArgs(
        instance_name=f'customized_fbw_d9d_{mode}',
        compute_config=ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=2,
            use_end2end=True,
            use_fbw=use_fbw,
            use_async_comm=mode == 'd9d_async',
            pas_config={
                'pipeline_nmicros': 2,
                'pipeline_nstages': 2,
                'pipeline_scheduler': '1f1b',
            },
        ),
        gen_reuse='override',
        gen_savedir=work_dir / 'gen',
        pas_policy='hybrid',
        model=ModelConfig(type=_D9DModel),
        optimizer=OptimizerConfig(type=torch.optim.Adam, args={'lr': 0.01}),
        dataset=DatasetConfig(type=_D9DDataset, train_args={'size': 16}),
        dataloader=DataloaderConfig(train_args={'drop_last': True}),
        checkpoint=CheckpointConfig(
            save_dir=work_dir / 'checkpoints',
            save_type='deduped',
            save_best=False,
        ),
        micro_batch_size=2,
        global_batch_size=4,
        max_train_steps=2,
        val_every_n_epochs=None,
        enable_progress_bar=False,
        log_progress_every_n_train_steps=None,
        seed=0,
    )


def _d9d_worker(root_dir: str, mode: str) -> Optional[tuple[int, int]]:
    work_dir = Path(root_dir) / mode
    _D9DStateStore.clear()
    _D9DLinearFunction.clear_counts()
    _OutputGradStats.clear()
    _GlobalGradContext.reset()
    patch = executor.custom_fbw(
        _d9d_backward_input,
        _d9d_backward_weight,
    ) if mode != 'baseline' else nullcontext()

    with patch:
        trainer = Trainer(train_args=_trainer_args(work_dir, mode))
        trainer.run()

    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 2

    output_grad_counts = None
    if mode != 'baseline':
        _D9DStateStore.check_clear()
        counts = torch.tensor(
            [
                _D9DLinearFunction.input_grad_computations,
                _D9DLinearFunction.weight_grad_computations,
                _OutputGradStats.none_grads,
                _OutputGradStats.tensor_grads,
            ],
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(counts)
        assert counts[0].item() > 0
        assert counts[1].item() > 0
        # The last stage seeds scalar-loss backward with None; an earlier stage
        # receives a real gradient tensor from the next stage.
        assert counts[2].item() > 0
        assert counts[3].item() > 0
        output_grad_counts = (counts[2].item(), counts[3].item())

    if trainer.rank == 0:
        checkpoint_files = list((work_dir / 'checkpoints' / 'last').glob('*.ckpt'))
        Trainer.merge_checkpoint(checkpoint_files, Path(root_dir) / f'{mode}.pt')
    torch.distributed.barrier()
    return output_grad_counts


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='lack of gpu devices',
)
def test_d9d_customized_fbw(tmp_path):
    launch_torchrun(2, _d9d_worker, str(tmp_path), 'baseline')
    d9d_counts = launch_torchrun(2, _d9d_worker, str(tmp_path), 'd9d')
    d9d_async_counts = launch_torchrun(2, _d9d_worker, str(tmp_path), 'd9d_async')

    # Every rank receives the same all-reduced pair: (None grads, Tensor grads).
    assert all(none_grads > 0 and tensor_grads > 0
               for none_grads, tensor_grads in d9d_counts.values())
    assert all(none_grads > 0 and tensor_grads > 0
               for none_grads, tensor_grads in d9d_async_counts.values())

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    d9d = torch.load(tmp_path / 'd9d.pt', weights_only=False)
    d9d_async = torch.load(tmp_path / 'd9d_async.pt', weights_only=False)
    # Optimizer-state parity catches missing or duplicated dW that a short model
    # parameter comparison could fail to expose clearly.
    assert_equal(baseline['model'], d9d['model'])
    assert_equal(baseline['optimizer'], d9d['optimizer'])
    assert_equal(baseline['model'], d9d_async['model'])
    assert_equal(baseline['optimizer'], d9d_async['optimizer'])
