"""
This test verifies the correctness of an operator's annotation by running its distributed versions.
The processing pipeline is:
1. generate the input and calculate the output for the operator on a single device
2. construct the partition search space based on its annotation
3. for each partition choice, nnscaler will generate runnable code with communication adapters automatically
4. compare each distributed result with single device version, the difference should be less than a threshold
NOTE: only consider partitioning along one dimension currently
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import subprocess
import torch

from nnscaler.graph.function.dimops import IRDimops, OpAnno, DimAnno
from nnscaler.ir.cten import IRTensor, IRObject


logger = logging.getLogger(__name__)


_SINGLE_GPU_TEST_FILE = "single_gpu_test.py"
_TWO_GPUS_TEST_FILE = "two_gpus_test.py"

module_template_common = """
import os
import numpy
import sys
import torch
import nnscaler

from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType
{import_cumsomized_func}

import nnscaler.graph
import nnscaler.graph.function
import nnscaler.graph.function.wrapnn

import torch
import numpy as np
import random

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, {args}):
        {func_sig_call}

        out = 0
        for output_index, one_out in enumerate([{outputs}]):
            if not isinstance(one_out, torch.Tensor):
                continue
            values = one_out.reshape(-1).float()
            weights = torch.cumsum(torch.ones_like(values), dim=0)
            out += (output_index + 1) * torch.sum(values * weights)
        return out

model = TestModule() #.to(torch.float16)
"""

module_template_single_main = """
# Load inputs from file, ensuring inputs.pt is always a tuple, even when there's only one input
{args}, = torch.load({inputs_path}, map_location=torch.device('cuda:0'))
{clone_args}

model = model.cuda()
{setup_call}

single_loss = model({args})
if single_loss.requires_grad:
    single_loss.backward()

grad_tensors = {grad_tensors}
verification_state = {state_call}
torch.save(
    {{
        'gradients': grad_tensors,
        'loss': single_loss.detach(),
        'state': verification_state,
    }},
    {single_result_path},
)
print('single gpu loss: ', single_loss)
"""

module_template_single = module_template_common + module_template_single_main

module_template_parallel_main = """
nnscaler.init()
rank_id = torch.distributed.get_rank()

{args}, = torch.load({inputs_path}, map_location=torch.device(f'cuda:{{rank_id}}'))
{clone_args}
{setup_call}

def policy(graph: IRGraph, resource) -> IRGraph:
    ngpus = {npartitions}
    partitioned = False

    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if not partitioned and node.signature == '{func_sig}':
            print('Partitioned node: ', node)
            sub_nodes = graph.partition(
                node, node.algorithm('dim'), idx={idx}, dim={dim}, num=ngpus)
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)

    assert partitioned, f'No node is partitioned for {func_sig}.'
    return graph

parallel_model = parallelize(
    model,
    dummy_forward_args={dummy_input_str},
    pas_policy=policy,
    compute_config=ComputeConfig({npartitions}, {npartitions}),
    reuse=ReuseType.OVERRIDE
)

parallel_model.train()
{setup_call}

parallel_loss = parallel_model({args})
if parallel_loss.requires_grad:
    parallel_loss.backward()

grad_tensors = {grad_tensors}
verification_state = {state_call}
torch.save(
    {{
        'gradients': grad_tensors,
        'loss': parallel_loss.detach(),
        'state': verification_state,
    }},
    {parallel_result_prefix} + str(rank_id) + '.pt',
)
print('two gpus loss: ', parallel_loss)
"""

module_template_parallel = module_template_common + module_template_parallel_main


@dataclass
class TensorInfo:
    value_form: str  # 'shape' or 'value'
    value: Union[Tuple[int], Any]
    dtype: torch.dtype = torch.float32
    requires_grad: bool = True

    # make TensorInfo hashable
    def __hash__(self):
        value = self.value
        if isinstance(value, slice):
            value = (value.start, value.stop, value.step)
        return hash((
            self.value_form,
            value,
            self.dtype,
            self.requires_grad,
        ))


@dataclass
class VerifyConfig:
    fsig: str
    args: List[TensorInfo]
    kwargs: Dict[str, Any]
    noutputs: int
    parti_options: List[Dict[str, int]]
    import_customized_func: str = ""
    setup_call: str = ""
    state_call: str = "None"
    non_grad_indices: List[int] = field(default_factory=list)
    workdir: Union[str, Path] = "."


def _complex(val: Any):
    """
    Convert IRObject to concrete value
    NOTE: only used for handling kwargs
    """
    if isinstance(val, tuple):
        return tuple(_complex(t) for t in val)
    if isinstance(val, list):
        return list(_complex(t) for t in val)
    if isinstance(val, dict):
        return {_complex(key): _complex(val) for key, val in val.items()}
    if isinstance(val, slice):
        return slice(_complex(val.start), _complex(val.stop), _complex(val.step))
    if isinstance(val, IRObject):
        assert not isinstance(val, IRTensor), "IRTensor should not be in kwargs"
        return _complex(val.value)
    return val


def get_candidate_options(
    anno: OpAnno, ins_outs_shape: List[TensorInfo], npartitions: int = 2
) -> List[Dict[str, int]]:
    """
    Get all the feasible partitions specified by the annotation of an operator.
    Checks whether the dimension can be divided, and also checks whether the size of the dimension can be evenly divided by the number of partitions
    Args:
        anno (OpAnno): operator annotation
        ins_outs_shape (List[TensorInfo]): input and output shapes
        npartitions (int, optional): number of partitions. Defaults to 2.
    Returns:
        List[Dict[str, int]]: a list of feasible partitions

    """
    all_configs = anno.transform_space()

    candidate_partitions = []
    for idx, dim in all_configs:
        if (
            ins_outs_shape[idx].value_form == "shape"
            and ins_outs_shape[idx].value[dim] % npartitions == 0
        ):
            candidate_partitions.append({"idx": idx, "dim": dim})

    return candidate_partitions


def handle_buffer_parameters(inputs, non_grad_indices):
    """
    Detach specified buffer parameters from the computational graph and disable their gradient computation.
    This is necessary for parameters that should not participate in the backward pass,
    such as statistical parameters in certain layers (e.g., running_mean in normalization layers).

    Args:
        inputs (List[torch.Tensor]): The list of input tensors.
        non_grad_indices (List[int]): The indices of buffer parameters in the input list.
    """
    for idx in non_grad_indices:
        if inputs[idx] is not None:
            inputs[idx] = inputs[idx].detach()
            inputs[idx].requires_grad = False


def _create_op_inputs(verify_config: VerifyConfig) -> List[Any]:
    """
    Create input tensors/non-tensors for the operator.
    The input tensors/non-tensors are only for args, not for kwargs.
    Args:
        verify_config (VerifyConfig): configuration for verifying the partitions
    Returns:
        List[Any]: input tensors
    """
    torch.manual_seed(0)
    inputs = []

    def process_slice(slice_obj):
        start = (
            slice_obj.start.value
            if isinstance(slice_obj.start, IRObject)
            else slice_obj.start
        )
        stop = (
            slice_obj.stop.value
            if isinstance(slice_obj.stop, IRObject)
            else slice_obj.stop
        )
        step = slice_obj.step
        return slice(start, stop, step)

    for i, tensor_info in enumerate(verify_config.args):
        if tensor_info.value_form == "shape":
            # Special handling: For torch. rsqrt, generate random integers between 1 and 10 to avoid invalid values
            if verify_config.fsig == "torch.rsqrt":
                inputs.append(
                    torch.randint(
                        1,
                        10,
                        tensor_info.value,
                        dtype=tensor_info.dtype,
                        requires_grad=tensor_info.requires_grad,
                    )
                )
            # Special handling: for the first parameter of torch.where which is a boolean mask
            elif verify_config.fsig == "torch.where" and i == 0:
                inputs.append(
                    torch.rand(
                        *tensor_info.value, dtype=tensor_info.dtype, requires_grad=tensor_info.requires_grad
                    )
                    > 0.5
                )
            elif verify_config.fsig == "torch.add" and tensor_info.value == (1,):
                # Special handling:add in the model generates values that cannot be partitioned
                inputs.append(torch.randn(4, dtype=tensor_info.dtype, requires_grad=tensor_info.requires_grad))
            else:
                shape = tensor_info.value
                if tensor_info.dtype == torch.bool:
                    inputs.append(torch.rand(shape) > 0.5)
                    continue
                if not (tensor_info.dtype.is_floating_point or tensor_info.dtype.is_complex):
                    inputs.append(
                        torch.randint(0, 10, shape, dtype=tensor_info.dtype)
                    )
                    continue
                if tensor_info.value == ():
                    inputs.append(
                        torch.randn(
                            (), dtype=tensor_info.dtype, requires_grad=tensor_info.requires_grad
                        ).squeeze()
                    )
                else:
                    inputs.append(
                        torch.randn(
                            *tensor_info.value,
                            dtype=tensor_info.dtype,
                            requires_grad=tensor_info.requires_grad,
                        )
                    )
        elif tensor_info.value_form == "value" and isinstance(tensor_info.value, slice):
            inputs.append(process_slice(tensor_info.value))
        else:
            inputs.append(tensor_info.value)
    if verify_config.non_grad_indices:
        handle_buffer_parameters(inputs, verify_config.non_grad_indices)
    return inputs


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _run_checked(command: List[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    import_paths = [
        os.getcwd() if path == "" else path
        for path in sys.path
        if isinstance(path, str)
    ]
    inherited_paths = env.get("PYTHONPATH", "").split(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        path for path in import_paths + inherited_paths if path
    ))
    subprocess.run(command, check=True, cwd=cwd, env=env)


def _assert_close_tree(
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(
                f"{path} type mismatch: Tensor != {type(actual).__name__}"
            )
        if expected.shape != actual.shape:
            raise AssertionError(
                f"{path} shape mismatch: {expected.shape} != {actual.shape}"
            )
        if expected.dtype != actual.dtype:
            raise AssertionError(
                f"{path} dtype mismatch: {expected.dtype} != {actual.dtype}"
            )
        if expected.is_floating_point() or expected.is_complex():
            if not torch.allclose(expected, actual, rtol=rtol, atol=atol):
                max_error = torch.max(torch.abs(expected - actual)).item()
                raise AssertionError(
                    f"{path} mismatch: max absolute error {max_error}"
                )
        elif not torch.equal(expected, actual):
            raise AssertionError(f"{path} tensor values differ")
        return
    if type(expected) is not type(actual):
        raise AssertionError(
            f"{path} type mismatch: "
            f"{type(expected).__name__} != {type(actual).__name__}"
        )
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(
                f"{path} keys mismatch: {expected.keys()} != {actual.keys()}"
            )
        for key in expected:
            _assert_close_tree(
                expected[key], actual[key], path=f"{path}[{key!r}]",
                rtol=rtol, atol=atol,
            )
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path} length mismatch: {len(expected)} != {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            _assert_close_tree(
                expected_item, actual_item, path=f"{path}[{index}]",
                rtol=rtol, atol=atol,
            )
        return
    if expected != actual:
        raise AssertionError(f"{path} mismatch: {expected!r} != {actual!r}")


def verify_partition_options(verify_config: VerifyConfig) -> bool:
    errors = []
    try:
        logger.info(f"Verifying partitions of {verify_config.fsig}...")
        workdir = Path(verify_config.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        inputs = _create_op_inputs(verify_config)
        inputs_path = workdir / f"{verify_config.fsig}_inputs.pt"
        single_result_path = workdir / f"{verify_config.fsig}_loss_single.pt"
        torch.save(inputs, inputs_path)
        logger.info(f"Input tensors saved to {inputs_path}")

        outputs_str = ", ".join([f"_out{i}" for i in range(verify_config.noutputs)])

        kwargs_str = ", ".join(
            [
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={_complex(v)}"
                for k, v in verify_config.kwargs.items()
            ]
        )

        func_sig_call = verify_config.fsig
        args_str = ", ".join([f"_in{i}" for i in range(len(verify_config.args))])
        tensor_member_methods_prefix = 'torch.Tensor.'
        if func_sig_call.startswith(tensor_member_methods_prefix):
            # workaround because tracer does not support tensor member methods
            func_sig_call = f'_in0.' + func_sig_call[len(tensor_member_methods_prefix):]
            func_args_str = ", ".join([f"_in{i}" for i in range(1, len(verify_config.args))])
        else:
            func_args_str = args_str

        if func_args_str:
            func_call = f"{outputs_str} = {func_sig_call}({func_args_str}, {kwargs_str})"
        else:
            func_call = f"{outputs_str} = {func_sig_call}({kwargs_str})"

        clone_args = "\n".join(
            f"_in{i} = _in{i}.detach().clone().requires_grad_"
            f"(_in{i}.requires_grad)"
            for i, tinfo in enumerate(verify_config.args)
            if tinfo.value_form == "shape"
        )

        dummy_input_str = (
            "{"
            + ", ".join([f'"_in{i}": _in{i}' for i in range(len(verify_config.args))])
            + "}"
        )

        grad_tensors = (
            "["
            + ", ".join(
                [
                    f"_in{i}.grad"
                    for i in range(len(verify_config.args))
                    if i not in verify_config.non_grad_indices
                    and verify_config.args[i].value_form == "shape"
                    and verify_config.args[i].requires_grad
                ]
            )
            + "]"
        )
        module_single_str = module_template_single.format(
            import_cumsomized_func=verify_config.import_customized_func,
            clone_args=clone_args,
            args=args_str,
            kwargs=kwargs_str,
            func_sig=verify_config.fsig,
            func_sig_call=func_call,
            outputs=outputs_str,
            grad_tensors=grad_tensors,
            setup_call=verify_config.setup_call,
            state_call=verify_config.state_call,
            inputs_path=repr(str(inputs_path)),
            single_result_path=repr(str(single_result_path)),
        )
        single_gpu_test_file = workdir / _SINGLE_GPU_TEST_FILE
        with single_gpu_test_file.open("w") as f:
            f.write(module_single_str)
        logger.info("Generated test code for single gpu and running...")
        _remove_file(str(single_result_path))
        _run_checked(
            [sys.executable, str(single_gpu_test_file)],
            cwd=workdir,
        )
        logger.info(
            f"Single GPU test completed. Output saved to {single_result_path}"
        )
        logger.info(f"verify_config: {verify_config}")
        logger.info(f"verify_config.parti_options: {verify_config.parti_options}")
        single = torch.load(
            single_result_path, map_location="cpu", weights_only=False
        )

        for poption in verify_config.parti_options:
            try:
                logger.info(f"Verifying the partition {poption}...")
                npartitions = poption.get("num", 2)
                parallel_result_prefix = (
                    workdir / f"{verify_config.fsig}_loss_para_"
                )
                module_para_str = module_template_parallel.format(
                    import_cumsomized_func=verify_config.import_customized_func,
                    clone_args=clone_args,
                    args=args_str,
                    kwargs=kwargs_str,
                    func_sig=verify_config.fsig,
                    func_sig_call=func_call,
                    outputs=outputs_str,
                    dummy_input_str=dummy_input_str,
                    grad_tensors=grad_tensors,
                    idx=poption["idx"],
                    dim=poption["dim"],
                    setup_call=verify_config.setup_call,
                    state_call=verify_config.state_call,
                    inputs_path=repr(str(inputs_path)),
                    npartitions=npartitions,
                    parallel_result_prefix=repr(str(parallel_result_prefix)),
                )
                two_gpus_test_file = workdir / _TWO_GPUS_TEST_FILE
                with two_gpus_test_file.open("w") as f:
                    f.write(module_para_str)
                logger.info("Generated test code for two gpus.")

                para_paths = [
                    Path(f"{parallel_result_prefix}{rank}.pt")
                    for rank in range(npartitions)
                ]
                for path in para_paths:
                    _remove_file(str(path))
                _run_checked(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nnodes=1",
                        f"--nproc_per_node={npartitions}",
                        str(two_gpus_test_file),
                    ],
                    cwd=workdir,
                )
                logger.info(
                    f"Two GPU test completed. Outputs saved to {para_paths}"
                )
                parallel = [
                    torch.load(path, map_location="cpu", weights_only=False)
                    for path in para_paths
                ]

                logger.info(f"Single loss: {single['loss']}")
                for rank, result in enumerate(parallel):
                    logger.info(
                        f"Multi-GPU loss (rank {rank}): {result['loss']}"
                    )
                    _assert_close_tree(
                        single["loss"], result["loss"],
                        path=f"rank {rank} loss",
                    )
                    _assert_close_tree(
                        single["gradients"], result["gradients"],
                        path=f"rank {rank} gradients",
                    )
                    if verify_config.state_call != "None":
                        _assert_close_tree(
                            single["state"], result["state"],
                            path=f"rank {rank} state",
                        )

                logger.info(
                    f"{verify_config.fsig} of partition {poption} passed the allclose comparison."
                )
            except Exception as e:
                error_message = f"Partition {poption} failed with error: {str(e)}"
                logger.error(error_message)
                errors.append(error_message)
        if errors:
            logger.error("Some partitions failed:")
            for error in errors:
                logger.error(error)
            return False
        else:
            logger.info(
                f"Verified all the partitions of {verify_config.fsig} successfully."
            )
            return True
    except Exception as e:
        logger.exception("Exception occurred during verification process")
        raise e
