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
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
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
        # Add clone to resolve the issue:
        # a leaf Variable that requires grad is being used in an in-place operation.
        {clone_args}

        {func_sig_call}

        out = 0
        for one_out in [{outputs}]:
            if not isinstance(one_out, torch.Tensor):
                continue
            out += torch.sum(one_out)
        return out

model = TestModule() #.to(torch.float16)
"""

module_template_single_main = """
# Load inputs from file, ensuring inputs.pt is always a tuple, even when there's only one input
{args}, = torch.load('{func_sig}_inputs.pt', map_location=torch.device('cuda:0'))

model = model.cuda()

single_loss = model({args})
single_loss.backward()

grad_tensors = {grad_tensors}
torch.save([grad_tensors, single_loss], '{func_sig}_loss_single.pt')
print('single gpu loss: ', single_loss)
"""

module_template_single = module_template_common + module_template_single_main

module_template_parallel_main = """
nnscaler.init()
rank_id = torch.distributed.get_rank()

{args}, = torch.load('{func_sig}_inputs.pt', map_location=torch.device(f'cuda:{{rank_id}}'))

def policy(graph: IRGraph, resource) -> IRGraph:
    ngpus = 2
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
    compute_config=ComputeConfig(2, 2),
    reuse=ReuseType.OVERRIDE
)

parallel_model.train()

parallel_loss = parallel_model({args})
parallel_loss.backward()

grad_tensors = {grad_tensors}
torch.save([grad_tensors, parallel_loss], '{func_sig}_loss_para_'+str(rank_id)+'.pt')
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
        return hash((self.value_form, value))


@dataclass
class VerifyConfig:
    fsig: str
    args: List[TensorInfo]
    kwargs: Dict[str, Any]
    noutputs: int
    parti_options: List[Dict[str, int]]
    import_customized_func: str = ""
    non_grad_indices: List[int] = field(default_factory=list)


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


def verify_partition_options(verify_config: VerifyConfig) -> bool:
    errors = []
    try:
        logger.info(f"Verifying partitions of {verify_config.fsig}...")
        inputs = _create_op_inputs(verify_config)
        torch.save(inputs, f"{verify_config.fsig}_inputs.pt")
        logger.info(f"Input tensors saved to {verify_config.fsig}_inputs.pt")

        outputs_str = ", ".join([f"_out{i}" for i in range(verify_config.noutputs)])

        kwargs_str = ", ".join(
            [
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={_complex(v)}"
                for k, v in verify_config.kwargs.items()
            ]
        )
        args_str = ", ".join([f"_in{i}" for i, tinfo in enumerate(verify_config.args)])
        func_sig_call = verify_config.fsig

        if args_str:
            func_call = f"{outputs_str} = {func_sig_call}({args_str}, {kwargs_str})"
        else:
            func_call = f"{outputs_str} = {func_sig_call}({kwargs_str})"

        clone_args_right = ", ".join(
            [
                f"_in{i}.clone()"
                for i, tinfo in enumerate(verify_config.args)
                if tinfo.value_form == "shape"
            ]
        )
        if clone_args_right:
            clone_args_left = ", ".join(
                [
                    f"_in{i}"
                    for i, tinfo in enumerate(verify_config.args)
                    if tinfo.value_form == "shape"
                ]
            )
            clone_args = f"{clone_args_left} = {clone_args_right}"
        else:
            clone_args = ""

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
        )
        with open(_SINGLE_GPU_TEST_FILE, "w") as f:
            f.write(module_single_str)
        logger.info("Generated test code for single gpu and running...")
        subprocess.run(["rm", "-f", f"{verify_config.fsig}_loss_single.pt"])
        subprocess.run(["python", _SINGLE_GPU_TEST_FILE])
        logger.info(
            f"Single GPU test completed. Output saved to {verify_config.fsig}_loss_single.pt"
        )
        logger.info(f"verify_config: {verify_config}")
        logger.info(f"verify_config.parti_options: {verify_config.parti_options}")

        for poption in verify_config.parti_options:
            try:
                logger.info(f"Verifying the partition {poption}...")
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
                )
                with open(_TWO_GPUS_TEST_FILE, "w") as f:
                    f.write(module_para_str)
                logger.info("Generated test code for two gpus.")

                subprocess.run(["rm", "-f", f"{verify_config.fsig}_loss_para_0.pt"])
                subprocess.run(["rm", "-f", f"{verify_config.fsig}_loss_para_1.pt"])
                subprocess.run(
                    [
                        "torchrun",
                        "--nproc_per_node=2",
                        "--nnodes=1",
                        "--rdzv-endpoint=localhost:23457",
                        _TWO_GPUS_TEST_FILE,
                    ]
                )
                logger.info(
                    f"Two GPU test completed. Outputs saved to {verify_config.fsig}_loss_para_0.pt and {verify_config.fsig}_loss_para_1.pt"
                )
                single = torch.load(f"{verify_config.fsig}_loss_single.pt")
                logger.info(
                    f"Loading single loss from: {verify_config.fsig}_loss_single.pt"
                )
                para0 = torch.load(f"{verify_config.fsig}_loss_para_0.pt")
                para1 = torch.load(f"{verify_config.fsig}_loss_para_1.pt")

                logger.info(f"Single loss: {single[1]}")
                logger.info(f"Multi-GPU loss (para0): {para0[1]}")
                logger.info(f"Multi-GPU loss (para1): {para1[1]}")

                assert torch.allclose(
                    single[1], para0[1], rtol=1e-3, atol=1e-5
                ), f"Loss mismatch between single and multi-GPU (para0)"
                assert torch.equal(
                    para0[1], para1[1].to(para0[1])
                ), f"Loss mismatch between multi-GPU (para0 and para1)"

                for i in range(len(single[0])):
                    if single[0][i] is None or para0[0][i] is None:
                        logger.debug(
                            f"Skipping comparison for index {i} because it is None"
                        )
                        continue
                    logger.debug(f"Absolute error: {single[0][i] - para0[0][i]}")
                    logger.debug(
                        f"Relative error: {(single[0][i] - para0[0][i]) / single[0][i]}"
                    )
                    assert torch.allclose(
                        single[0][i], para0[0][i], rtol=1e-3, atol=1e-5
                    ), f"Gradient mismatch between single and multi-GPU (para0)"
                    assert torch.equal(
                        para0[0][i], para1[0][i].to(para0[0][i])
                    ), f"Gradient mismatch between multi-GPU (para0 and para1)"

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
