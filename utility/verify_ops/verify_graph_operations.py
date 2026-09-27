import argparse
import importlib
import os
import sys
import torch
from nnscaler.graph.function.dimops import DimAnno, IRDimops, OpAnno
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.cten import IRObject, IRTensor
from nnscaler.graph.parser.register import CustomizedOps
from pathlib import Path
import logging

from verify_dimops import TensorInfo, get_candidate_options

_VERIFIED_OPS_FILE_NAME = "verified_ops.pt"
_DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/nnscaler"))


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_verified_ops(outdir: Path):
    verified_ops_file = outdir / _VERIFIED_OPS_FILE_NAME
    if verified_ops_file.exists():
        logger.info(f"{verified_ops_file} exists, load it.")
        return torch.load(verified_ops_file, weights_only=False)
    else:
        logger.info(f"{verified_ops_file} does not exist, start from scratch.")
        return set()


def save_verified_ops(outdir: Path, verified_ops: set):
    verified_ops_file = outdir / _VERIFIED_OPS_FILE_NAME
    torch.save(verified_ops, verified_ops_file)
    logger.info(f"Verification results saved to {verified_ops_file}")


def _import_customized_op(signature: str) -> None:
    if CustomizedOps.exist(signature):
        return
    if signature.startswith((
        "torch.",
        "_operator.",
        "nnscaler.runtime.function.",
    )):
        return
    parts = signature.split(".")
    for end in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:end])
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(
                f"{exc.name}."
            ):
                continue
            raise
        if CustomizedOps.exist(signature):
            return


def _callable_expression(callback) -> str:
    return f"{callback.__module__}.{callback.__qualname__}"


def _verification_code(signature: str):
    _import_customized_op(signature)
    if not CustomizedOps.exist(signature):
        return "", "", "None"

    runtime_fn = CustomizedOps.kOpRuntime[signature]
    verification = CustomizedOps.kOpVerification.get(signature)
    callbacks = []
    if verification is not None:
        callbacks = [
            callback
            for callback in (verification.setup_fn, verification.state_fn)
            if callback is not None
        ]

    modules = {runtime_fn.__module__}
    modules.update(callback.__module__ for callback in callbacks)
    import_code = "\n".join(
        f"import {module_name}" for module_name in sorted(modules)
    )
    setup_call = (
        f"{_callable_expression(verification.setup_fn)}()"
        if verification is not None and verification.setup_fn is not None
        else ""
    )
    state_call = (
        f"{_callable_expression(verification.state_fn)}()"
        if verification is not None and verification.state_fn is not None
        else "None"
    )
    return import_code, setup_call, state_call


def verify_op_partitions(graph: IRGraph, outdir: Path) -> bool:
    """
    Test if the partitioned ops in the graph are computationally correct.

    Args:
        graph (IRGraph): the graph to be verified
        outdir (Path): the directory to save the verified ops

    Returns:
        bool: whether all partition contracts were verified successfully.
    """
    from verify_dimops import (
        VerifyConfig,
        TensorInfo,
        verify_partition_options,
    )

    verified_ops = load_verified_ops(outdir)
    skipped_nodes = []
    failed_nodes = []

    gnodes = graph.nodes(flatten=True)
    for idx, node in enumerate(gnodes):
        logger.info(f"node: {node}")
        logger.info(f"Verification progress: {idx} / {len(gnodes)}")
        if node.isfw() and isinstance(node, IRDimops):
            ins_info = [
                (
                    TensorInfo(
                        "shape",
                        _input.shape,
                        dtype=_input.dtype or torch.float32,
                        requires_grad=_input.requires_grad,
                    )
                    if isinstance(_input, IRTensor)
                    else TensorInfo(
                        "value",
                        _input.value if isinstance(_input, IRObject) else _input,
                    )
                )
                for _input in node.inputs()
            ]
            if not ins_info:
                skipped_nodes.append(f"{node.signature} (type: {type(node)})")
                logger.info(f"ins_info is empty for node: {node.signature}, skipping.")
                continue

            outs_info = [
                (
                    TensorInfo(
                        "shape",
                        output.shape,
                        dtype=output.dtype or torch.float32,
                        requires_grad=output.requires_grad,
                    )
                    if isinstance(output, IRTensor)
                    else TensorInfo(
                        "value",
                        output.value if isinstance(output, IRObject) else output,
                    )
                )
                for output in node.outputs()
            ]
            verification_key = (
                node.signature,
                repr(node.anno),
                tuple(ins_info + outs_info),
                repr(node.kwargs),
            )
            if verification_key in verified_ops:
                logger.info(f"{node.signature} has been verified before, skip.")
                continue

            logger.info(f"Node annos: {node.signature}, {node.anno}")

            parti_options = get_candidate_options(node.anno, ins_info + outs_info)

            logger.info(f"Candidate partition options: {parti_options}")

            try:
                import_code, setup_call, state_call = _verification_code(
                    node.signature
                )
                verify_config = VerifyConfig(
                    fsig=node.signature,
                    args=ins_info,
                    kwargs=node.kwargs,
                    noutputs=len(node.outputs()),
                    parti_options=parti_options,
                    import_customized_func=import_code,
                    setup_call=setup_call,
                    state_call=state_call,
                )
                iscorrect = verify_partition_options(verify_config)
            except Exception as e:
                logger.exception(
                    f"Verification could not run for {node.signature}: {e}"
                )
                failed_nodes.append(node.signature)
                continue
            if not iscorrect:
                logger.error(f"Verification failed for {node.signature}.")
                failed_nodes.append(node.signature)
                continue

            verified_ops.add(verification_key)
            save_verified_ops(outdir, verified_ops)

    if skipped_nodes:
        logger.info("Skipped the following nodes due to empty ins_info:")
        for node_info in skipped_nodes:
            logger.info(f" - {node_info}")
    if failed_nodes:
        logger.error(
            "Partition verification failed for: %s",
            ", ".join(sorted(set(failed_nodes))),
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify partitions of operations in an IRGraph."
    )
    parser.add_argument(
        "--graph", type=str, required=True, help="Path to the graph file."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Optional directory to save the verified operations. If not provided, results will be saved to the default cache directory.",
    )

    args = parser.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file {graph_path} does not exist.")

    graph = IRGraph.load(graph_path)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = _DEFAULT_CACHE_DIR

    outdir.mkdir(parents=True, exist_ok=True)
    if not verify_op_partitions(graph, outdir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
