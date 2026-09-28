import argparse
import hashlib
import importlib
import inspect
import marshal
import os
import sys
from enum import Enum
import torch
from typing import Dict, List, Optional, Sequence, Tuple
from nnscaler.graph.function.dimops import DimAnno, IRDimops, OpAnno
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.cten import IRObject, IRTensor
from nnscaler.graph.parser.register import CustomizedOps
from pathlib import Path
import logging

from nnscaler.graph.verification_runner import TensorInfo, get_candidate_options

_VERIFIED_OPS_FILE_NAME = "verified_ops.pt"
_DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/nnscaler"))


logger = logging.getLogger(__name__)


class AnnotationVerification(Enum):
    OFF = "off"
    STATIC = "static"
    USED = "used"
    ALL = "all"


PartitionOptions = Sequence[Tuple[IRDimops, List[Dict[str, int]]]]


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


def _callable_fingerprint(callback) -> Optional[str]:
    if callback is None:
        return None
    try:
        content = inspect.getsource(callback).encode()
    except (OSError, TypeError):
        code = getattr(callback, "__code__", None)
        if code is not None:
            content = marshal.dumps(code)
        else:
            content = (
                f"{callback.__module__}.{callback.__qualname__}"
            ).encode()
    return hashlib.sha256(content).hexdigest()


def _verification_fingerprint(signature: str) -> Optional[Tuple]:
    _import_customized_op(signature)
    if not CustomizedOps.exist(signature):
        return None

    verification = CustomizedOps.kOpVerification.get(signature)
    return (
        hashlib.sha256(
            CustomizedOps.kOpCodeDef[signature].encode()
        ).hexdigest(),
        _callable_fingerprint(CustomizedOps.kOpRuntime[signature]),
        _callable_fingerprint(CustomizedOps.kOpFakeRuntime.get(signature)),
        _callable_fingerprint(CustomizedOps.kOpEmit.get(signature)),
        _callable_fingerprint(
            verification.setup_fn if verification is not None else None
        ),
        _callable_fingerprint(
            verification.state_fn if verification is not None else None
        ),
    )


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

    modules = set()
    runtime_module = getattr(runtime_fn, "__module__", None)
    if runtime_module is not None:
        modules.add(runtime_module)
    runtime_owner = getattr(runtime_fn, "__self__", None)
    runtime_owner_module = getattr(runtime_owner, "__module__", None)
    if runtime_owner_module is not None:
        modules.add(runtime_owner_module)
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


def _node_infos(node: IRDimops) -> Tuple[List[TensorInfo], List[TensorInfo]]:
    ins_info = [
        (
            TensorInfo(
                "shape",
                input_.shape,
                dtype=input_.dtype or torch.float32,
                requires_grad=input_.requires_grad,
            )
            if isinstance(input_, IRTensor)
            else TensorInfo(
                "value",
                input_.value if isinstance(input_, IRObject) else input_,
            )
        )
        for input_ in node.inputs()
    ]
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
    return ins_info, outs_info


def _nodes_and_options(
    graph: IRGraph,
    partition_options: Optional[PartitionOptions],
) -> List[Tuple[IRDimops, List[Dict[str, int]]]]:
    if partition_options is not None:
        return list(partition_options)

    result = []
    for node in graph.nodes(flatten=True):
        if node.isfw() and isinstance(node, IRDimops):
            ins_info, outs_info = _node_infos(node)
            result.append((
                node,
                get_candidate_options(node.anno, ins_info + outs_info),
            ))
    return result


def validate_graph_annotations(
    graph: IRGraph,
    partition_options: Optional[PartitionOptions] = None,
) -> None:
    """Validate that annotation-declared partitions satisfy the dim algorithm."""
    errors = []
    for node, options in _nodes_and_options(graph, partition_options):
        for option in options:
            try:
                satisfied = node.algorithm("dim").satisfy(
                    idx=option["idx"],
                    dim=option["dim"],
                    num=option.get("num", 2),
                )
                if not satisfied:
                    raise ValueError("partition algorithm rejected the option")
            except Exception as exc:
                errors.append(
                    f"{node.signature} partition {option} is invalid: {exc}"
                )
    if errors:
        raise ValueError("Annotation validation failed:\n" + "\n".join(errors))


def verify_op_partitions(
    graph: IRGraph,
    outdir: Path,
    partition_options: Optional[PartitionOptions] = None,
) -> bool:
    """
    Test if the partitioned ops in the graph are computationally correct.

    Args:
        graph (IRGraph): the graph to be verified
        outdir (Path): the directory to save the verified ops

    Returns:
        bool: whether all partition contracts were verified successfully.
    """
    from nnscaler.graph.verification_runner import (
        VerifyConfig,
        TensorInfo,
        verify_partition_options,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    verified_ops = load_verified_ops(outdir)
    skipped_nodes = []
    failed_nodes = []

    nodes_and_options = _nodes_and_options(graph, partition_options)
    for idx, (node, parti_options) in enumerate(nodes_and_options):
        logger.info(f"node: {node}")
        logger.info(
            f"Verification progress: {idx} / {len(nodes_and_options)}"
        )
        ins_info, outs_info = _node_infos(node)
        if not ins_info:
            skipped_nodes.append(f"{node.signature} (type: {type(node)})")
            logger.info(f"ins_info is empty for node: {node.signature}, skipping.")
            continue
        if not parti_options:
            logger.info(
                f"No feasible partition options for {node.signature}, skipping."
            )
            continue

        partition_key = tuple(
            (option["idx"], option["dim"], option.get("num", 2))
            for option in parti_options
        )
        verification_key = (
            node.signature,
            repr(node.anno),
            tuple(ins_info + outs_info),
            repr(node.kwargs),
            partition_key,
            _verification_fingerprint(node.signature),
        )
        if verification_key in verified_ops:
            logger.info(f"{node.signature} has been verified before, skip.")
            continue

        logger.info(f"Node annos: {node.signature}, {node.anno}")
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
                workdir=outdir / "runs",
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
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
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
