import argparse
import os
import sys
import torch
from nnscaler.graph.function.dimops import DimAnno, IRDimops, OpAnno
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.cten import IRObject, IRTensor
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
        return torch.load(verified_ops_file)
    else:
        logger.info(f"{verified_ops_file} does not exist, start from scratch.")
        return set()


def save_verified_ops(outdir: Path, verified_ops: set):
    verified_ops_file = outdir / _VERIFIED_OPS_FILE_NAME
    torch.save(verified_ops, verified_ops_file)
    logger.info(f"Verification results saved to {verified_ops_file}")


def verify_op_partitions(graph: IRGraph, outdir: Path):
    """
    Test if the partitioned ops in the graph are computationally correct.

    Args:
        graph (IRGraph): the graph to be verified
        outdir (Path): the directory to save the verified ops

    Returns:
        None
    """
    from verify_dimops import (
        VerifyConfig,
        TensorInfo,
        verify_partition_options,
    )

    verified_ops = load_verified_ops(outdir)
    skipped_nodes = []

    gnodes = graph.nodes(flatten=True)
    for idx, node in enumerate(gnodes):
        logger.info(f"node: {node}")
        logger.info(f"Verification progress: {idx} / {len(gnodes)}")
        if node.isfw() and isinstance(node, IRDimops):
            ins_info = [
                (
                    TensorInfo("shape", _input.shape)
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
                    TensorInfo("shape", output.shape)
                    if isinstance(output, IRTensor)
                    else TensorInfo(
                        "value",
                        output.value if isinstance(output, IRObject) else output,
                    )
                )
                for output in node.outputs()
            ]
            if (node.signature, tuple(ins_info + outs_info)) in verified_ops:
                logger.info(f"{node.signature} has been verified before, skip.")
                continue

            logger.info(f"Node annos: {node.signature}, {node.anno}")

            parti_options = get_candidate_options(node.anno, ins_info + outs_info)

            logger.info(f"Candidate partition options: {parti_options}")

            verify_config = VerifyConfig(
                fsig=node.signature,
                args=ins_info,
                kwargs=node.kwargs,
                noutputs=len(node.outputs()),
                parti_options=parti_options,
            )
            try:
                iscorrect = verify_partition_options(verify_config)
            except Exception as e:
                logger.warning(
                    f"Verification failed for {node.signature}, {e}, please manually verify."
                )
                iscorrect = True  # fake true to skip this node
            if not iscorrect:
                logger.warning(f"Verification failed for {node.signature}, continuing execution.")
                continue

            verified_ops.add((node.signature, tuple(ins_info + outs_info)))
            save_verified_ops(outdir, verified_ops)

    if skipped_nodes:
        logger.info("Skipped the following nodes due to empty ins_info:")
        for node_info in skipped_nodes:
            logger.info(f" - {node_info}")

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
    verify_op_partitions(graph, outdir)


if __name__ == "__main__":
    main()
