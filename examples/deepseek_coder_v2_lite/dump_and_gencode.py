#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Dump & Regenerate Execution Graph Example for DeepSeek Coder V2 Lite
=====================================================================

This script demonstrates how to use nnScaler's execution plan dump and
code-regeneration features with the DeepSeek Coder V2 Lite model.

After a normal compilation (``python train.py --run_mode compile ...``),
nnScaler now automatically saves the full execution plan alongside the
generated code. This script shows how to:

1. **Inspect** the execution graph (JSON analysis) to understand
   computation and communication dependencies.
2. **Regenerate** distributed training code from a saved execution plan
   without recompilation.

Prerequisites
-------------
- Run the normal compilation first::

    python train.py --run_mode compile \\
        --plan_ngpus 4 --runtime_ngpus 8 \\
        --model_id deepseek-ai/DeepSeek-Coder-V2-Lite-Base \\
        --dataset_path ./bookcorpus_2k

  This generates the execution plan at::

    .nnscaler/_parallel_modules/.../execplan.pkl
    .nnscaler/_parallel_modules/.../execplan.pkl.json

Usage
-----
::

    # Inspect the graph
    python dump_and_gencode.py inspect \\
        --plan_dir .nnscaler/_parallel_modules/<path_to_module_dir>

    # Regenerate code to a new directory
    python dump_and_gencode.py gencode \\
        --plan_dir .nnscaler/_parallel_modules/<path_to_module_dir> \\
        --num_ranks 8 \\
        --outdir ./regenerated_code

For a quick test without DeepSeek dependencies, you can also run::

    python dump_and_gencode.py demo
"""

import argparse
import json
import os
import sys


def cmd_inspect(args):
    """Inspect an execution plan's JSON analysis dump."""
    json_file = os.path.join(args.plan_dir, 'execplan.pkl.json')
    if not os.path.exists(json_file):
        print(f'Error: {json_file} not found.')
        print('Run the compilation first (python train.py --run_mode compile ...)')
        sys.exit(1)

    with open(json_file) as f:
        data = json.load(f)

    meta = data['metadata']
    print('=' * 60)
    print('Execution Graph Summary')
    print('=' * 60)
    print(f"  Training mode:   {meta['train']}")
    print(f"  Num devices:     {meta['num_devices']}")
    print(f"  Devices:         {meta['devices']}")
    print(f"  Has scheduler:   {meta.get('has_scheduler', 'N/A')}")
    print()

    # Graph attributes (parameters)
    attrs = data.get('graph_attributes', [])
    params = [a for a in attrs if a.get('is_param')]
    buffers = [a for a in attrs if a.get('is_buffer')]
    print(f"  Parameters:      {len(params)}")
    print(f"  Buffers:         {len(buffers)}")
    print()

    # Per-device summary
    for devid, dev_data in data['per_device'].items():
        summary = dev_data['summary']
        print(f'  Device {devid}:')
        print(f'    Total nodes:       {summary["total_nodes"]}')
        print(f'    Forward nodes:     {summary["forward_nodes"]}')
        print(f'    Backward nodes:    {summary["backward_nodes"]}')
        print(f'    Communication:     {summary["communication_nodes"]}')
        print(f'    Weight reducers:   {summary["weight_reducer_nodes"]}')
        print(f'    Data nodes:        {summary["data_nodes"]}')
        print(f'    Dependency edges:  {len(dev_data["edges"])}')
        print()

    # Communication breakdown
    for devid, dev_data in data['per_device'].items():
        comm_nodes = [
            n for n in dev_data['nodes'].values()
            if n['category'] == 'communication'
        ]
        if comm_nodes:
            print(f'  Device {devid} Communication Breakdown:')
            prim_counts = {}
            total_vol = 0
            for cn in comm_nodes:
                for prim in cn.get('comm_primitives', []):
                    prim_counts[prim] = prim_counts.get(prim, 0) + 1
                total_vol += cn.get('total_comm_volume', 0)
            for prim, count in sorted(prim_counts.items()):
                print(f'    {prim}: {count}')
            if total_vol > 0:
                print(f'    Total comm volume: {total_vol:,} elements')
            print()

    if args.verbose:
        # Print execution order for device 0
        first_dev = list(data['per_device'].keys())[0]
        dev_data = data['per_device'][first_dev]
        print(f'  Execution Order (Device {first_dev}):')
        for i, nid in enumerate(dev_data['execution_order']):
            node = dev_data['nodes'][nid]
            phase = 'FW' if node['is_forward'] else 'BW'
            print(f'    [{i:3d}] {phase} {node["category"]:20s} '
                  f'{node["name"]:15s} {node["signature"][:40]}')
        print()

    print(f'Full JSON analysis: {json_file}')


def cmd_gencode(args):
    """Regenerate distributed code from a saved execution plan."""
    from nnscaler.execplan.graphdump import gencode_from_file

    pkl_file = os.path.join(args.plan_dir, 'execplan.pkl')
    if not os.path.exists(pkl_file):
        print(f'Error: {pkl_file} not found.')
        print('Run the compilation first (python train.py --run_mode compile ...)')
        sys.exit(1)

    print(f'Loading execution plan from: {pkl_file}')
    files = gencode_from_file(
        pkl_file,
        num_ranks=args.num_ranks,
        outdir=args.outdir,
        runtime_ndevs=args.runtime_ndevs,
    )

    print(f'Generated {len(files)} code files in {args.outdir}:')
    for f in files:
        size = os.path.getsize(f)
        print(f'  {os.path.basename(f):20s} ({size:,} bytes)')


def cmd_demo(args):
    """Run a self-contained demo with a tiny MLP model (no DeepSeek dependencies)."""
    import tempfile

    import torch
    from nnscaler.ir.tensor import IRFullTensor
    from nnscaler.ir.unique import IDGenerator
    import nnscaler.graph.function.function as F
    from nnscaler.graph.graph import IRGraph
    from nnscaler.execplan.execplan import ExecutionPlan
    from nnscaler.execplan.graphdump import (
        dump_execution_graph,
        save_execution_plan,
        load_execution_plan,
        gencode_from_execution_plan,
    )

    print('=' * 60)
    print('Demo: Dump & Gencode with a simple MLP graph')
    print('=' * 60)
    print()

    # --- Step 1: Build a simple computation graph ---
    print('[Step 1] Building computation graph (Linear -> Linear -> Loss)...')
    IDGenerator().clear()

    def _t(shape, rg=True):
        return IRFullTensor(shape, requires_grad=rg).tosub()

    data = _t([32, 128], False)   # batch=32, features=128
    w1 = _t([64, 128])            # weight 1: out=64, in=128
    h = _t([32, 64])
    linear1 = F.Linear(data, w1)
    linear1.set_output(0, h)

    w2 = _t([10, 64])             # weight 2: out=10, in=64
    logits = _t([32, 10])
    linear2 = F.Linear(h, w2)
    linear2.set_output(0, logits)

    loss = _t([1])
    sum_op = F.Sum(logits)
    sum_op.set_output(0, loss)

    graph = IRGraph([linear1, linear2, sum_op], [data], [loss], 'DemoMLP')
    graph.backward(loss)

    for node in graph.nodes():
        node.device = [0]

    execplan = ExecutionPlan.from_graph(graph)
    print(f'  Nodes: {len(execplan.seq(0))} on device 0')
    print()

    # --- Step 2: Dump the execution graph ---
    print('[Step 2] Dumping execution graph...')
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save full plan (dill) + companion JSON
        pkl_file = os.path.join(tmpdir, 'execplan.pkl')
        save_execution_plan(execplan, pkl_file, save_json=True)
        print(f'  Saved: {pkl_file} ({os.path.getsize(pkl_file):,} bytes)')
        print(f'  Saved: {pkl_file}.json ({os.path.getsize(pkl_file + ".json"):,} bytes)')
        print()

        # Show JSON summary
        result = dump_execution_graph(execplan)
        summary = result['per_device'][0]['summary']
        print('  Graph Summary:')
        print(f'    Forward ops:       {summary["forward_nodes"]}')
        print(f'    Backward ops:      {summary["backward_nodes"]}')
        print(f'    Communication:     {summary["communication_nodes"]}')
        print(f'    Dependencies:      {len(result["per_device"][0]["edges"])} edges')
        print()

        # Show dependency chain
        edges = result['per_device'][0]['edges']
        nodes = result['per_device'][0]['nodes']
        print('  Dependency Chain:')
        for edge in edges:
            src = nodes[edge['from']]
            dst = nodes[edge['to']]
            phase_src = 'FW' if src['is_forward'] else 'BW'
            phase_dst = 'FW' if dst['is_forward'] else 'BW'
            print(f'    {phase_src} {src["name"]:10s} -> {phase_dst} {dst["name"]:10s}'
                  f'  (tensor: {edge["tensor_name"]}, tid={edge["tensor_tid"]})')
        print()

        # --- Step 3: Regenerate code from the saved plan ---
        print('[Step 3] Regenerating code from saved plan...')
        loaded = load_execution_plan(pkl_file)
        print(f'  Loaded plan: {len(loaded.devices())} device(s), '
              f'{len(loaded.seq(0))} nodes on device 0')

        outdir = os.path.join(tmpdir, 'generated')
        files = gencode_from_execution_plan(loaded, num_ranks=1, outdir=outdir)
        print(f'  Generated: {files[0]}')
        print()

        # Show generated code snippet
        with open(files[0]) as f:
            code = f.read()

        print('[Generated Code Preview]')
        print('-' * 60)
        # Show the GenModel class and train_step
        lines = code.split('\n')
        in_section = False
        for line in lines:
            if 'class GenModel' in line or 'def _train_step' in line:
                in_section = True
            if in_section:
                print(line)
                if line.strip().startswith('return') and in_section:
                    print()
                    in_section = False
        print('-' * 60)
        print()
        print('Demo complete!')


def main():
    parser = argparse.ArgumentParser(
        description='Dump & regenerate execution graphs for nnScaler models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # inspect
    p_inspect = subparsers.add_parser(
        'inspect', help='Inspect execution graph from JSON analysis dump')
    p_inspect.add_argument(
        '--plan_dir', required=True,
        help='Directory containing execplan.pkl.json')
    p_inspect.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show execution order details')

    # gencode
    p_gencode = subparsers.add_parser(
        'gencode', help='Regenerate distributed code from saved execution plan')
    p_gencode.add_argument(
        '--plan_dir', required=True,
        help='Directory containing execplan.pkl')
    p_gencode.add_argument(
        '--num_ranks', type=int, required=True,
        help='Number of rank files to generate')
    p_gencode.add_argument(
        '--outdir', default='./regenerated',
        help='Output directory for generated code')
    p_gencode.add_argument(
        '--runtime_ndevs', type=int, default=None,
        help='Data-parallel scaling target (optional)')

    # demo
    subparsers.add_parser(
        'demo', help='Run self-contained demo with a tiny MLP graph')

    args = parser.parse_args()
    if args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'gencode':
        cmd_gencode(args)
    elif args.command == 'demo':
        cmd_demo(args)


if __name__ == '__main__':
    main()
