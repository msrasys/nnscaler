# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from examples.inspect_parallel_plan import inspect_plan


def test_inspect_plan_detects_sharding_pipeline_and_collectives(tmp_path):
    rank0 = tmp_path / "gencode0.py"
    rank1 = tmp_path / "gencode1.py"
    rank0.write_text(
        "\n".join(
            [
                "self.add_full_map('w0', 1, True, 'layer0.weight', (8, 4), "
                "(slice(0, 4, None), slice(0, 4, None)), 1)",
                "x = nnscaler.runtime.adapter.nn.allreduce_identity(x, ranks=[0, 1])",
                "y = expert(x, local_expert_start=0, local_expert_end=2)",
            ]
        ),
        encoding="utf-8",
    )
    rank1.write_text(
        "self.add_full_map('w1', 2, True, 'layer1.weight', (8, 4), "
        "(slice(4, 8, None), slice(0, 4, None)), 1)\n",
        encoding="utf-8",
    )

    report = inspect_plan(tmp_path)

    assert report["rank_count"] == 2
    assert report["pipeline_detected"]
    assert report["collectives"] == {"allreduce_identity": 1}
    assert report["expert_ranges"] == [(0, 2)]
    assert report["ranks"][0]["sharded_parameter_count"] == 1


def test_inspect_plan_does_not_treat_buffer_differences_as_pipeline(tmp_path):
    for rank, buffer_name in enumerate(("cache0", "cache1")):
        (tmp_path / f"gencode{rank}.py").write_text(
            "\n".join(
                [
                    "self.add_full_map('w', 1, True, 'weight', (4,), "
                    "(slice(0, 4, None),), 1)",
                    f"self.add_full_map('b', 2, False, '{buffer_name}', (4,), "
                    "(slice(0, 4, None),), 1)",
                ]
            ),
            encoding="utf-8",
        )

    report = inspect_plan(tmp_path)

    assert report["distinct_parameter_sets"] == 1
    assert report["distinct_state_sets"] == 2
    assert not report["pipeline_detected"]


def test_inspect_plan_accepts_scalar_ellipsis_slicer(tmp_path):
    (tmp_path / "gencode0.py").write_text(
        "self.add_full_map('scalar', 1, False, 'counter', (), ..., 1)\n",
        encoding="utf-8",
    )

    report = inspect_plan(tmp_path)

    assert report["rank_count"] == 1
    assert report["ranks"][0]["sharded_buffer_count"] == 0
