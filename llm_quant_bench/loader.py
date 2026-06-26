from __future__ import annotations

from pathlib import Path

import yaml

from llm_quant_bench.config import PROJECT_ROOT
from llm_quant_bench.models import TestCase

DIMENSION_DIRS = {
    "t1": "t1_indicator",
    "t2": "t2_pattern",
    "t3": "t3_json",
    "t4": "t4_rule",
    "t5": "t5_anomaly",
    "t6": "t6_longctx",
    "t7": "t7_constraint",
}


def load_dimension(dimension: str, tests_dir: str | Path | None = None) -> list[TestCase]:
    base = Path(tests_dir) if tests_dir else PROJECT_ROOT / "benchmarks"
    dir_name = DIMENSION_DIRS.get(dimension)
    if not dir_name:
        raise ValueError(f"Unknown dimension: {dimension}")

    dim_dir = base / dir_name
    if not dim_dir.exists():
        return []

    cases: list[TestCase] = []
    for yaml_file in sorted(dim_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        if not data or "cases" not in data:
            continue
        for item in data["cases"]:
            item.setdefault("dimension", dimension)
            cases.append(TestCase(**item))
    return cases


def load_all(
    dimensions: list[str] | None = None,
    tests_dir: str | Path | None = None,
) -> list[TestCase]:
    dims = dimensions or list(DIMENSION_DIRS.keys())
    cases: list[TestCase] = []
    for dim in dims:
        cases.extend(load_dimension(dim, tests_dir))
    return cases
