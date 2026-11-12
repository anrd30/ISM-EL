"""Static audit of the pytest skip conditions defined as module-level values.

Imports each test module that defines a skip flag and reads the flag; no test is run and
pytest is not invoked. ``baselines/conftest.py`` skips tests marked ``slow`` unless
``--runslow`` or ``BASELINES_RUN_SLOW`` is set; that condition is reported as a count of
``@pytest.mark.slow`` test functions.

Usage::

    python scripts/skip_audit.py
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# (module, flag attribute, kind). kind "present": flag True means the dependency exists (skip when False).
# kind "paths": the attributes name Path objects that all have to exist.
PRESENT_FLAGS = [
    ("agent.test_agent", "_V1_PRESENT"),
    ("agent.test_evaluate_policy", "_V1_PRESENT"),
    ("agent.test_evaluate_policy", "_SPLIT_PRESENT"),
    ("sim.test_latency_filter", "_V1_PRESENT"),
    ("sim.test_recovery_env", "_V1_PRESENT"),
    ("sim.test_scenario_features", "_V1_PRESENT"),
    ("sim.test_scenario_features", "_TABLE_PRESENT"),
    ("data_pipeline.tests.test_split_long_profile", "_SOURCE_PRESENT"),
    ("hazard.test_evaluate_v1", "HAVE_REAL_DATA"),
    ("hazard.test_v1", "_REAL_ARTIFACTS_PRESENT"),
]
PATH_CONDITIONS = [
    ("data_pipeline.tests.test_no_leakage", ("FILTERED", "INDEX")),
    ("hazard.test_build_fleet_features", ("_REAL_TABLE",)),
]
IMPORT_SKIPS = [("pyarrow", "data_pipeline/tests (8 modules)"), ("xgboost", "hazard/test_v1.py, hazard/test_evaluate_v1.py"),
                ("pandas", "contracts/test_contracts.py")]


def _import(name: str):
    try:
        return importlib.import_module(name), None
    except BaseException as exc:  # a module-level importorskip raises Skipped
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def _slow_tests(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            for dec in node.decorator_list:
                if ast.unparse(dec) == "pytest.mark.slow":
                    names.append(node.name)
    return names


def main() -> int:
    rows: list[tuple[str, bool | None, str]] = []  # (condition, would_skip, detail)
    for module, flag in PRESENT_FLAGS:
        mod, err = _import(module)
        if mod is None:
            rows.append((f"{module}.{flag}", None, f"module not importable: {err}"))
            continue
        value = bool(getattr(mod, flag))
        deps = {
            "_V1_PRESENT": f"{getattr(mod, 'V1_ARTIFACT_DIR', '')}/meta.json",
            "_SPLIT_PRESENT": f"{getattr(mod, 'TRAIN_CSV', '')} and {getattr(mod, 'HOLDOUT_CSV', '')}",
            "_TABLE_PRESENT": f"{getattr(mod, 'FLEET_FEATURES_PATH', '')}",
            "_SOURCE_PRESENT": f"{getattr(mod, 'SOURCE', '')}",
            "_REAL_ARTIFACTS_PRESENT": f"{getattr(mod, 'ARTIFACT_DIR', '')}/meta.json",
            "HAVE_REAL_DATA": "v1 artifacts and processed tables (see module)",
        }[flag]
        rows.append((f"{module}.{flag} = {value}", not value, f"depends on {deps}"))
    for module, attrs in PATH_CONDITIONS:
        mod, err = _import(module)
        if mod is None:
            rows.append((f"{module} {attrs}", None, f"module not importable: {err}"))
            continue
        paths = [Path(getattr(mod, a)) for a in attrs]
        exist = [p.exists() for p in paths]
        rows.append((f"{module} {'+'.join(attrs)} exist = {all(exist)}", not all(exist),
                     "depends on " + ", ".join(f"{p} (exists={e})" for p, e in zip(paths, exist))))
    for pkg, where in IMPORT_SKIPS:
        found = importlib.util.find_spec(pkg) is not None
        rows.append((f"importorskip({pkg}) importable = {found}", not found, f"used by {where}"))

    slow = _slow_tests(REPO / "baselines" / "test_baselines.py")
    slow_active = not os.environ.get("BASELINES_RUN_SLOW")
    rows.append((f"baselines/conftest.py slow-marker skip: {len(slow)} test functions marked slow", slow_active,
                 f"skipped unless --runslow or BASELINES_RUN_SLOW is set (env set = {not slow_active}); "
                 f"a skipped function counts as {len(slow)} skipped tests unless parametrized"))

    print("skip audit (module-level values, no test executed)")
    for cond, would_skip, detail in rows:
        state = "UNKNOWN" if would_skip is None else ("WOULD SKIP" if would_skip else "runs")
        print(f"- [{state}] {cond}: {detail}")
    true_count = sum(1 for _, s, _ in rows if s)
    unknown = sum(1 for _, s, _ in rows if s is None)
    print(f"conditions that would skip: {true_count}; unknown: {unknown}; slow-marked test functions: {len(slow)}")
    print("run-time conditions not evaluated: contracts/test_adapter_conformance.py env-factory skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
