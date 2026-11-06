"""Tests for the A4 evaluation script.

Metric, threshold and binning tests run on small synthetic arrays. The
bulk-versus-per-row cross-check needs the production artifacts and the
processed Backblaze tables and is skipped when either is absent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
from sklearn.metrics import average_precision_score

from hazard import evaluate_v1 as ev
from hazard import fit_v1
from hazard.v0 import FEATURE_NAMES, HeuristicHazardV0

HAZARD_DIR = Path(ev.__file__).resolve().parent
HAVE_REAL_DATA = (
    (ev.ARTIFACT_DIR / "meta.json").exists()
    and fit_v1.TIMELINES.exists()
    and fit_v1.LABELS.exists()
    and fit_v1.DRIVE_SPLIT.exists()
)


# ---- bulk path against the per-row interface -------------------------------


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="v1 artifacts or processed tables absent")
def test_batch_path_matches_the_per_row_interface_on_held_out_rows() -> None:
    frame = next(ev.load_test_split(max_chunks=1))
    assert len(frame) >= 200
    assert ev.cross_check(frame, n_rows=200, device="cpu") < 1e-7


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="v1 artifacts or processed tables absent")
def test_batch_path_matches_the_per_row_interface_with_missing_and_extreme_values() -> None:
    frame = next(ev.load_test_split(max_chunks=1)).head(300).copy()
    columns = list(ev._load(ev.ARTIFACT_DIR).meta["value_columns"])
    frame.loc[frame.index[:50], columns[0]] = np.nan
    frame.loc[frame.index[50:100], columns[1]] = 1e300
    frame.loc[frame.index[100:150], "model"] = "NOT A MODEL"
    assert ev.cross_check(frame, n_rows=300, device="cpu") < 1e-7


def test_v0_batch_equals_the_v0_interface_row_by_row() -> None:
    rng = np.random.default_rng(3)
    names = [n for n in FEATURE_NAMES if n != "smart_188_raw"]
    frame = pd.DataFrame({n: rng.choice([0.0, 1.0, 5.0, 40.0, np.nan], size=500) for n in names})
    batch = ev.predict_v0_batch(frame)
    model = HeuristicHazardV0()
    expected = [model(record)[0] for record in frame.to_dict("records")]
    assert np.array_equal(batch, np.array(expected))


# ---- threshold rule ---------------------------------------------------------


def threshold_fixture() -> tuple[np.ndarray, np.ndarray]:
    negatives = np.arange(1, 1001) / 1000.0
    scores = np.concatenate([negatives, [0.995, 0.5]])
    labels = np.concatenate([np.zeros(1000, dtype=int), [1, 1]])
    return scores, labels


def test_threshold_is_the_most_permissive_point_within_the_fpr_target() -> None:
    scores, labels = threshold_fixture()
    threshold, fpr = ev.select_threshold(scores, labels, 0.01)
    assert threshold == pytest.approx(0.99)
    assert fpr == pytest.approx(0.01)
    flagged = scores > threshold
    assert flagged[labels == 0].sum() == 10 and flagged[labels == 1].sum() == 1


def test_threshold_for_a_tighter_target() -> None:
    scores, labels = threshold_fixture()
    threshold, fpr = ev.select_threshold(scores, labels, 0.001)
    assert threshold == pytest.approx(0.999) and fpr == pytest.approx(0.001)


def test_threshold_never_exceeds_the_target_for_any_target() -> None:
    rng = np.random.default_rng(1)
    scores = np.round(rng.random(5000), 2)
    labels = (rng.random(5000) < 0.1).astype(int)
    for target in (0.5, 0.1, 0.02, 0.001):
        threshold, fpr = ev.select_threshold(scores, labels, target)
        if threshold is not None:
            assert fpr <= target


def test_target_is_unattainable_when_a_tie_group_at_the_top_exceeds_it() -> None:
    labels = np.array([0] * 100 + [1] * 5)
    assert ev.select_threshold(np.full(105, 0.3), labels, 0.01) == (None, 0.0)
    scores = np.concatenate([np.full(100, 0.3), np.full(5, 0.3)])
    assert ev.select_threshold(scores, labels, 0.001)[0] is None


def test_zero_allowed_false_positives_flags_only_rows_above_every_negative() -> None:
    scores, labels = threshold_fixture()
    scores = np.append(scores, 1.2)
    labels = np.append(labels, 1)
    threshold, fpr = ev.select_threshold(scores, labels, 0.0004)
    assert threshold == pytest.approx(1.0) and fpr == 0.0
    assert (scores > threshold).sum() == 1


# ---- binning ---------------------------------------------------------------


def test_equal_frequency_bins_have_equal_counts_on_distinct_scores() -> None:
    bins = ev.equal_frequency_bins(np.arange(100) / 100.0, 4)
    assert np.bincount(bins).tolist() == [25, 25, 25, 25]
    assert (np.diff(bins) >= 0).all()


def test_equal_scores_share_a_bin_and_the_bin_count_can_shrink() -> None:
    scores = np.concatenate([np.zeros(80), np.arange(1, 21) / 100.0])
    bins = ev.equal_frequency_bins(scores, 10)
    assert len(set(bins[:80])) == 1
    assert len(np.unique(bins)) < 10
    assert (np.diff(bins) >= 0).all()


def test_reliability_table_accounts_for_every_row_and_positive() -> None:
    rng = np.random.default_rng(2)
    scores = rng.random(1000)
    labels = (rng.random(1000) < scores).astype(int)
    table = ev.reliability_table(scores, labels, 5)
    assert sum(r["rows"] for r in table) == 1000
    assert sum(r["positives"] for r in table) == labels.sum()
    assert [r["rows"] for r in table] == [200] * 5
    means = [r["mean_pred"] for r in table]
    assert means == sorted(means)
    for r in table:
        assert r["observed"] == pytest.approx(r["positives"] / r["rows"])


def test_bin_count_keeps_ten_failures_per_bin_on_average() -> None:
    assert ev.bin_count(7754) == 20
    assert ev.bin_count(55) == 5
    assert ev.bin_count(3) == 1
    for positives in (10, 99, 400, 5000):
        assert positives / ev.bin_count(positives) >= ev.MIN_FAILURES_PER_BIN


# ---- weighted metrics and the drive bootstrap -------------------------------


def test_pr_auc_matches_average_precision_with_ties() -> None:
    rng = np.random.default_rng(4)
    scores = np.round(rng.random(2000), 1)
    labels = (rng.random(2000) < 0.05 + 0.3 * scores).astype(int)
    drives = rng.integers(0, 50, 2000)
    ranked = ev.Ranked(scores, labels, drives)
    ap, _ = ranked.metrics(np.ones(50), [])
    assert ap == pytest.approx(average_precision_score(labels, scores))


def test_integer_drive_weights_equal_duplicating_the_drives_rows() -> None:
    rng = np.random.default_rng(5)
    scores = np.round(rng.random(600), 2)
    labels = (rng.random(600) < 0.1 + 0.2 * scores).astype(int)
    drives = rng.integers(0, 12, 600)
    counts = rng.integers(0, 4, 12).astype(float)
    repeated = np.repeat(np.arange(600), counts[drives].astype(int))
    expected = average_precision_score(labels[repeated], scores[repeated])
    ap, _ = ev.Ranked(scores, labels, drives).metrics(counts, [])
    assert ap == pytest.approx(expected)
    error = np.bincount(drives, weights=(scores - labels) ** 2, minlength=12)
    rows = np.bincount(drives, minlength=12).astype(float)
    brier = ev.brier_weighted(error, rows, counts)
    assert brier == pytest.approx(np.mean((scores[repeated] - labels[repeated]) ** 2))


def test_recall_at_a_frozen_threshold_counts_rows_strictly_above_it() -> None:
    scores = np.array([0.9, 0.8, 0.8, 0.5, 0.1])
    labels = np.array([1, 1, 0, 1, 0])
    ranked = ev.Ranked(scores, labels, np.arange(5))
    assert ranked.above(0.8) == 1 and ranked.above(0.79) == 3 and ranked.above(None) == 0
    _, [(recall, fpr)] = ranked.metrics(np.ones(5), [ranked.above(0.79)])
    assert recall == pytest.approx(2 / 3) and fpr == pytest.approx(1 / 2)


def synthetic_split(seed: int, drives: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per = 30
    serial = np.repeat([f"d{i}" for i in range(drives)], per)
    failing = np.repeat(rng.random(drives) < 0.2, per)
    p1 = np.clip(rng.random(drives * per) * 0.2 + failing * 0.3, 0, 0.99)
    return pd.DataFrame({
        "serial_number": serial, "model": "m", "vendor": "v",
        "label": (rng.random(drives * per) < p1).astype(np.int8),
        ev.V0: np.round(rng.random(drives * per), 1), ev.V1: p1,
    })


def test_evaluate_cell_on_a_synthetic_split() -> None:
    test, val = synthetic_split(1), synthetic_split(2)
    mask = np.ones(len(test), dtype=bool)
    cell = ev.evaluate_cell(0, test, val, mask, mask, resamples=30)
    assert cell["rows"] == len(test) and cell["drives"] == 120
    assert cell["thin"] == (cell["failed_drives"] < ev.MIN_FAILED_DRIVES)
    base = cell["base_rate"]
    constant = cell["predictors"][ev.BASELINE]
    assert constant["pr_auc"] == pytest.approx(base)
    assert constant["brier"] == pytest.approx(base * (1 - base), rel=1e-6)
    assert all(not t["attainable"] for t in constant["targets"])
    assert constant["reliability"] == []
    for name in (ev.V0, ev.V1):
        e = cell["predictors"][name]
        lo, hi = e["pr_auc_ci"]
        assert lo <= hi
        assert sum(r["rows"] for r in e["reliability"]) == len(test)
    assert set(cell["diff_v1_minus_v0"]) == {"pr_auc", "pr_auc_ci", "brier", "brier_ci"}
    assert ev.evaluate_cell(0, test, val, mask, mask, resamples=30)["predictors"][ev.V1]["pr_auc_ci"] == \
        cell["predictors"][ev.V1]["pr_auc_ci"]


def test_bootstrap_interval_is_wider_for_fewer_drives() -> None:
    mask = lambda f: np.ones(len(f), dtype=bool)  # noqa: E731
    few, many = synthetic_split(3, drives=30), synthetic_split(3, drives=300)
    val = synthetic_split(4)
    width = []
    for frame in (few, many):
        cell = ev.evaluate_cell(0, frame, val, mask(frame), mask(val), resamples=60)
        lo, hi = cell["predictors"][ev.V1]["brier_ci"]
        width.append(hi - lo)
    assert width[0] > width[1]


def test_verdict_names_the_leader() -> None:
    def cell(ap, brier):
        return {"diff_v1_minus_v0": {"pr_auc": 0.0, "pr_auc_ci": ap, "brier": 0.0, "brier_ci": brier}}
    assert ev.verdict(cell((0.01, 0.02), (-2e-5, -1e-5))).startswith("v1 better than v0")
    assert ev.verdict(cell((-0.02, -0.01), (1e-5, 2e-5))).startswith("v0 better than v1")
    assert ev.verdict(cell((0.01, 0.02), (1e-5, 2e-5))).startswith("mixed")
    assert ev.verdict(cell((-0.01, 0.02), (-1e-5, 1e-5))).startswith("no difference")
    assert ev.verdict(cell((0.01, 0.02), (-1e-5, 1e-5))).startswith("v1 ahead on PR-AUC")


def test_write_record_replaces_only_the_generated_blocks(tmp_path: Path) -> None:
    path = tmp_path / "record.md"
    path.write_text("head\n<!-- a4:evidence -->\nold\n<!-- /a4:evidence -->\nmid\n"
                    "<!-- a4:verdict -->\nold\n<!-- /a4:verdict -->\ntail\n", encoding="utf-8")
    test, val = synthetic_split(1), synthetic_split(2)
    mask = np.ones(len(test), dtype=bool)
    results = {"pooled": ev.evaluate_cell(0, test, val, mask, mask, resamples=10)}
    meta = {"device": "cpu", "cross_rows": 200, "cross_diff": 0.0, "test_rows": len(test), "val_rows": len(val)}
    ev.write_record(results, meta, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("head\n") and text.endswith("tail\n") and "\nmid\n" in text
    assert "\nold\n" not in text and "| pooled |" in text and "- Pooled:" in text
    ev.write_record(results, meta, path)
    assert path.read_text(encoding="utf-8") == text
    path.write_text("no blocks", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.write_record(results, meta, path)


# ---- guardrails ------------------------------------------------------------


WORD = re.compile(r"(?<![A-Za-z])test(?:ing)?(?![A-Za-z])", re.IGNORECASE)


@pytest.mark.parametrize("name", ["v0.py", "v1.py", "fit_v1.py"])
def test_untouched_modules_still_have_no_reference_to_the_held_out_split(name: str) -> None:
    assert not WORD.search((HAZARD_DIR / name).read_text(encoding="utf-8"))


def test_the_held_out_split_is_read_only_through_load_test_split() -> None:
    tree = ast.parse((HAZARD_DIR / "evaluate_v1.py").read_text(encoding="utf-8"))
    literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "test"]
    assert len(literals) == 1
    assigned = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "HELD_OUT_SPLIT" for t in n.targets)]
    assert len(assigned) == 1 and assigned[0].value is literals[0]

    uses: list[str] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Name) and child.id == "HELD_OUT_SPLIT" and isinstance(child.ctx, ast.Load):
                uses.append(scope)
            visit(child, inner)

    visit(tree, "<module>")
    assert uses == ["load_test_split"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "load_test_split"]
    assert calls, "the held-out loader is never used"
