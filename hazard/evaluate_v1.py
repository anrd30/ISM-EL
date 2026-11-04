"""A4: score hazard v0 and hazard v1 on the held-out split of the Backblaze tables.

Usage (from the repository root)::

    python -m hazard.evaluate_v1                 # print every number
    python -m hazard.evaluate_v1 --write-record  # also fill the generated blocks of A4_DECISION.md

Nothing is fitted here. Both models are used as they stand. Validation rows are
read only to choose the two decision thresholds; every reported metric is
computed on the held-out rows (label != -1, drives of that split). The held-out
rows are read by ``load_test_split`` and nowhere else in this file.

Metrics per cell (pooled, vendor, drive model) and per predictor:

- PR-AUC: average precision, ties in the score handled as one group.
- Brier score and an equal-frequency reliability table.
- Recall at a false-positive-rate target of 1 % and 0.1 %. Rule (one-sided, as
  the largest-N rule of the load profiles): with ``k = floor(target x N_neg)``
  validation negatives allowed above the threshold, the threshold is the
  ``(k + 1)``-th largest validation negative score and a row is flagged when
  its score is strictly above it. The validation false-positive rate is then at
  most the target and the flagged set is as large as that allows. If no
  validation row is above the threshold the target is reported as unattainable.
  The threshold is frozen and applied to the held-out rows.

Confidence intervals are percentile intervals of ``BOOTSTRAP_RESAMPLES`` drive
bootstrap resamples (whole drives drawn with replacement within a cell, all
rows of a drawn drive carried with it). The frozen thresholds are not
re-derived inside a resample.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import xgboost as xgb

from hazard import fit_v1
from hazard.v0 import FEATURE_NAMES as V0_FEATURES
from hazard.v0 import HeuristicHazardV0
from hazard.v1 import ARTIFACT_DIR, FittedHazardV1, _load

HELD_OUT_SPLIT = "test"
VALIDATION_SPLIT = fit_v1.VAL_SPLIT
RECORD = Path(__file__).resolve().parent / "A4_DECISION.md"

BASELINE = "constant_base_rate (diagnostic)"
V0 = "hazard_v0"
V1 = "hazard_v1"
PREDICTORS = (BASELINE, V0, V1)

FPR_TARGETS = (0.01, 0.001)
RELIABILITY_BINS = 20
MIN_FAILURES_PER_BIN = 10
MIN_FAILED_DRIVES = 50
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 20220401
CROSS_CHECK_ROWS = 200
CROSS_CHECK_TOLERANCE = 1e-4

BLOCK = "<!-- a4:{name} -->\n{body}\n<!-- /a4:{name} -->"


# ---- reading the splits ----------------------------------------------------


def select_split_rows(labels: pd.DataFrame, drive_split: pd.DataFrame, split_name: str) -> np.ndarray:
    """Boolean mask over ``labels`` rows: the drive is in ``split_name`` and label != -1."""
    joined = labels[["serial_number", "label"]].merge(
        drive_split[["serial_number", "split"]], on="serial_number", how="left", validate="many_to_one"
    )
    return ((joined["split"] == split_name) & (joined["label"] != fit_v1.CENSORED)).to_numpy()


def _read_split(split_name: str, max_chunks: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield frames of the rows of ``split_name`` with a label: serial_number, model, label, value columns."""
    drive_split = pd.read_parquet(fit_v1.DRIVE_SPLIT, columns=["serial_number", "split"])
    values = list(_load(ARTIFACT_DIR).meta["value_columns"])
    read = [*fit_v1.KEY_COLUMNS, fit_v1.MODEL_COLUMN, *values]
    for index, (left, right) in enumerate(fit_v1.paired_chunks(read)):
        if max_chunks is not None and index >= max_chunks:
            return
        labels = right.select(["serial_number", "label"]).to_pandas()
        mask = select_split_rows(labels, drive_split, split_name)
        if not mask.any():
            continue
        chosen = left.filter(pa.array(mask)).to_pandas()
        chosen["label"] = labels["label"].to_numpy()[mask]
        yield chosen[["serial_number", fit_v1.MODEL_COLUMN, "label", *values]].reset_index(drop=True)


def load_test_split(max_chunks: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield the held-out rows with a label. The only reader of the held-out split."""
    return _read_split(HELD_OUT_SPLIT, max_chunks)


def load_val_split(max_chunks: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield the validation rows with a label (used only to choose thresholds)."""
    return _read_split(VALIDATION_SPLIT, max_chunks)


# ---- predictors ------------------------------------------------------------


def select_device() -> str:
    return fit_v1.select_device()[0]


@lru_cache(maxsize=None)
def _bulk_boosters(device: str) -> tuple:
    """Return the ensemble's boosters on ``device`` (the cached CPU boosters when ``device`` is cpu)."""
    ensemble = _load(ARTIFACT_DIR)
    if device == "cpu":
        return ensemble.boosters
    boosters = []
    for name in ensemble.meta["boosters"]:
        booster = xgb.Booster()
        booster.load_model(ARTIFACT_DIR / name)
        booster.set_param({"device": device})
        boosters.append(booster)
    return tuple(boosters)


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the many-row model input in training column order for a frame of value columns and ``model``.

    Same semantics as ``hazard.v1.build_feature_frame``: values that are not
    finite after narrowing to float32 are NaN, a flag is 1 where its value is
    NaN, and a model name outside the fitted categories is a missing category.
    ``cross_check`` compares the two paths.
    """
    meta = _load(ARTIFACT_DIR).meta
    values = frame[meta["value_columns"]].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore"):
        narrowed = values.astype(np.float32)
    narrowed[~np.isfinite(narrowed)] = np.nan
    out = pd.DataFrame(narrowed, columns=meta["value_columns"], index=frame.index)
    flags = pd.DataFrame(np.isnan(narrowed).astype(np.float32), columns=meta["flag_columns"], index=frame.index)
    out = pd.concat([out, flags], axis=1)
    out[fit_v1.MODEL_COLUMN] = pd.Categorical(frame[fit_v1.MODEL_COLUMN], categories=meta["model_categories"])
    return out[meta["feature_columns"]]


def predict_v1_batch(frame: pd.DataFrame, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_horizon, uncertainty)`` of ``FittedHazardV1`` for every row of ``frame``."""
    ensemble = _load(ARTIFACT_DIR)
    matrix = xgb.DMatrix(model_frame(frame), enable_categorical=True)
    calibrated = np.stack(
        [np.interp(booster.predict(matrix), x, y)
         for booster, (x, y) in zip(_bulk_boosters(device), ensemble.thresholds)]
    )
    p = np.clip(calibrated.mean(axis=0), 0.0, np.nextafter(1.0, 0.0))
    return p, np.minimum(calibrated.std(axis=0), 0.5)


def predict_v0_batch(frame: pd.DataFrame) -> np.ndarray:
    """Return ``HeuristicHazardV0`` probabilities for every row of ``frame``.

    The model is called once per distinct combination of the columns it reads
    and the result is mapped back to the rows.
    """
    columns = [name for name in V0_FEATURES if name in frame.columns]
    groups = frame.groupby(columns, dropna=False, sort=False).ngroup().to_numpy()
    first = np.unique(groups, return_index=True)[1]
    model = HeuristicHazardV0()
    probabilities = np.array(
        [model(dict(zip(columns, row)))[0] for row in frame[columns].to_numpy()[first].tolist()]
    )
    return probabilities[groups]


def cross_check(frame: pd.DataFrame, n_rows: int = CROSS_CHECK_ROWS, seed: int = BOOTSTRAP_SEED,
                device: str = "cpu") -> float:
    """Return the largest absolute difference between the per-row interface and the batch path.

    Samples ``n_rows`` rows of ``frame``, calls ``FittedHazardV1`` on each as a
    mapping, and compares ``p_horizon`` and ``uncertainty`` with the batch
    outputs for the same rows.
    """
    rows = frame.sample(n=min(n_rows, len(frame)), random_state=seed)
    batch_p, batch_u = predict_v1_batch(rows, device)
    model = FittedHazardV1()
    columns = list(_load(ARTIFACT_DIR).meta["value_columns"]) + [fit_v1.MODEL_COLUMN]
    worst = 0.0
    for i, record in enumerate(rows[columns].to_dict("records")):
        p, u = model(record)
        worst = max(worst, abs(p - batch_p[i]), abs(u - batch_u[i]))
    return worst


def score_chunks(chunks: Iterator[pd.DataFrame], device: str) -> pd.DataFrame:
    """Return serial_number, model, vendor, label and the v0 and v1 scores for every row of ``chunks``."""
    parts = []
    for frame in chunks:
        p1, _ = predict_v1_batch(frame, device)
        parts.append(pd.DataFrame({
            "serial_number": frame["serial_number"].to_numpy(),
            fit_v1.MODEL_COLUMN: frame[fit_v1.MODEL_COLUMN].to_numpy(),
            "label": frame["label"].to_numpy().astype(np.int8),
            V0: predict_v0_batch(frame),
            V1: p1,
        }))
    scored = pd.concat(parts, ignore_index=True)
    scored["vendor"] = scored[fit_v1.MODEL_COLUMN].map({m: vendor_of(m) for m in scored[fit_v1.MODEL_COLUMN].unique()})
    return scored


def vendor_of(model: str) -> str:
    if model.startswith("ST"):
        return "Seagate"
    if model.startswith("TOSHIBA"):
        return "Toshiba"
    raise ValueError(f"unknown vendor for model {model!r}")


# ---- metrics ---------------------------------------------------------------


def select_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float | None, float]:
    """Return ``(threshold, fpr)`` for the one-sided rule of the module docstring.

    A row is flagged when its score is strictly above ``threshold``. ``fpr`` is
    the false-positive rate on these rows. ``threshold`` is ``None`` when no
    row is above it, i.e. the target is unattainable.
    """
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed = int(np.floor(target_fpr * len(negatives)))
    threshold = -np.inf if allowed >= len(negatives) else float(negatives[allowed])
    if not (scores > threshold).any():
        return None, 0.0
    return threshold, float(np.mean(negatives > threshold))


def equal_frequency_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Return a bin index per score from quantile edges; equal scores always share a bin.

    The number of bins is ``n_bins`` unless ties collapse edges, in which case it is smaller.
    """
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, scores, side="left")


def reliability_table(scores: np.ndarray, labels: np.ndarray, n_bins: int) -> list[dict]:
    """Return one row per equal-frequency bin: rows, positives, mean score, observed rate."""
    bins = equal_frequency_bins(scores, n_bins)
    table = []
    for b in np.unique(bins):
        inside = bins == b
        table.append({"bin": int(b), "rows": int(inside.sum()), "positives": int(labels[inside].sum()),
                      "mean_pred": float(scores[inside].mean()), "observed": float(labels[inside].mean())})
    return table


def bin_count(positives: int) -> int:
    """Number of reliability bins: ``RELIABILITY_BINS``, reduced to keep ``MIN_FAILURES_PER_BIN`` failures per bin."""
    return int(max(1, min(RELIABILITY_BINS, positives // MIN_FAILURES_PER_BIN)))


class Ranked:
    """Rows of one cell and predictor sorted by descending score, for weighted metrics."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray, drives: np.ndarray) -> None:
        order = np.argsort(-scores, kind="stable")
        self.score = scores[order]
        self.label = labels[order].astype(np.float64)
        self.drive = drives[order]
        self.ends = np.append(np.flatnonzero(self.score[1:] != self.score[:-1]), len(self.score) - 1)

    def above(self, threshold: float | None) -> int:
        """Number of rows with a score strictly above ``threshold``."""
        if threshold is None:
            return 0
        return int(np.searchsorted(-self.score, -threshold, side="left"))

    def metrics(self, counts: np.ndarray, flagged: list[int]) -> tuple[float, list[tuple[float, float]]]:
        """Return ``(weighted PR-AUC, [(recall, fpr) per entry of flagged])`` for per-drive ``counts``."""
        w = counts[self.drive]
        cum_all = np.cumsum(w)
        cum_pos = np.cumsum(w * self.label)
        tp, seen = cum_pos[self.ends], cum_all[self.ends]
        positives = cum_pos[-1]
        negatives = cum_all[-1] - positives
        if positives == 0:
            return float("nan"), [(float("nan"), float("nan"))] * len(flagged)
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        recall = tp / positives
        ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
        out = []
        for k in flagged:
            if k == 0:
                out.append((float("nan"), float("nan")))
            else:
                out.append((float(cum_pos[k - 1] / positives),
                            float((cum_all[k - 1] - cum_pos[k - 1]) / negatives) if negatives > 0 else float("nan")))
        return ap, out


def brier_weighted(drive_error: np.ndarray, drive_rows: np.ndarray, counts: np.ndarray) -> float:
    """Brier score for per-drive ``counts`` from per-drive squared-error sums and row counts."""
    return float(counts @ drive_error / (counts @ drive_rows))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def cells(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the evaluation cells as name -> boolean row mask."""
    out = {"pooled": np.ones(len(frame), dtype=bool)}
    for vendor in sorted(frame["vendor"].unique()):
        out[f"vendor={vendor}"] = (frame["vendor"] == vendor).to_numpy()
    for model in sorted(frame[fit_v1.MODEL_COLUMN].unique()):
        out[f"model={model}"] = (frame[fit_v1.MODEL_COLUMN] == model).to_numpy()
    return out


def evaluate_cell(index: int, test: pd.DataFrame, val: pd.DataFrame, test_mask: np.ndarray,
                  val_mask: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Compute point estimates, thresholds and drive-bootstrap intervals for one cell."""
    t = test[test_mask].reset_index(drop=True)
    v = val[val_mask].reset_index(drop=True)
    y = t["label"].to_numpy()
    drives = pd.factorize(t["serial_number"])[0]
    n_drives = int(drives.max()) + 1
    failed_drives = int(len(np.unique(drives[y == 1])))
    base = float(y.mean())
    cell = {"rows": int(len(t)), "positives": int(y.sum()), "drives": n_drives, "failed_drives": failed_drives,
            "base_rate": base, "thin": failed_drives < MIN_FAILED_DRIVES, "predictors": {}}
    val_rate = float(v["label"].mean())
    ones = np.ones(n_drives)
    rng = np.random.default_rng([BOOTSTRAP_SEED, index])
    draws = [np.bincount(rng.integers(0, n_drives, n_drives), minlength=n_drives).astype(np.float64)
             for _ in range(resamples)]
    rows_per_drive = np.bincount(drives, minlength=n_drives).astype(np.float64)

    for name in PREDICTORS:
        score = t[name].to_numpy() if name != BASELINE else np.full(len(t), base)
        val_score = v[name].to_numpy() if name != BASELINE else np.full(len(v), val_rate)
        ranked = Ranked(score, y, drives)
        thresholds = [select_threshold(val_score, v["label"].to_numpy(), target) for target in FPR_TARGETS]
        flagged = [ranked.above(th) for th, _ in thresholds]
        ap, rf = ranked.metrics(ones, flagged)
        error = np.bincount(drives, weights=(score - y) ** 2, minlength=n_drives)
        brier = brier_weighted(error, rows_per_drive, ones)
        boot_ap, boot_brier = np.empty(resamples), np.empty(resamples)
        boot_recall = np.full((len(FPR_TARGETS), resamples), np.nan)
        for r, counts in enumerate(draws):
            boot_ap[r], rec = ranked.metrics(counts, flagged)
            boot_brier[r] = brier_weighted(error, rows_per_drive, counts)
            for j, (recall, _) in enumerate(rec):
                boot_recall[j, r] = recall
        entry = {
            "pr_auc": ap, "pr_auc_ci": percentile_interval(boot_ap),
            "brier": brier, "brier_ci": percentile_interval(boot_brier),
            "targets": [],
            "reliability": [] if name == BASELINE else reliability_table(score, y, bin_count(int(y.sum()))),
            "_boot_ap": boot_ap, "_boot_brier": boot_brier,
        }
        for j, target in enumerate(FPR_TARGETS):
            threshold, val_fpr = thresholds[j]
            recall, test_fpr = rf[j]
            entry["targets"].append({
                "target_fpr": target, "threshold": threshold, "val_fpr": val_fpr,
                "attainable": threshold is not None, "recall": None if threshold is None else recall,
                "recall_ci": percentile_interval(boot_recall[j]), "test_fpr": None if threshold is None else test_fpr,
            })
        cell["predictors"][name] = entry

    a, b = cell["predictors"][V0], cell["predictors"][V1]
    cell["diff_v1_minus_v0"] = {
        "pr_auc": b["pr_auc"] - a["pr_auc"], "pr_auc_ci": percentile_interval(b["_boot_ap"] - a["_boot_ap"]),
        "brier": b["brier"] - a["brier"], "brier_ci": percentile_interval(b["_boot_brier"] - a["_boot_brier"]),
    }
    for entry in cell["predictors"].values():
        del entry["_boot_ap"], entry["_boot_brier"]
    return cell


def verdict(cell: Mapping) -> str:
    """Return a plain verdict for a cell from the paired drive-bootstrap differences (v1 minus v0)."""
    d = cell["diff_v1_minus_v0"]
    ap_lo, ap_hi = d["pr_auc_ci"]
    br_lo, br_hi = d["brier_ci"]
    ap_better, ap_worse = ap_lo > 0, ap_hi < 0
    br_better, br_worse = br_hi < 0, br_lo > 0
    if ap_better and br_better:
        word = "v1 better than v0 on PR-AUC and Brier"
    elif ap_worse and br_worse:
        word = "v0 better than v1 on PR-AUC and Brier"
    elif (ap_better or br_better) and (ap_worse or br_worse):
        word = "mixed: v1 and v0 each lead on one of PR-AUC and Brier"
    elif ap_better or br_better:
        word = "v1 ahead on " + ("PR-AUC" if ap_better else "Brier") + ", no difference at 95% on the other"
    elif ap_worse or br_worse:
        word = "v0 ahead on " + ("PR-AUC" if ap_worse else "Brier") + ", no difference at 95% on the other"
    else:
        word = "no difference at 95% on PR-AUC or Brier"
    return (f"{word} (PR-AUC v1-v0 {d['pr_auc']:+.5f} [{ap_lo:+.5f}, {ap_hi:+.5f}]; "
            f"Brier v1-v0 {d['brier']:+.3e} [{br_lo:+.3e}, {br_hi:+.3e}])")


# ---- output ----------------------------------------------------------------


def _ci(pair: tuple[float, float], fmt: str) -> str:
    return f"[{pair[0]:{fmt}}, {pair[1]:{fmt}}]"


def result_lines(results: Mapping[str, dict]) -> list[str]:
    lines = []
    for name, cell in results.items():
        flag = f" FEW_FAILED_DRIVES(<{MIN_FAILED_DRIVES})" if cell["thin"] else ""
        lines.append(f"cell={name} rows={cell['rows']} positives={cell['positives']} drives={cell['drives']} "
                     f"failed_drives={cell['failed_drives']} base_rate={cell['base_rate']:.6f}{flag}")
        for predictor, e in cell["predictors"].items():
            lines.append(f"  {predictor}: pr_auc={e['pr_auc']:.5f} ci95={_ci(e['pr_auc_ci'], '.5f')} "
                         f"brier={e['brier']:.6e} ci95={_ci(e['brier_ci'], '.6e')}")
            for t in e["targets"]:
                if not t["attainable"]:
                    lines.append(f"    fpr_target={t['target_fpr']:.3%}: unattainable on val "
                                 f"(no val row above the threshold)")
                else:
                    lines.append(f"    fpr_target={t['target_fpr']:.3%}: threshold>{t['threshold']:.6e} "
                                 f"val_fpr={t['val_fpr']:.5%} test_fpr={t['test_fpr']:.5%} "
                                 f"recall={t['recall']:.4f} ci95={_ci(t['recall_ci'], '.4f')}")
        lines.append(f"  diff v1-v0: {verdict(cell)}")
        for predictor, e in cell["predictors"].items():
            if not e["reliability"]:
                continue
            lines.append(f"  reliability {predictor} bins={len(e['reliability'])}")
            for row in e["reliability"]:
                few = " few_failures" if row["positives"] < MIN_FAILURES_PER_BIN else ""
                lines.append(f"    bin={row['bin']:>2} rows={row['rows']:>8} positives={row['positives']:>6} "
                             f"mean_pred={row['mean_pred']:.6f} observed={row['observed']:.6f}{few}")
    return lines


def evidence_block(results: Mapping[str, dict], meta: Mapping) -> str:
    lines = [f"- Device: {meta['device']}; per-row cross-check on {meta['cross_rows']} held-out rows, "
             f"largest absolute difference {meta['cross_diff']:.3e}.",
             f"- Rows scored: held-out {meta['test_rows']:,}, validation {meta['val_rows']:,}. "
             f"Drive bootstrap resamples per cell: {BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED}.",
             "", "| Cell | Failed drives | Predictor | PR-AUC [95%] | Brier [95%] | Recall @1% FPR [95%] | "
             "Recall @0.1% FPR [95%] |", "|---|---|---|---|---|---|---|"]
    for name, cell in results.items():
        mark = " (few)" if cell["thin"] else ""
        for predictor, e in cell["predictors"].items():
            recalls = []
            for t in e["targets"]:
                recalls.append("unattainable" if not t["attainable"]
                               else f"{t['recall']:.3f} {_ci(t['recall_ci'], '.3f')}")
            lines.append(f"| {name} | {cell['failed_drives']}{mark} | {predictor} | "
                         f"{e['pr_auc']:.4f} {_ci(e['pr_auc_ci'], '.4f')} | "
                         f"{e['brier']:.3e} {_ci(e['brier_ci'], '.3e')} | {recalls[0]} | {recalls[1]} |")
    lines += ["", "Frozen thresholds (flag when score is strictly above), chosen on validation rows of each cell:", ""]
    lines += ["| Cell | Predictor | Target FPR | Threshold | Val FPR | Test FPR |", "|---|---|---|---|---|---|"]
    for name, cell in results.items():
        for predictor, e in cell["predictors"].items():
            for t in e["targets"]:
                if t["attainable"]:
                    lines.append(f"| {name} | {predictor} | {t['target_fpr']:.1%} | {t['threshold']:.4e} | "
                                 f"{t['val_fpr']:.4%} | {t['test_fpr']:.4%} |")
                else:
                    lines.append(f"| {name} | {predictor} | {t['target_fpr']:.1%} | unattainable | - | - |")
    pooled = results["pooled"]["predictors"]
    for predictor in (V0, V1):
        rows = pooled[predictor]["reliability"]
        lines += ["", f"Pooled reliability, {predictor} ({len(rows)} equal-frequency bins):", "",
                  "| Bin | Rows | Positives | Mean prediction | Observed rate |", "|---|---|---|---|---|"]
        lines += [f"| {r['bin']} | {r['rows']:,} | {r['positives']:,} | {r['mean_pred']:.6f} | "
                  f"{r['observed']:.6f} |" for r in rows]
    return "\n".join(lines)


def verdict_block(results: Mapping[str, dict]) -> str:
    lines = [f"- Pooled: {verdict(results['pooled'])}."]
    for name, cell in results.items():
        if name != "pooled":
            lines.append(f"- {name}{' (few failed drives)' if cell['thin'] else ''}: {verdict(cell)}.")
    return "\n".join(lines)


def write_record(results: Mapping[str, dict], meta: Mapping, path: Path = RECORD) -> None:
    text = path.read_text(encoding="utf-8")
    for name, body in (("evidence", evidence_block(results, meta)), ("verdict", verdict_block(results))):
        pattern = re.compile(rf"<!-- a4:{name} -->.*?<!-- /a4:{name} -->", re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"{path.name} has no a4:{name} block")
        text = pattern.sub(lambda _m: BLOCK.format(name=name, body=body), text)
    path.write_text(text, encoding="utf-8")


def run(write: bool = False) -> dict:
    started = time.time()
    device = select_device()
    print(f"device={device} bootstrap_resamples={BOOTSTRAP_RESAMPLES} seed={BOOTSTRAP_SEED}")
    first = next(load_test_split(max_chunks=None))
    diff = cross_check(first, device=device)
    print(f"cross_check rows={min(CROSS_CHECK_ROWS, len(first))} max_abs_diff={diff:.3e}")
    if diff > CROSS_CHECK_TOLERANCE:
        raise RuntimeError(f"batch path differs from the per-row interface by {diff:.3e}")
    test = score_chunks(load_test_split(), device)
    print(f"test_rows={len(test)} test_positives={int(test['label'].sum())} seconds={time.time() - started:.0f}")
    val = score_chunks(load_val_split(), device)
    print(f"val_rows={len(val)} val_positives={int(val['label'].sum())} seconds={time.time() - started:.0f}")
    test_cells, val_cells = cells(test), cells(val)
    results = {}
    for index, name in enumerate(test_cells):
        results[name] = evaluate_cell(index, test, val, test_cells[name], val_cells[name])
        print(f"cell={name} done seconds={time.time() - started:.0f}")
    for line in result_lines(results):
        print(line)
    meta = {"device": device, "cross_rows": min(CROSS_CHECK_ROWS, len(first)), "cross_diff": diff,
            "test_rows": len(test), "val_rows": len(val)}
    if write:
        write_record(results, meta)
        print(f"record={RECORD.name} updated")
    print(f"wall_clock_seconds={time.time() - started:.0f}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-record", action="store_true")
    run(parser.parse_args(argv).write_record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
