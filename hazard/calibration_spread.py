"""Drive-bootstrap spread of the v1 calibration residual on the held-out split.

Usage (from the repository root)::

    python -m hazard.calibration_spread

Read-only analysis; it writes ``hazard/calibration_spread.json`` and nothing
else. The held-out rows are read through ``hazard.evaluate_v1.load_test_split``
and scored with ``hazard.evaluate_v1.score_chunks`` (same path as the A4
evaluation); the predictor is the pooled v1 probability.

Point estimate: ``hazard.evaluate_v1.reliability_table`` with
``bin_count(positives)`` equal-frequency bins; the residual of a bin is
``observed - mean_pred``.

Bootstrap: the pooled cell of ``evaluate_v1.evaluate_cell`` (cell index 0) is
reproduced exactly: seed ``[BOOTSTRAP_SEED, 0]``, ``BOOTSTRAP_RESAMPLES``
resamples of whole drives drawn with replacement, carried as per-drive draw
counts. Within a resample every row of a drive is weighted by its draw count and
the bins are the ones of the point estimate (fixed edges, so a bin is the same
set of score values in every resample); the bin's weighted mean prediction and
weighted observed rate give that resample's residual. Re-deriving equal-
frequency edges per resample is not done: v1's scores contain many tied values
and the bins would not be comparable across resamples.

Per bin: point residual; mean, standard error (standard deviation across
resamples) and the 2.5th, 50th and 97.5th percentiles of the resampled
residual. The log ratio ``ln(observed / mean_pred)`` is reported in the same
way (resamples in which a bin has no positive row give no log ratio and are
excluded; their number is recorded).

Headline numbers (weights: rows per bin in the point estimate):

* ``mean_bin_se``: row-weighted mean of the per-bin standard errors of the residual;
* ``mean_abs_residual``: row-weighted mean of ``|residual|``, point estimate and
  bootstrap 2.5th / 50th / 97.5th percentiles;
* ``weighted_mean_residual``: row-weighted mean of the signed residual, point
  estimate, standard error and percentiles;
* the same three for the log ratio (``mean_bin_se`` is over the bins with a
  finite log ratio in every resample; the weights of other bins are dropped).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from hazard import evaluate_v1 as ev

OUT_PATH = Path(__file__).resolve().parent / "calibration_spread.json"
POOLED_CELL_INDEX = 0
PERCENTILES = (2.5, 50.0, 97.5)


def stats(values: np.ndarray) -> dict:
    """Mean, standard error and percentiles of the finite entries of ``values``."""
    finite = values[np.isfinite(values)]
    p = np.percentile(finite, PERCENTILES)
    return {"mean": float(finite.mean()), "se": float(finite.std(ddof=1)),
            "p2_5": float(p[0]), "p50": float(p[1]), "p97_5": float(p[2]), "n": int(len(finite))}


def drive_draws(n_drives: int, resamples: int = ev.BOOTSTRAP_RESAMPLES) -> list[np.ndarray]:
    """Return the per-drive draw counts of the pooled cell of ``evaluate_v1.evaluate_cell``."""
    rng = np.random.default_rng([ev.BOOTSTRAP_SEED, POOLED_CELL_INDEX])
    return [np.bincount(rng.integers(0, n_drives, n_drives), minlength=n_drives).astype(np.float64)
            for _ in range(resamples)]


def spread(scores: np.ndarray, labels: np.ndarray, drives: np.ndarray,
           resamples: int = ev.BOOTSTRAP_RESAMPLES) -> dict:
    """Return the per-bin and pooled residual spread for per-row ``scores``, ``labels`` and ``drives``."""
    n_drives = int(drives.max()) + 1
    n_bins = ev.bin_count(int(labels.sum()))
    point = ev.reliability_table(scores, labels, n_bins)
    bins = np.unique(ev.equal_frequency_bins(scores, n_bins), return_inverse=True)[1]
    k = len(point)
    cell = bins * n_drives + drives
    rows_bd = np.bincount(cell, minlength=k * n_drives).reshape(k, n_drives).astype(np.float64)
    pos_bd = np.bincount(cell, weights=labels, minlength=k * n_drives).reshape(k, n_drives)
    pred_bd = np.bincount(cell, weights=scores, minlength=k * n_drives).reshape(k, n_drives)

    def table(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = rows_bd @ w
        return rows, pos_bd @ w / rows, pred_bd @ w / rows

    ones = np.ones(n_drives)
    rows0, obs0, pred0 = table(ones)
    for i, row in enumerate(point):
        if (rows0[i] != row["rows"] or not np.isclose(obs0[i], row["observed"], rtol=1e-12, atol=0.0)
                or not np.isclose(pred0[i], row["mean_pred"], rtol=1e-12, atol=0.0)):
            raise RuntimeError(f"bin {i} differs from evaluate_v1.reliability_table")
    resid0 = obs0 - pred0
    weight = rows0 / rows0.sum()

    resid = np.empty((resamples, k))
    log_ratio = np.empty((resamples, k))
    for r, w in enumerate(drive_draws(n_drives, resamples)):
        _, obs, pred = table(w)
        resid[r] = obs - pred
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio[r] = np.where(obs > 0, np.log(obs / pred), np.nan)

    bins_out = []
    for i, row in enumerate(point):
        lr = float(np.log(row["observed"] / row["mean_pred"])) if row["observed"] > 0 else float("nan")
        bins_out.append({
            "bin": i, "rows": row["rows"], "positives": row["positives"],
            "mean_pred": row["mean_pred"], "observed": row["observed"],
            "residual": float(resid0[i]), "residual_boot": stats(resid[:, i]),
            "log_ratio": lr, "log_ratio_boot": stats(log_ratio[:, i]),
            "log_ratio_excluded_resamples": int((~np.isfinite(log_ratio[:, i])).sum()),
        })

    abs_boot = np.abs(resid) @ weight
    signed_boot = resid @ weight
    se_bin = resid.std(axis=0, ddof=1)
    full = np.isfinite(log_ratio).all(axis=0) & np.isfinite(np.log(np.where(obs0 > 0, obs0 / pred0, np.nan)))
    lw = rows0[full] / rows0[full].sum()
    log0 = np.log(obs0[full] / pred0[full])
    lr_full = log_ratio[:, full]
    headline = {
        "weights": "rows per bin in the point estimate",
        "mean_bin_se": float(weight @ se_bin),
        "mean_abs_residual": {"point": float(weight @ np.abs(resid0)), "boot": stats(abs_boot)},
        "weighted_mean_residual": {"point": float(weight @ resid0), "boot": stats(signed_boot)},
        "log_ratio": {
            "bins_used": int(full.sum()),
            "mean_bin_se": float(lw @ lr_full.std(axis=0, ddof=1)),
            "mean_abs": {"point": float(lw @ np.abs(log0)), "boot": stats(np.abs(lr_full) @ lw)},
            "weighted_mean": {"point": float(lw @ log0), "boot": stats(lr_full @ lw)},
        },
    }
    return {"bins": bins_out, "headline": headline}


def format_report(result: dict) -> list[str]:
    lines = ["bin | rows | positives | mean_pred | observed | residual | boot se | boot 2.5-97.5 residual"]
    for b in result["bins"]:
        r = b["residual_boot"]
        lines.append(f"{b['bin']:>3} | {b['rows']:>8} | {b['positives']:>5} | {b['mean_pred']:.6f} | "
                     f"{b['observed']:.6f} | {b['residual']:+.6f} | {r['se']:.6f} | "
                     f"[{r['p2_5']:+.6f}, {r['p97_5']:+.6f}]")
    h = result["headline"]
    a, s, lr = h["mean_abs_residual"], h["weighted_mean_residual"], h["log_ratio"]
    lines += [
        f"headline mean_bin_se (row-weighted mean of per-bin residual SE): {h['mean_bin_se']:.6f}",
        f"headline mean_abs_residual: point {a['point']:.6f} boot [{a['boot']['p2_5']:.6f}, {a['boot']['p97_5']:.6f}]",
        f"headline weighted_mean_residual: point {s['point']:+.6f} se {s['boot']['se']:.6f} "
        f"boot [{s['boot']['p2_5']:+.6f}, {s['boot']['p97_5']:+.6f}]",
        f"log_ratio (bins_used={lr['bins_used']}): mean_bin_se {lr['mean_bin_se']:.4f} "
        f"mean_abs point {lr['mean_abs']['point']:.4f} "
        f"boot [{lr['mean_abs']['boot']['p2_5']:.4f}, {lr['mean_abs']['boot']['p97_5']:.4f}] "
        f"weighted_mean point {lr['weighted_mean']['point']:+.4f} se {lr['weighted_mean']['boot']['se']:.4f}",
    ]
    return lines


def run(out_path: Path = OUT_PATH) -> dict:
    started = time.time()
    device = ev.select_device()
    print(f"device={device} bootstrap_resamples={ev.BOOTSTRAP_RESAMPLES} seed={ev.BOOTSTRAP_SEED}")
    first = next(ev.load_test_split(max_chunks=None))
    diff = ev.cross_check(first, device=device)
    print(f"cross_check rows={min(ev.CROSS_CHECK_ROWS, len(first))} max_abs_diff={diff:.3e}")
    if diff > ev.CROSS_CHECK_TOLERANCE:
        raise RuntimeError(f"batch path differs from the per-row interface by {diff:.3e}")
    test = ev.score_chunks(ev.load_test_split(), device)
    print(f"test_rows={len(test)} test_positives={int(test['label'].sum())} seconds={time.time() - started:.0f}")
    labels = test["label"].to_numpy()
    drives = ev.pd.factorize(test["serial_number"])[0]
    result = spread(test[ev.V1].to_numpy(), labels, drives)
    result["meta"] = {
        "predictor": ev.V1, "split": "held-out", "rows": int(len(test)), "positives": int(labels.sum()),
        "drives": int(drives.max()) + 1, "bins": len(result["bins"]),
        "bootstrap_resamples": ev.BOOTSTRAP_RESAMPLES, "bootstrap_seed": [ev.BOOTSTRAP_SEED, POOLED_CELL_INDEX],
        "binning": "equal-frequency edges of the point estimate, fixed across resamples",
        "residual": "observed - mean_pred",
    }
    for line in format_report(result):
        print(line)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={out_path} wall_clock_seconds={time.time() - started:.0f}")
    return result


if __name__ == "__main__":
    run()
    sys.exit(0)
