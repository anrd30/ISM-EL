"""How unusual is ``profile_diurnal_long``, and a long-window workload not selected for its daily cycle.

Usage (repo root, on a checkout with the Tencent pipeline outputs in
``data_pipeline/data/processed/``)::

    python -m agent.workload_selection_check --out runs/selection_check

Reads ``tencent_disk_runs.parquet``, ``tencent_disk_summary.parquet`` and
``tencent_load_clean.parquet`` exactly as ``data_pipeline/tencent/build_profiles.py``
does and calls its own functions (``eligible_disks``, ``collect``,
``candidate_frame``, ``build_profile``, ``rank_uids``, ``PROFILE_SPECS``).
Nothing in ``data_pipeline/`` is written; ``manifest.json`` and
``PREPROCESSING.md`` are untouched. Outputs go to ``--out``.

Step 0, reproduction. The diurnal rule is re-applied to the 14-day window. The
selected volumes must equal the ``disk_uids`` of
``profiles/profile_diurnal_long.meta.json`` and the aggregate must equal
``profiles/profile_diurnal_long.csv`` row for row; otherwise the script stops,
because nothing it reports would describe the committed profile.

Step 1, the candidate distribution. ``daily_strength`` (share of a volume's
total-bandwidth variance explained by its time-of-day mean, over the volume's
whole clean trace) for every candidate the long-window diurnal ranking ranks:
eligible for the 14-day window, not excluded by the bytes-per-I/O rule, at
least ``MIN_DAYS`` of clean rows. Reported as quantiles and a 0.05-wide
histogram, with the selected volumes' values and percentile ranks. The same
statistic is computed for each aggregate written in Step 2.

Step 2, alternative long windows. The same 14-day window and the same
procedure, with the volumes ranked by the ``bursty`` and the ``idle`` rules of
``PROFILE_SPECS`` instead of ``diurnal``, and N tuned against each rule's own
target p95. Each is written as ``profile_<name>_long.csv`` with a sidecar, and
as ``_train`` / ``_holdout`` halves cut at the same row as
``profile_diurnal_long`` (first 10 of 14 days), so the evaluation protocol of
``agent/FORECAST_PPO_DECISION.md`` applies unchanged.

Writes ``selection_check.json``, ``selection_check.md``,
``daily_strength_candidates.csv`` and the CSVs above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.tencent import build_profiles as bp
from data_pipeline.tencent import extract_load as el

__all__ = ["ALTERNATIVE_RULES", "aggregate_daily_strength", "main"]

#: ``PROFILE_SPECS`` rules used for the alternative long windows.
ALTERNATIVE_RULES: tuple[str, ...] = ("bursty", "idle")
_QUANTILES: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def aggregate_daily_strength(frame: pd.DataFrame) -> dict[str, float]:
    """``daily_strength`` and forecast errors of a replay frame's total bandwidth.

    Returns the time-of-day variance share (``build_profiles.daily_stats``), the
    one-interval RMSE of the last value, and the RMSE of the same-time-yesterday
    value as a forecast of the next interval, in MB/s.
    """
    total = (frame["read_bw_kbps"] + frame["write_bw_kbps"]).to_numpy(dtype="float64")
    stats = bp.daily_stats(frame["t_seconds"].to_numpy(dtype="int64"), total)
    mbps = total / bp.KBPS_PER_MBPS
    day = bp.DAY_ROWS
    return {
        "daily_strength": float(stats["daily_strength"]),
        "rmse_last_value_mbps": float(np.sqrt(np.mean(np.diff(mbps) ** 2))),
        "rmse_same_time_yesterday_mbps": float(np.sqrt(np.mean((mbps[day:] - mbps[:-day]) ** 2))),
        "mean_mbps": float(mbps.mean()),
        "p95_mbps": float(np.percentile(mbps, 95)),
        "max_mbps": float(mbps.max()),
    }


def _write_replay(frame: pd.DataFrame, path: Path, meta: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(frame: pd.DataFrame, path: Path, meta: dict) -> dict[str, str]:
    cut = bp.split_row(len(frame))
    out = {}
    for suffix, part in (("train", frame.iloc[:cut]), ("holdout", frame.iloc[cut:])):
        part = part.reset_index(drop=True).copy()
        part["t_seconds"] = part["t_seconds"] - int(part["t_seconds"].iloc[0])
        part_meta = dict(meta, profile=f"{meta['profile']}_{suffix}", rows=len(part), split_from=path.name)
        out[suffix] = _write_replay(part, path.with_name(f"{path.stem}_{suffix}.csv"), part_meta)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(prog="python -m agent.workload_selection_check", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_parquet(bp.IN_RUNS)
    summary = pd.read_parquet(bp.IN_SUMMARY)
    start = el.shared_window_stats(runs, bp.LONG_WINDOW_ROWS)["largest_group_start_ts"]
    if start is None:
        raise SystemExit("no gap-free 14-day window shared by any volume")
    uids = bp.eligible_disks(runs, start, bp.LONG_WINDOW_ROWS)
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    windows, periodicity = bp.collect(pf, groups, uids, start, bp.LONG_WINDOW_ROWS)
    cand = bp.candidate_frame(summary, periodicity)
    position = {u: i for i, u in enumerate(uids)}

    # Step 0: reproduce the committed long window.
    committed_meta = json.loads((bp.ROOT / "profiles" / "profile_diurnal_long.meta.json").read_text(encoding="utf-8"))
    committed = pd.read_csv(bp.ROOT / "profiles" / "profile_diurnal_long.csv")
    diurnal = bp.build_profile("diurnal", bp.PROFILE_SPECS["diurnal"], cand, windows, position, start,
                               bp.LONG_WINDOW_ROWS)
    same_uids = list(diurnal["uids"]) == list(committed_meta["disk_uids"])
    same_rows = diurnal["frame"].reset_index(drop=True).equals(committed.astype(diurnal["frame"].dtypes.to_dict()))
    print(f"reproduction start_ts={start} volumes_with_window={len(uids)} n={diurnal['n']} "
          f"same_uids={same_uids} same_rows={same_rows}", flush=True)
    if not (same_uids and same_rows):
        raise SystemExit("the diurnal rule does not reproduce profile_diurnal_long; nothing below would describe it")

    # Step 1: daily_strength over the ranked candidate pool.
    pool = cand[~cand["implausible_io"]].dropna(subset=["daily_strength"]).copy()
    selected = set(diurnal["uids"])
    pool["selected"] = pool["disk_uid"].isin(selected)
    pool["percentile_rank"] = pool["daily_strength"].rank(pct=True) * 100.0
    pool.sort_values("daily_strength", ascending=False)[
        ["disk_uid", "daily_strength", "days", "mean_total_bw_kbps", "burstiness", "selected", "percentile_rank"]
    ].to_csv(out / "daily_strength_candidates.csv", index=False, lineterminator="\n")
    values = pool["daily_strength"].to_numpy()
    sel = pool[pool["selected"]]
    bins = np.arange(0.0, 1.0001, 0.05)
    hist, _ = np.histogram(values, bins=bins)
    step1 = {
        "window_start_ts": int(start),
        "volumes_with_window": len(uids),
        "excluded_by_bytes_per_io_rule": int(cand["implausible_io"].sum()),
        "ranked_candidates": int(len(pool)),
        "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in _QUANTILES},
        "mean": float(values.mean()),
        "selected_n": int(len(sel)),
        "selected_min": float(sel["daily_strength"].min()),
        "selected_median": float(sel["daily_strength"].median()),
        "selected_max": float(sel["daily_strength"].max()),
        "selected_min_percentile_rank": float(sel["percentile_rank"].min()),
        "share_of_pool_at_or_above_selected_min": float((values >= sel["daily_strength"].min()).mean()),
        "histogram_bins": [float(b) for b in bins],
        "histogram_counts": [int(c) for c in hist],
    }

    # Step 2: alternative long windows over the same window and candidates.
    workloads = {"diurnal_long": {"rule": bp.PROFILE_SPECS["diurnal"]["rule"], "n": diurnal["n"],
                                  **aggregate_daily_strength(diurnal["frame"])}}
    for rule in ALTERNATIVE_RULES:
        spec = bp.PROFILE_SPECS[rule]
        built = bp.build_profile(rule, spec, cand, windows, position, start, bp.LONG_WINDOW_ROWS)
        name = f"{rule}_long"
        path = out / f"profile_{name}.csv"
        meta = {
            "profile": name,
            "synthetic": False,
            "source": "Tencent Cloud Volume Workload Traces",
            "label": f"aggregate of {built['n']} {spec['label']} volumes over the 14-day window of "
                     f"profile_diurnal_long (selected extremes, not a representative sample)",
            "disclosure": spec["disclosure"],
            "selection_rule": spec["rule"],
            "target_p95_mbps": spec["target_mbps"],
            "n_disks": built["n"],
            "disk_uids": list(built["uids"]),
            "window_start_ts": int(start),
            "rows": len(built["frame"]),
            "written_by": "agent/workload_selection_check.py",
        }
        sha = _write_replay(built["frame"], path, meta)
        halves = _split(built["frame"], path, meta)
        workloads[name] = {"rule": spec["rule"], "n": built["n"], "sha256": sha,
                           "train_sha256": halves["train"], "holdout_sha256": halves["holdout"],
                           "overlap_with_diurnal_long": len(set(built["uids"]) & selected),
                           **aggregate_daily_strength(built["frame"])}
        print(f"workload={name} n={built['n']} daily_strength={workloads[name]['daily_strength']:.3f} "
              f"mean_mbps={workloads[name]['mean_mbps']:.1f} p95_mbps={workloads[name]['p95_mbps']:.1f}", flush=True)
    del windows

    result = {"step1": step1, "step2": workloads}
    (out / "selection_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    q = step1["quantiles"]
    lines = [
        "# daily_strength of the long-window candidate pool",
        "",
        f"- window start {start}, volumes with the window {len(uids)}, ranked candidates {len(pool)}",
        f"- pool: median {q['q50']:.4f}, p90 {q['q90']:.4f}, p99 {q['q99']:.4f}, max {q['q100']:.4f}",
        f"- selected {len(sel)}: min {step1['selected_min']:.4f}, median {step1['selected_median']:.4f}, "
        f"max {step1['selected_max']:.4f}; lowest selected is at percentile rank "
        f"{step1['selected_min_percentile_rank']:.2f}",
        "",
        "| workload | rule | n | daily_strength | RMSE last value (MB/s) | RMSE same time yesterday (MB/s) | mean MB/s | p95 MB/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, w in workloads.items():
        lines.append(f"| {name} | {w['rule']} | {w['n']} | {w['daily_strength']:.4f} | {w['rmse_last_value_mbps']:.2f} | "
                     f"{w['rmse_same_time_yesterday_mbps']:.2f} | {w['mean_mbps']:.1f} | {w['p95_mbps']:.1f} |")
    (out / "selection_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
