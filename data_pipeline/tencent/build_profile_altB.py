"""Further 14-day profiles, each disjoint from the earlier ones in time window and in volumes.

``altB`` is the second periodic profile; ``C`` (diurnal rule) and ``D`` (volumes in random order) are the
fresh populations of the pre-registered evaluation, see ``build_population``.

Second 14-day periodic profile, disjoint from ``diurnal_long`` in time window and in volumes.

``diurnal_long`` (``build_profiles.build_long_window``) uses the window shared by the most
volumes. This module applies the same construction (``eligible_disks``, ``collect``,
``candidate_frame``, ``build_profile`` with the diurnal spec, the same ``DIURNAL_TARGET_MBPS``
ceiling, the six-column replay schema with ``t_seconds`` from 0, ``validate_replay_frame``) to a
different window and candidate pool:

  window   the start covered by a gap-free run of ``LONG_WINDOW_ROWS`` samples in the most volumes
           among starts whose window shares no sample time with the ``diurnal_long`` window
  volumes  every ``disk_uid`` listed in ``profile_diurnal_long.meta.json`` is removed from the
           candidate pool before ranking

Writes ``profiles/profile_diurnal_long_altB.csv`` and ``.meta.json`` and the section
``## B5b`` of ``PREPROCESSING.md``. Existing profile files are not read for writing.

Usage::

    python -I -m data_pipeline.tencent.build_profile_altB [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.tencent import build_profiles as bp

NAME = "diurnal_long_altB"
REFERENCE_NAME = bp.LONG_NAME
SECTION_TITLE = "## B5b — Independent periodic profile (diurnal_long_altB)"


def window_candidates(runs: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Every start (``start_ts``) with the number of volumes holding a gap-free run of ``rows`` samples.

    Starts are on the sampling grid of one offset class; a row is one maximal interval of starts
    with a constant count.
    """
    full = runs[runs["run_rows"] >= rows]
    out = []
    for cls in np.unique((full["run_start_ts"] % bp.STEP_S).to_numpy()):
        part = full[(full["run_start_ts"] % bp.STEP_S) == cls]
        lo = ((part["run_start_ts"].to_numpy() - cls) // bp.STEP_S).astype("int64")
        hi = lo + (part["run_rows"].to_numpy() - rows)
        points = np.unique(np.concatenate((lo, hi + 1)))
        delta = np.zeros(len(points), dtype="int64")
        np.add.at(delta, np.searchsorted(points, lo), 1)
        np.add.at(delta, np.searchsorted(points, hi + 1), -1)
        count = np.cumsum(delta)
        for i in range(len(points) - 1):
            out.append((int(points[i] * bp.STEP_S + cls), int(points[i + 1] - points[i]), int(count[i])))
    return pd.DataFrame(out, columns=["start_ts", "width_rows", "volumes"])


def time_disjoint_best(runs: pd.DataFrame, rows: int, ref_start: "int | Sequence[int]") -> tuple[int, pd.DataFrame]:
    """Start of the best-covered window sharing no sample time with any ``[ref, ref + rows)``; earliest on ties."""
    span = rows * bp.STEP_S
    refs = [int(ref_start)] if isinstance(ref_start, (int, np.integer)) else [int(r) for r in ref_start]
    cand = window_candidates(runs, rows)
    keep = pd.Series(True, index=cand.index)
    for ref in refs:
        keep &= (cand["start_ts"] + span <= ref) | (cand["start_ts"] >= ref + span)
    disjoint = cand[keep]
    disjoint = disjoint.sort_values(["volumes", "start_ts"], ascending=[False, True], kind="stable")
    if disjoint.empty:
        raise ValueError("no window disjoint in time from the reference window")
    return int(disjoint.iloc[0]["start_ts"]), cand


def build(write: bool = True) -> dict:
    ref_meta = json.loads((bp.PROFILES_DIR / f"profile_{REFERENCE_NAME}.meta.json").read_text(encoding="utf-8"))
    ref_start = int(ref_meta["window_start_ts"])
    excluded = list(ref_meta["disk_uids"])

    runs = pd.read_parquet(bp.IN_RUNS)
    start, cand_windows = time_disjoint_best(runs, bp.LONG_WINDOW_ROWS, ref_start)
    uids = bp.eligible_disks(runs, start, bp.LONG_WINDOW_ROWS)
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    windows, periodicity = bp.collect(pf, groups, uids, start, bp.LONG_WINDOW_ROWS)
    cand = bp.candidate_frame(pd.read_parquet(bp.IN_SUMMARY), periodicity)
    position = {u: i for i, u in enumerate(uids)}
    spec = bp.PROFILE_SPECS["diurnal"]

    # same construction without the volume exclusion, for the overlap record only
    unexcluded = bp.build_profile("diurnal", spec, cand, windows, position, start, bp.LONG_WINDOW_ROWS)
    pool = cand[~cand["disk_uid"].isin(excluded)]
    p = bp.build_profile("diurnal", spec, pool, windows, position, start, bp.LONG_WINDOW_ROWS)
    del windows

    rows = len(p["frame"])
    split = bp.split_row(rows)
    ref_start_cover = int(cand_windows.loc[cand_windows["start_ts"] == ref_start, "volumes"].max())
    w = {
        "name": NAME, "uids": p["uids"], "n": p["n"], "frame": p["frame"], "start_ts": start,
        "end_ts": start + (rows - 1) * bp.STEP_S, "rows": rows, "requested_rows": bp.LONG_WINDOW_ROWS,
        "volumes_with_window": len(uids), "ranked_candidates": p["candidates"],
        "p95_mbps": p["p95_mbps"], "mean_mbps": p["mean_mbps"], "max_mbps": p["max_mbps"],
        "target_mbps": p["target_mbps"], "split_row": split, "split_t_seconds": split * bp.STEP_S,
        "split_ts": start + split * bp.STEP_S,
    }
    sidecar = bp.long_sidecar(w)
    sidecar["label"] = (f"aggregate of {w['n']} {spec['label']} volumes over a shared "
                        f"{bp.LONG_WINDOW_DAYS}-day window disjoint in time and in volumes from "
                        f"{REFERENCE_NAME} ({bp.DISCLOSURE})")
    sidecar["reference_profile"] = REFERENCE_NAME
    sidecar["reference_window_start_ts"] = ref_start
    sidecar["reference_window_volumes"] = ref_start_cover
    sidecar["excluded_disk_uids"] = excluded
    sidecar["excluded_pool_size"] = int(len(pool))
    sidecar["overlap_with_reference_disks"] = len(set(p["uids"]) & set(excluded))
    sidecar["selection_without_exclusion"] = {
        "n_disks": unexcluded["n"], "p95_mbps": unexcluded["p95_mbps"],
        "overlap_with_reference_disks": len(set(unexcluded["uids"]) & set(excluded)),
    }
    sidecar["window_choice"] = (
        "the start covered by a gap-free run of the window length in the most volumes among starts "
        "whose window shares no sample time with the reference window")

    out = {"w": w, "sidecar": sidecar, "reference_start": ref_start, "window_table": cand_windows,
           "unexcluded": unexcluded, "excluded": excluded}
    if write:
        bp.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        csv = bp.PROFILES_DIR / f"profile_{NAME}.csv"
        meta = bp.PROFILES_DIR / f"profile_{NAME}.meta.json"
        w["frame"].to_csv(csv, index=False, lineterminator="\n")
        bp.write_json(meta, sidecar)
        out["csv"], out["meta"] = csv, meta
        out["csv_sha256"], out["meta_sha256"] = bp.sha256_of(csv), bp.sha256_of(meta)
    return out


def report_lines(r: dict) -> list[str]:
    w, s = r["w"], r["sidecar"]
    lines = [
        f"profile={NAME} n_disks={w['n']} start_ts={w['start_ts']} start_utc={bp.utc(w['start_ts'])} "
        f"end_ts={w['end_ts']} end_utc={bp.utc(w['end_ts'])} rows={w['rows']} requested_rows={w['requested_rows']}",
        f"profile={NAME} volumes_with_window={w['volumes_with_window']} ranked_candidates={w['ranked_candidates']} "
        f"excluded_disks={len(r['excluded'])} scaling_factor=1 target_p95_mbps={bp.fmt(w['target_mbps'])} "
        f"p95_mbps={bp.fmt(w['p95_mbps'])} mean_mbps={bp.fmt(w['mean_mbps'])} max_mbps={bp.fmt(w['max_mbps'])}",
        f"profile={NAME} reference={REFERENCE_NAME} reference_start_ts={r['reference_start']} "
        f"shared_sample_times=0 overlap_with_reference_disks={s['overlap_with_reference_disks']} "
        f"without_exclusion_n_disks={s['selection_without_exclusion']['n_disks']} "
        f"without_exclusion_overlap_with_reference_disks={s['selection_without_exclusion']['overlap_with_reference_disks']}",
    ]
    if "csv" in r:
        lines += [
            f"output=profiles/{r['csv'].name} rows={w['rows']} sha256={r['csv_sha256']}",
            f"output=profiles/{r['meta'].name} sha256={r['meta_sha256']}",
        ]
    return lines


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        f"{SECTION_TITLE}\n\n"
        f"- Additive profile `{NAME}`; `{REFERENCE_NAME}` and its splits are unchanged. Same construction as "
        f"`{REFERENCE_NAME}`: the diurnal rule (highest `daily_strength`, the median-bytes-per-I/O eligibility "
        f"rule, N the largest whose aggregate p95 is at most the diurnal target x {1 + bp.CEILING_TOLERANCE:g}) "
        f"over a {bp.LONG_WINDOW_DAYS}-day window ({bp.LONG_WINDOW_ROWS} samples), six-column replay schema, "
        "`t_seconds` from 0, `validate_replay_frame` before writing, no scaling factor\n"
        f"- Window: the start covered by a gap-free run of that length in the most volumes among starts whose "
        f"window shares no sample time with the `{REFERENCE_NAME}` window\n"
        f"- Volumes: every `disk_uid` in `profile_{REFERENCE_NAME}.meta.json` is removed from the candidate pool "
        "before ranking. The record below also gives the selection without that removal\n"
        f"- Files: `profiles/profile_{NAME}.csv` and `profiles/profile_{NAME}.meta.json` (volume list, window, "
        "counts, aggregate p95 and mean, excluded volumes, overlap figures). Produced by "
        "`tencent/build_profile_altB.py`\n"
        "- Counts and measurements from the last run of `tencent/build_profile_altB.py`:\n\n"
        f"{body}\n\n"
    )
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    pattern = re.escape(SECTION_TITLE) + r"\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        new = re.sub(pattern, lambda _m: section.rstrip("\n"), text, count=1, flags=re.S)
    else:
        anchor = "\n## A2 — Horizon labels and drive split"
        if anchor not in text:
            raise SystemExit("A2 section not found in PREPROCESSING.md")
        new = text.replace(anchor, "\n" + section.rstrip("\n") + "\n" + anchor, 1)
    bp.PREPROCESSING.write_text(new, encoding="utf-8", newline="\n")


#: Fresh populations: ``rank`` is ``"diurnal"`` (highest ``daily_strength``) or ``"random"`` (sampled order).
POPULATIONS: dict[str, dict] = {
    "C": {"name": "diurnal_long_popC", "rank": "diurnal",
          "avoid": ["diurnal_long", "diurnal_long_altB"]},
    "D": {"name": "diurnal_long_popD", "rank": "random", "seed": 20261119,
          "avoid": ["diurnal_long", "diurnal_long_altB", "diurnal_long_popC"]},
}
B5C_TITLE = "## B5c — Fresh periodic populations (diurnal_long_popC, diurnal_long_popD)"


def _reference_meta(name: str) -> dict:
    return json.loads((bp.PROFILES_DIR / f"profile_{name}.meta.json").read_text(encoding="utf-8"))


def build_population(kind: str, write: bool = True) -> dict:
    """Build population ``C`` or ``D`` (``POPULATIONS``): a window disjoint in time and volumes from the profiles it avoids.

    The window is the best-covered 14-day window (``time_disjoint_best``). ``C`` ranks the remaining volumes with the
    diurnal rule of ``build_profiles.py``. ``D`` puts the eligible volumes (a gap-free run over the window and not
    ``implausible_io``) in the order of ``numpy.random.default_rng(seed).permutation`` of their sorted ``disk_uid``
    list, and takes the largest prefix whose aggregate p95 is at most the diurnal target x ``1 + CEILING_TOLERANCE``
    (``build_profiles.tune_n``, mode ``not_above``). No volume is skipped or swapped.
    """
    spec = POPULATIONS[kind]
    refs = [_reference_meta(n) for n in spec["avoid"]]
    excluded = sorted({u for m in refs for u in m["disk_uids"]})
    runs = pd.read_parquet(bp.IN_RUNS)
    start, cand_windows = time_disjoint_best(runs, bp.LONG_WINDOW_ROWS, [int(m["window_start_ts"]) for m in refs])
    uids = bp.eligible_disks(runs, start, bp.LONG_WINDOW_ROWS)
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    windows, periodicity = bp.collect(pf, groups, uids, start, bp.LONG_WINDOW_ROWS)
    cand = bp.candidate_frame(pd.read_parquet(bp.IN_SUMMARY), periodicity)
    position = {u: i for i, u in enumerate(uids)}
    diurnal = bp.PROFILE_SPECS["diurnal"]
    pool = cand[~cand["disk_uid"].isin(excluded)]
    if spec["rank"] == "diurnal":
        p = bp.build_profile("diurnal", diurnal, pool, windows, position, start, bp.LONG_WINDOW_ROWS)
        n_eligible = int(p["candidates"])
        rule = diurnal["rule"]
        label = (f"aggregate of {p['n']} {diurnal['label']} volumes over a {bp.LONG_WINDOW_DAYS}-day window "
                 f"disjoint in time and in volumes from {', '.join(spec['avoid'])} ({bp.DISCLOSURE})")
    else:
        eligible = sorted(pool.loc[~pool["implausible_io"], "disk_uid"])
        order = [eligible[i] for i in np.random.default_rng(int(spec["seed"])).permutation(len(eligible))]
        idx = [position[u] for u in order]
        totals = windows[idx][:, :, bp.READ_BW] + windows[idx][:, :, bp.WRITE_BW]
        curve = np.percentile(np.cumsum(totals, axis=0), 95, axis=1) / bp.KBPS_PER_MBPS
        n = bp.tune_n(curve, diurnal["target_mbps"], "not_above")
        agg = pd.DataFrame(windows[idx[:n]].sum(axis=0), columns=bp.VALUE_COLUMNS)
        agg.insert(0, "timestamp", start + np.arange(bp.LONG_WINDOW_ROWS, dtype="int64") * bp.STEP_S)
        replay = bp.contracts.to_replay_frame(agg)
        bp.contracts.validate_replay_frame(replay)
        total = (replay["read_bw_kbps"] + replay["write_bw_kbps"]).to_numpy(dtype="float64")
        p = {"n": n, "frame": replay, "uids": order[:n], "candidates": len(eligible),
             "target_mbps": diurnal["target_mbps"], **bp.total_bw_stats(total)}
        n_eligible = len(eligible)
        rule = (f"uniform random order of the eligible volumes (numpy default_rng({spec['seed']}), a permutation of the "
                "sorted disk_uid list); no ranking by daily-cycle strength")
        label = (f"aggregate of {p['n']} volumes drawn in random order over a {bp.LONG_WINDOW_DAYS}-day window "
                 f"disjoint in time and in volumes from {', '.join(spec['avoid'])} ({bp.DISCLOSURE})")
    del windows
    rows = len(p["frame"])
    split = bp.split_row(rows)
    w = {
        "name": spec["name"], "uids": p["uids"], "n": p["n"], "frame": p["frame"], "start_ts": start,
        "end_ts": start + (rows - 1) * bp.STEP_S, "rows": rows, "requested_rows": bp.LONG_WINDOW_ROWS,
        "volumes_with_window": len(uids), "ranked_candidates": n_eligible,
        "p95_mbps": p["p95_mbps"], "mean_mbps": p["mean_mbps"], "max_mbps": p["max_mbps"],
        "target_mbps": p["target_mbps"], "split_row": split, "split_t_seconds": split * bp.STEP_S,
        "split_ts": start + split * bp.STEP_S,
    }
    sidecar = bp.long_sidecar(w)
    sidecar["label"] = label
    sidecar["selection_rule"] = rule
    sidecar["avoided_profiles"] = list(spec["avoid"])
    sidecar["avoided_window_start_ts"] = [int(m["window_start_ts"]) for m in refs]
    sidecar["excluded_disk_uids"] = excluded
    sidecar["overlap_with_excluded_disks"] = len(set(p["uids"]) & set(excluded))
    sidecar["rank"] = spec["rank"]
    if "seed" in spec:
        sidecar["rng_seed"] = int(spec["seed"])
    sidecar["window_choice"] = (
        "the start covered by a gap-free run of the window length in the most volumes (earliest on ties) among starts "
        "whose window shares no sample time with the avoided windows")
    avoided_spans = [(int(m["window_start_ts"]), int(m["window_end_ts"])) for m in refs]
    if any(not (w["end_ts"] < a or w["start_ts"] > b) for a, b in avoided_spans):
        raise ValueError("the window overlaps an avoided window in time")
    if sidecar["overlap_with_excluded_disks"]:
        raise ValueError("the volume set overlaps an excluded volume")
    out = {"w": w, "sidecar": sidecar, "kind": kind}
    if write:
        csv = bp.PROFILES_DIR / f"profile_{spec['name']}.csv"
        meta = bp.PROFILES_DIR / f"profile_{spec['name']}.meta.json"
        w["frame"].to_csv(csv, index=False, lineterminator="\n")
        bp.write_json(meta, sidecar)
        out["csv"], out["meta"] = csv, meta
    return out


def population_lines(name: str) -> list[str]:
    """Counts and hashes of a written population, read from its files."""
    meta = _reference_meta(name)
    csv = bp.PROFILES_DIR / f"profile_{name}.csv"
    return [
        f"profile={name} n_disks={meta['n_disks']} rank={meta['rank']} start_ts={meta['window_start_ts']} "
        f"start_utc={meta['window_start_utc']} end_ts={meta['window_end_ts']} end_utc={meta['window_end_utc']} "
        f"rows={meta['rows']} requested_rows={meta['requested_rows']}"
        + (f" rng_seed={meta['rng_seed']}" if "rng_seed" in meta else ""),
        f"profile={name} volumes_with_window={meta['volumes_with_window']} eligible_pool={meta['ranked_candidates']} "
        f"excluded_disks={len(meta['excluded_disk_uids'])} overlap_with_excluded_disks={meta['overlap_with_excluded_disks']} "
        f"scaling_factor=1 target_p95_mbps={bp.fmt(meta['target_p95_mbps'])} p95_mbps={bp.fmt(meta['p95_mbps'])} "
        f"mean_mbps={bp.fmt(meta['mean_mbps'])} max_mbps={bp.fmt(meta['max_mbps'])}",
        f"output=profiles/{csv.name} rows={meta['rows']} sha256={bp.sha256_of(csv)}",
        f"output=profiles/{csv.with_suffix('.meta.json').name} sha256={bp.sha256_of(csv.with_suffix('.meta.json'))}",
    ]


def update_preprocessing_c() -> None:
    """Write section B5c from the sidecars of the populations that exist."""
    present = [POPULATIONS[k]["name"] for k in ("C", "D") if (bp.PROFILES_DIR / f"profile_{POPULATIONS[k]['name']}.csv").exists()]
    body = "\n".join(f"- `{ln}`" for name in present for ln in population_lines(name))
    section = (
        f"{B5C_TITLE}\n\n"
        "- Two further 14-day profiles for a pre-registered evaluation; `diurnal_long`, `diurnal_long_altB` and their splits are "
        "unchanged. Each is the sum of N real volumes over one shared 14-day window (4032 samples, 300 s), six-column "
        "replay schema, `t_seconds` from 0, `validate_replay_frame` before writing, no scaling factor, N the largest whose "
        f"aggregate p95 is at most the diurnal target x {1 + bp.CEILING_TOLERANCE:g}\n"
        "- Window: the best-covered 14-day window (most volumes with a gap-free run, earliest start on ties) that shares no "
        "sample time with the earlier profiles' windows. Volumes: those of the earlier profiles are removed from the pool\n"
        "- `diurnal_long_popC`: the volumes are ranked with the diurnal rule (highest `daily_strength`) of `build_profiles.py`\n"
        "- `diurnal_long_popD`: the eligible volumes (gap-free over the window, not `implausible_io`) are taken in the order "
        "of `numpy.random.default_rng(20261119).permutation` of their sorted `disk_uid` list; N is the largest prefix whose "
        "aggregate p95 is within the target. No ranking by daily-cycle strength, no volume skipped or swapped\n"
        "- Files: `profiles/profile_<name>.csv` and `.meta.json` (volume list, window, seed, counts, aggregate p95 and mean, "
        "excluded volumes), listed in `manifest.json`. Produced by `tencent/build_profile_altB.py --population C|D`\n"
        "- Counts and measurements from the last run:\n\n"
        f"{body}\n\n"
    )
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    pattern = re.escape(B5C_TITLE) + r"\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        new = re.sub(pattern, lambda _m: section.rstrip("\n"), text, count=1, flags=re.S)
    else:
        anchor = "\n## A2 — Horizon labels and drive split"
        if anchor not in text:
            raise SystemExit("A2 section not found in PREPROCESSING.md")
        new = text.replace(anchor, "\n" + section.rstrip("\n") + "\n" + anchor, 1)
    bp.PREPROCESSING.write_text(new, encoding="utf-8", newline="\n")


def update_manifest_files(names: Sequence[str]) -> None:
    """Upsert the CSV and sidecar of each profile in ``manifest.json`` (``files`` entries: file, rows, sha256)."""
    data = json.loads(bp.MANIFEST.read_text(encoding="utf-8"))
    entries = []
    for name in names:
        csv = bp.PROFILES_DIR / f"profile_{name}.csv"
        meta = csv.with_suffix(".meta.json")
        rows = int(len(pd.read_csv(csv)))
        entries.append({"file": csv.relative_to(bp.ROOT).as_posix(), "rows": rows, "sha256": bp.sha256_of(csv)})
        entries.append({"file": meta.relative_to(bp.ROOT).as_posix(), "rows": None, "sha256": bp.sha256_of(meta)})
    listed = {e["file"] for e in entries}
    data["files"] = [e for e in data.get("files", []) if e.get("file") not in listed] + entries
    bp.MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m data_pipeline.tencent.build_profile_altB",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--population", choices=("altB", "C", "D"), default="altB", help="profile to build")
    p.add_argument("--dry-run", action="store_true", help="build and validate without writing")
    args = p.parse_args(argv)
    if args.population != "altB":
        r = build_population(args.population, write=not args.dry_run)
        w = r["w"]
        print(f"profile={w['name']} n_disks={w['n']} start_utc={bp.utc(w['start_ts'])} end_utc={bp.utc(w['end_ts'])} "
              f"rows={w['rows']} p95_mbps={bp.fmt(w['p95_mbps'])} mean_mbps={bp.fmt(w['mean_mbps'])} "
              f"eligible_pool={w['ranked_candidates']}")
        if not args.dry_run:
            update_preprocessing_c()
            update_manifest_files([w["name"]])
        return 0
    r = build(write=not args.dry_run)
    lines = report_lines(r)
    print("\n".join(lines))
    if not args.dry_run:
        update_preprocessing(lines)
        update_manifest_files([NAME])
    return 0


if __name__ == "__main__":
    sys.exit(main())
