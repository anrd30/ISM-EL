"""
Tests for check_outputs.py on a small synthetic output tree.

The tree holds tiny Parquet tables for every expected processed output, copies of
the committed profile CSVs and sidecars, a small PNG, a manifest with correct
hashes and row counts, and a PREPROCESSING.md with matching `output=` lines. The
module's ROOT, MANIFEST and PREPROCESSING are redirected into tmp_path.
"""
import base64
import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import contracts
from data_pipeline import check_outputs as co
from data_pipeline.backblaze import extract as bb_extract
from data_pipeline.backblaze import labels as bb_labels
from data_pipeline.backblaze import timelines as bb_timelines

REAL_PROFILES = Path(co.__file__).resolve().parent / "profiles"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
DAY1, DAY2 = datetime.date(2022, 1, 1), datetime.date(2022, 1, 2)
KEYLESS_SUFFIXES = (".png", ".json")
SEAGATE, TOSHIBA = "SEA", "TOS"
FIRST_DAY = datetime.date(2022, 1, 1)
BB_DAYS = 45


def backblaze_world():
    """A small consistent Backblaze dataset: extract, timelines, drive index and label lines.

    S1 (Seagate, 45 days, no failure), S2 (Seagate, fails on day 40), T1 (Toshiba, 20 days,
    smart_187_raw and smart_188_raw never reported). Eight 3-day drives per model fail on day 3, so
    every split of the A2 drive split holds a failed drive of each model."""
    rows = []
    drives = [("S1", SEAGATE, 45, None), ("S2", SEAGATE, 40, 40), ("T1", TOSHIBA, 20, None)]
    drives += [(f"{model}F{k}", model, 3, 3) for model in (SEAGATE, TOSHIBA) for k in range(8)]
    for serial, model, last_day, fail_day in drives:
        for d in range(1, last_day + 1):
            row = {"serial_number": serial, "model": model,
                   "date": FIRST_DAY + datetime.timedelta(days=d - 1),
                   "capacity_bytes": 4_000_000_000_000, "failure": int(d == fail_day)}
            for k, column in enumerate(bb_timelines.SMART_COLUMNS):
                row[column] = float(d + k)
            if model == TOSHIBA:
                row["smart_187_raw"] = row["smart_188_raw"] = np.nan
            rows.append(row)
    filtered = pd.DataFrame(rows)
    timeline = bb_timelines.build_timelines(filtered)
    index = bb_timelines.build_drive_index(filtered[bb_timelines.CORE_IN])
    labels = bb_timelines.row_labels(timeline, index)
    totals = {m: bb_timelines.label_counts(labels[(timeline["model"] == m).to_numpy()])
              for m in (SEAGATE, TOSHIBA)}
    lines = bb_timelines.label_lines(totals, index, bb_timelines.drive_label_flags(timeline, labels))
    expected = {m: (int(filtered.loc[filtered["model"] == m, "failure"].sum()),
                    int((filtered["model"] == m).sum())) for m in (SEAGATE, TOSHIBA)}
    split = bb_labels.assign_splits(index)
    label_frame = pd.DataFrame({
        "serial_number": timeline["serial_number"].astype(object), "date": timeline["date"],
        "model": timeline["model"].astype(object), "label": labels.astype("int8")})
    label_frame["date"] = label_frame["date"].dt.date
    for frame in (timeline, index):
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].dt.date
    return filtered, timeline, index, lines, expected, label_frame, split


def disk_index_frame(uids=("d1", "d2")):
    n = len(uids)
    return pd.DataFrame({
        "disk_uid": list(uids),
        "disk_capacity": np.full(n, 50, dtype="int64"), "disk_attr": np.ones(n, dtype="int64"),
        "disk_type": np.ones(n, dtype="int64"), "user_type": np.zeros(n, dtype="int64"),
        "vm_cpu": np.full(n, 2, dtype="int64"), "vm_memory": np.full(n, 4, dtype="int64"),
    }, columns=list(contracts.DISK_INDEX_COLUMNS))


class Tree:
    """A synthetic output tree plus the bookkeeping that keeps its manifest and PREPROCESSING.md right."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: dict[str, dict] = {}
        self.label_lines: list[str] = []
        self.a2_lines: list[str] = []
        self.sources: dict[str, str] = {}

    def path(self, rel: str) -> Path:
        return self.root / rel

    def record(self, rel: str) -> None:
        path = self.path(rel)
        rows = None if path.suffix in KEYLESS_SUFFIXES else co.actual_rows(path)
        self.entries[rel] = {"file": rel, "rows": rows, "sha256": co.sha256_of(path)}

    def save(self) -> None:
        profiles = {
            n: {"n_disks": json.loads(self.path(co.PROFILE_SIDECAR[n]).read_text(encoding="utf-8"))["n_disks"]}
            for n in co.PROFILE_NAMES
        }
        files = [self.entries[r] for r in co.ALL_OUTPUTS if r in self.entries]
        manifest = {"files": files, "sources": {"backblaze": self.sources}, "profiles": profiles}
        self.path("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        filtered = self.entries[co.BACKBLAZE_FILTERED]
        lines = []
        for entry in files:
            if entry["file"] == co.BACKBLAZE_FILTERED:
                lines.append(f"- `output={entry['file']} sha256={entry['sha256']}`")
            elif entry["rows"] is None:
                lines.append(f"- `output={entry['file']} sha256={entry['sha256']}`")
            else:
                lines.append(f"- `output={entry['file']} rows={entry['rows']} sha256={entry['sha256']}`")
        text = (
            "# P\n\n## B1 — Backblaze extract\n\n"
            f"- `rows_out={filtered['rows']}`\n\n## B2 — Per-drive timelines\n\n"
            + "\n".join(f"- `{line}`" for line in self.label_lines) + "\n"
            + "\n".join(lines) + "\n\n## A2 — Horizon labels and drive split\n\n"
            + "\n".join(f"- `{line}`" for line in self.a2_lines) + "\n\n## B6 — Acceptance\n"
        )
        self.path("PREPROCESSING.md").write_text(text, encoding="utf-8", newline="\n")

    def rewrite_preprocessing(self, old: str, new: str) -> None:
        path = self.path("PREPROCESSING.md")
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, old
        path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    (root / "tests").mkdir(parents=True)
    (root / co.LEAKAGE_TEST).write_text("# leakage test\n", encoding="utf-8")
    filtered, timeline, index, label_lines, expected, label_frame, split_frame = backblaze_world()
    monkeypatch.setattr(bb_extract, "MODELS", (SEAGATE, TOSHIBA))
    monkeypatch.setattr(bb_extract, "SEAGATE_MODELS", (SEAGATE,))
    monkeypatch.setattr(bb_extract, "EXPECTED_PER_MODEL", expected)
    monkeypatch.setattr(bb_extract, "EXPECTED_POOLED",
                        (sum(f for f, _ in expected.values()), sum(d for _, d in expected.values())))
    monkeypatch.setattr(bb_extract, "EXPECTED_FILES", BB_DAYS)
    monkeypatch.setattr(bb_extract, "EXPECTED_DATE_RANGE", (
        FIRST_DAY.isoformat(), (FIRST_DAY + datetime.timedelta(days=BB_DAYS - 1)).isoformat()))
    monkeypatch.setattr(bb_extract, "EXCLUDED_FILES", {})
    (root / "profiles").mkdir(parents=True)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    for name in co.PROFILE_NAMES:
        shutil.copy(REAL_PROFILES / f"profile_{name}.csv", root / "profiles")
        shutil.copy(REAL_PROFILES / f"profile_{name}.meta.json", root / "profiles")
        pd.read_csv(root / co.PROFILE_CSV[name]).to_parquet(root / co.PROFILE_PARQUET[name], index=False)
    (root / co.PROFILE_PLOT).write_bytes(PNG_BYTES)

    tables = {
        co.BACKBLAZE_FILTERED: filtered,
        co.BACKBLAZE_TIMELINES: timeline,
        co.BACKBLAZE_DRIVE_INDEX: index,
        co.BACKBLAZE_LABELS: label_frame,
        co.BACKBLAZE_SPLIT: split_frame,
        co.TENCENT_CLEAN: pd.DataFrame({"disk_uid": ["d1", "d1", "d2"], "timestamp": [1, 2, 3]}),
        co.TENCENT_SUMMARY: pd.DataFrame({"disk_uid": ["d1", "d2"], "rows_clean": [2, 1]}),
        co.TENCENT_RUNS: pd.DataFrame({"disk_uid": ["d1", "d2"], "run_start_ts": [1, 3], "run_rows": [2, 1]}),
        co.TENCENT_DISK_INDEX: disk_index_frame(),
        co.PROFILE_CANDIDATES: pd.DataFrame({"disk_uid": ["d1"], "days": [1.0]}),
    }
    for rel, frame in tables.items():
        frame.to_parquet(root / rel, index=False)
    disks = [
        {"profile": name, "rank": i + 1, "disk_uid": f"{name}{i}", "key_value": float(i)}
        for name in co.PROFILE_NAMES
        for i in range(json.loads((root / co.PROFILE_SIDECAR[name]).read_text(encoding="utf-8"))["n_disks"])
    ]
    pd.DataFrame(disks).to_parquet(root / co.PROFILE_DISKS, index=False)

    built = Tree(root)
    built.label_lines = label_lines
    label_counts = bb_timelines.label_counts(label_frame["label"].to_numpy())
    built.a2_lines = bb_labels.report_lines(
        {"rows": len(label_frame), "totals": label_counts},
        bb_labels.split_statistics(root / co.BACKBLAZE_LABELS, split_frame))
    built.sources = {(FIRST_DAY + datetime.timedelta(days=i)).isoformat() + ".csv": "0" * 64
                     for i in range(BB_DAYS)}
    for rel in co.ALL_OUTPUTS:
        built.record(rel)
    built.save()
    monkeypatch.setattr(co, "ROOT", root)
    monkeypatch.setattr(co, "MANIFEST", root / "manifest.json")
    monkeypatch.setattr(co, "PREPROCESSING", root / "PREPROCESSING.md")
    return built


def run(capsys, *args):
    code = co.main(list(args))
    return code, capsys.readouterr().out


def test_valid_tree_passes(tree, capsys):
    code, out = run(capsys)
    assert code == 0
    assert out.rstrip().endswith("RESULT: PASS")
    assert "[FAIL]" not in out and "[MISSING]" not in out and "[SKIP]" not in out
    assert "[PASS] sidecar profiles/profile_idle.meta.json" in out
    assert "[PASS] latency ladder profiles/profile_peak.csv: safe_actions=1 expected=1" in out
    assert "[PASS] validate_disk_index_frame data/processed/tencent_disk_index.parquet" in out
    assert "[PASS] manifest lists every output file" in out


def test_one_changed_byte_is_a_sha256_failure(tree, capsys):
    png = tree.path(co.PROFILE_PLOT)
    data = bytearray(png.read_bytes())
    data[-1] ^= 1
    png.write_bytes(bytes(data))
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] sha256 {co.PROFILE_PLOT}: sha256 differs from manifest" in out
    assert out.rstrip().endswith("RESULT: FAIL")


def test_a_corrupted_parquet_file_fails_without_crashing(tree, capsys):
    path = tree.path(co.BACKBLAZE_DRIVE_INDEX)
    path.write_bytes(path.read_bytes() + b"x")
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] sha256 {co.BACKBLAZE_DRIVE_INDEX}" in out and "unreadable" in out
    assert f"[FAIL] unique key (serial_number) {co.BACKBLAZE_DRIVE_INDEX}: file unreadable" in out
    assert f"[FAIL] rows vs PREPROCESSING.md {co.BACKBLAZE_DRIVE_INDEX}: file unreadable" in out


def test_a_deleted_file_is_missing_and_the_run_is_incomplete(tree, capsys):
    tree.path(co.TENCENT_RUNS).unlink()
    code, out = run(capsys)
    assert code == 1
    assert f"[MISSING] sha256 {co.TENCENT_RUNS}" in out
    assert f"[MISSING] rows vs PREPROCESSING.md {co.TENCENT_RUNS}" in out
    assert "[FAIL]" not in out
    assert out.rstrip().endswith("RESULT: INCOMPLETE")


def test_a_row_count_disagreeing_with_preprocessing_is_a_failure(tree, capsys):
    recorded = tree.entries[co.BACKBLAZE_TIMELINES]
    tree.rewrite_preprocessing(f"output={co.BACKBLAZE_TIMELINES} rows={recorded['rows']} ",
                               f"output={co.BACKBLAZE_TIMELINES} rows={recorded['rows'] + 1} ")
    code, out = run(capsys)
    assert code == 1
    assert (f"[FAIL] rows vs PREPROCESSING.md {co.BACKBLAZE_TIMELINES}: "
            f"file={recorded['rows']} recorded={recorded['rows'] + 1}") in out
    assert f"[PASS] sha256 {co.BACKBLAZE_TIMELINES}" in out


def test_the_b1_row_count_comes_from_rows_out(tree, capsys):
    rows = tree.entries[co.BACKBLAZE_FILTERED]["rows"]
    tree.rewrite_preprocessing(f"rows_out={rows}", f"rows_out={rows + 5}")
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] rows vs PREPROCESSING.md {co.BACKBLAZE_FILTERED}: file={rows} recorded={rows + 5}" in out


def test_an_output_without_a_recorded_row_count_is_a_failure(tree, capsys):
    rows = tree.entries[co.TENCENT_SUMMARY]["rows"]
    tree.rewrite_preprocessing(f"output={co.TENCENT_SUMMARY} rows={rows} ", f"output={co.TENCENT_SUMMARY} ")
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] rows vs PREPROCESSING.md {co.TENCENT_SUMMARY}: no row count recorded" in out


def test_a_disk_index_with_a_duplicate_disk_uid_fails_the_contract_validator(tree, capsys):
    disk_index_frame(("d1", "d1", "d2")).to_parquet(tree.path(co.TENCENT_DISK_INDEX), index=False)
    tree.record(co.TENCENT_DISK_INDEX)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] validate_disk_index_frame {co.TENCENT_DISK_INDEX}: " in out and "duplicate" in out
    assert f"[PASS] sha256 {co.TENCENT_DISK_INDEX}" in out


def test_a_profile_with_288_rows_and_corrupted_timestamps_fails(tree, capsys):
    csv = tree.path(co.PROFILE_CSV["idle"])
    frame = pd.read_csv(csv)
    frame.loc[100, "t_seconds"] += 7
    frame.to_csv(csv, index=False, lineterminator="\n")
    tree.record(co.PROFILE_CSV["idle"])
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] replay contract {co.PROFILE_CSV['idle']}: step at index 100 is 307 s" in out
    assert f"[PASS] rows vs PREPROCESSING.md {co.PROFILE_CSV['idle']}: file=288 recorded=288" in out
    assert "[FAIL] latency ladder profiles/profile_idle.csv: ladder not computed" in out
    assert "[FAIL] CSV equals Parquet profile_idle" in out


def test_a_profile_with_the_wrong_columns_or_length_fails(tree, capsys):
    csv = tree.path(co.PROFILE_CSV["peak"])
    frame = pd.read_csv(csv)
    frame[list(reversed(frame.columns))].to_csv(csv, index=False, lineterminator="\n")
    short = tree.path(co.PROFILE_CSV["bursty"])
    pd.read_csv(short).iloc[:100].to_csv(short, index=False, lineterminator="\n")
    for rel in (co.PROFILE_CSV["peak"], co.PROFILE_CSV["bursty"]):
        tree.record(rel)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] replay contract {co.PROFILE_CSV['peak']}: columns=" in out
    assert f"[FAIL] replay contract {co.PROFILE_CSV['bursty']}: rows=100 expected=288" in out


def test_a_sidecar_marked_synthetic_fails(tree, capsys):
    sidecar = tree.path(co.PROFILE_SIDECAR["bursty"])
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["synthetic"] = True
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tree.record(co.PROFILE_SIDECAR["bursty"])
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] sidecar {co.PROFILE_SIDECAR['bursty']}: synthetic=True, expected false" in out


def test_a_sidecar_with_the_wrong_profile_name_or_disk_count_fails(tree, capsys):
    sidecar = tree.path(co.PROFILE_SIDECAR["idle"])
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.update({"profile": "peak", "n_disks": payload["n_disks"] + 1})
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    tree.record(co.PROFILE_SIDECAR["idle"])
    tree.save()
    manifest = json.loads(tree.path("manifest.json").read_text(encoding="utf-8"))
    manifest["profiles"]["idle"]["n_disks"] -= 1
    tree.path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, out = run(capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith(f"[FAIL] sidecar {co.PROFILE_SIDECAR['idle']}"))
    assert "profile='peak', expected 'idle'" in line and "n_disks=" in line


def test_an_unparseable_sidecar_fails_without_crashing(tree, capsys):
    tree.path(co.PROFILE_SIDECAR["diurnal"]).write_text("{not json", encoding="utf-8")
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] sidecar {co.PROFILE_SIDECAR['diurnal']}: not parseable" in out


def test_a_missing_sidecar_is_missing(tree, capsys):
    tree.path(co.PROFILE_SIDECAR["peak"]).unlink()
    code, out = run(capsys)
    assert code == 1
    assert f"[MISSING] sidecar {co.PROFILE_SIDECAR['peak']}" in out


def test_the_ladder_expectation_is_enforced(tree, capsys, monkeypatch):
    monkeypatch.setattr(co, "EXPECTED_SAFE_ACTIONS", {**co.EXPECTED_SAFE_ACTIONS, "idle": 3})
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] latency ladder profiles/profile_idle.csv: safe_actions=4 expected=3" in out


def test_a_missing_manifest_entry_is_reported(tree, capsys):
    manifest = json.loads(tree.path("manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [e for e in manifest["files"] if e["file"] != co.PROFILE_PLOT]
    tree.path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] manifest lists every output file" in out and co.PROFILE_PLOT in out
    assert f"[FAIL] referenced {co.PROFILE_PLOT}: not listed in manifest.json" in out


def test_profile_disk_counts_must_agree_with_the_manifest(tree, capsys):
    disks = pd.read_parquet(tree.path(co.PROFILE_DISKS))
    disks.iloc[1:].to_parquet(tree.path(co.PROFILE_DISKS), index=False)
    tree.record(co.PROFILE_DISKS)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] manifest profile disk counts vs tencent_profile_disks.parquet" in out


def test_committed_only_without_data_processed_passes_with_skip_lines(tree, capsys):
    shutil.rmtree(tree.root / "data")
    code, out = run(capsys, "--committed-only")
    assert code == 0
    assert out.rstrip().endswith("RESULT: PASS")
    assert "[MISSING]" not in out and "[FAIL]" not in out
    processed = [r for r in co.ALL_OUTPUTS if r.startswith("data/processed/")]
    for rel in processed:
        assert f"[SKIP] sha256 {rel}: not tracked in git" in out
    assert "[SKIP] unique key (serial_number, date) data/processed/backblaze_timelines.parquet: " \
           "not tracked in git" in out
    assert "[SKIP] validate_disk_index_frame data/processed/tencent_disk_index.parquet: not tracked in git" in out
    assert "[SKIP] manifest profile disk counts vs tencent_profile_disks.parquet: not tracked in git" in out
    for rel in list(co.PROFILE_CSV.values()) + list(co.PROFILE_SIDECAR.values()) + [co.PROFILE_PLOT]:
        assert f"[PASS] sha256 {rel}" in out
    assert "[PASS] replay contract profiles/profile_idle.csv" in out
    assert "[PASS] latency ladder profiles/profile_idle.csv" in out
    assert not any(line.startswith("[SKIP]") and "not tracked in git" not in line for line in out.splitlines())


def test_the_default_mode_reports_missing_processed_files_as_incomplete(tree, capsys):
    shutil.rmtree(tree.root / "data")
    code, out = run(capsys)
    assert code == 1
    assert "[SKIP]" not in out and "[MISSING] sha256 data/processed/tencent_load_clean.parquet" in out
    assert out.rstrip().endswith("RESULT: INCOMPLETE")


def test_committed_only_still_fails_on_a_changed_committed_file(tree, capsys):
    csv = tree.path(co.PROFILE_CSV["idle"])
    csv.write_bytes(csv.read_bytes() + b"\n")
    code, out = run(capsys, "--committed-only")
    assert code == 1
    assert f"[FAIL] sha256 {co.PROFILE_CSV['idle']}" in out and out.rstrip().endswith("RESULT: FAIL")


def test_a_fail_outranks_a_missing_in_the_final_line(tree, capsys):
    tree.path(co.TENCENT_RUNS).unlink()
    png = tree.path(co.PROFILE_PLOT)
    png.write_bytes(png.read_bytes() + b"x")
    code, out = run(capsys)
    assert code == 1 and out.rstrip().endswith("RESULT: FAIL")


def test_null_key_values_are_not_reported_as_duplicates():
    """Rows (None, d1), (a, d2), (b, None) are distinct; a factorize-based encoding gave -1 codes
    for the nulls and collided (a, d2) with (b, None)."""
    frame = pd.DataFrame({"serial_number": [None, "a", "b"], "date": [DAY1, DAY2, None]})
    assert co.duplicate_key_rows(frame, ["serial_number", "date"]) == 0
    assert co.duplicate_key_rows(pd.concat([frame, frame.iloc[:1]]), ["serial_number", "date"]) == 1


def test_a_table_with_null_key_values_passes_the_key_check(tree, capsys):
    frame = pd.DataFrame({"serial_number": [None, "a", "b"], "date": [DAY1, DAY2, None], "failure": [0, 0, 1]})
    frame.to_parquet(tree.path(co.BACKBLAZE_FILTERED), index=False)
    tree.record(co.BACKBLAZE_FILTERED)
    tree.save()
    code, out = run(capsys)
    assert f"[PASS] unique key (serial_number, date) {co.BACKBLAZE_FILTERED}: duplicate_rows=0" in out
    assert f"[FAIL] backblaze model set {co.BACKBLAZE_FILTERED}" in out


def test_a_genuine_duplicate_key_is_a_failure(tree, capsys):
    frame = pd.DataFrame({"serial_number": ["a", "a", "b"], "date": [DAY1, DAY1, DAY1], "failure": [0, 0, 1]})
    frame.to_parquet(tree.path(co.BACKBLAZE_TIMELINES), index=False)
    tree.record(co.BACKBLAZE_TIMELINES)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] unique key (serial_number, date) {co.BACKBLAZE_TIMELINES}: duplicate_rows=1" in out
    assert "[FAIL] backblaze label counts vs PREPROCESSING.md" in out


def test_an_unknown_argument_is_a_usage_error(capsys):
    assert co.main(["--bogus"]) == 2
    assert "--committed-only" in capsys.readouterr().out


def test_a_missing_manifest_and_preprocessing_are_reported_without_crashing(tree, capsys):
    tree.path("manifest.json").unlink()
    tree.path("PREPROCESSING.md").unlink()
    code, out = run(capsys)
    assert code == 1
    assert "[MISSING] manifest.json exists" in out and "[MISSING] PREPROCESSING.md exists" in out


def test_recorded_rows_reads_output_lines_and_the_b1_rows_out_line():
    text = ("## B1 — Backblaze extract\n- `rows_out=7`\n\n## B2 — x\n"
            "- `output=data/processed/a.parquet rows=3 sha256=ab`\n- `output=p.png sha256=cd`\n")
    assert co.recorded_rows(text) == {"data/processed/a.parquet": 3, co.BACKBLAZE_FILTERED: 7}
    assert co.recorded_rows("no sections") == {}


def test_actual_rows_and_safe_rows(tmp_path):
    parquet = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(parquet, index=False)
    csv = tmp_path / "t.csv"
    csv.write_text("a\n1\n2\n")
    png = tmp_path / "t.png"
    png.write_bytes(PNG_BYTES)
    assert co.actual_rows(parquet) == 3 and co.actual_rows(csv) == 2 and co.actual_rows(png) is None
    parquet.write_bytes(b"not parquet")
    assert co.safe_rows(parquet) is None


def test_the_replay_columns_come_from_the_contract():
    assert co.REPLAY_COLUMNS is contracts.REPLAY_COLUMNS
    assert co.PROFILE_ROWS == 288


# -- Backblaze checks --

def test_the_valid_tree_passes_every_backblaze_check(tree, capsys):
    code, out = run(capsys)
    assert code == 0, out
    for label in (f"backblaze model set {co.BACKBLAZE_FILTERED}", f"backblaze counts {SEAGATE}",
                  f"backblaze counts {TOSHIBA}", "backblaze counts pooled",
                  f"backblaze date range {co.BACKBLAZE_FILTERED}",
                  f"backblaze NaN pattern {co.BACKBLAZE_FILTERED}",
                  f"backblaze shared column coverage {co.BACKBLAZE_FILTERED}",
                  "backblaze source files in manifest",
                  "backblaze label counts vs PREPROCESSING.md", f"leakage test present {co.LEAKAGE_TEST}",
                  "leakage cut-and-recompute backblaze_timelines"):
        assert any(line.startswith(f"[PASS] {label}") for line in out.splitlines()), label


def test_a_wrong_model_set_fails(tree, capsys, monkeypatch):
    monkeypatch.setattr(bb_extract, "MODELS", (SEAGATE, TOSHIBA, "EXTRA"))
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze model set {co.BACKBLAZE_FILTERED}: models=['{SEAGATE}', '{TOSHIBA}']" in out


def test_a_wrong_failure_count_fails(tree, capsys, monkeypatch):
    expected = dict(bb_extract.EXPECTED_PER_MODEL)
    failures, days = expected[SEAGATE]
    expected[SEAGATE] = (failures + 1, days)
    monkeypatch.setattr(bb_extract, "EXPECTED_PER_MODEL", expected)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze counts {SEAGATE}: failures={failures} expected={failures + 1}" in out
    assert f"[PASS] backblaze counts {TOSHIBA}" in out


def test_a_wrong_drive_day_count_or_pooled_total_fails(tree, capsys, monkeypatch):
    expected = dict(bb_extract.EXPECTED_PER_MODEL)
    failures, days = expected[TOSHIBA]
    expected[TOSHIBA] = (failures, days + 1)
    pooled_failures, pooled_days = bb_extract.EXPECTED_POOLED
    monkeypatch.setattr(bb_extract, "EXPECTED_PER_MODEL", expected)
    monkeypatch.setattr(bb_extract, "EXPECTED_POOLED", (pooled_failures, pooled_days + 1))
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze counts {TOSHIBA}" in out and "[FAIL] backblaze counts pooled" in out


def test_a_missing_extract_is_missing_not_a_crash(tree, capsys):
    tree.path(co.BACKBLAZE_FILTERED).unlink()
    code, out = run(capsys)
    assert code == 1
    assert f"[MISSING] backblaze model set {co.BACKBLAZE_FILTERED}" in out
    assert "[MISSING] leakage cut-and-recompute backblaze_timelines" in out
    assert f"[MISSING] backblaze shared column coverage {co.BACKBLAZE_FILTERED}" in out
    assert "[FAIL]" not in out and out.rstrip().endswith("RESULT: INCOMPLETE")


def test_the_date_range_and_file_count_are_checked(tree, capsys, monkeypatch):
    monkeypatch.setattr(bb_extract, "EXPECTED_FILES", BB_DAYS + 1)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze date range {co.BACKBLAZE_FILTERED}: first=2022-01-01 last=2022-02-14 distinct_dates=45" in out
    assert "[FAIL] backblaze source files in manifest: files=45 expected=46" in out


def test_the_source_files_must_match_the_calendar_minus_the_excluded_files(tree, capsys, monkeypatch):
    monkeypatch.setattr(bb_extract, "EXCLUDED_FILES", {"q/2022-01-05.csv": "test reason"})
    monkeypatch.setattr(bb_extract, "EXPECTED_FILES", BB_DAYS - 1)
    manifest = json.loads(tree.path("manifest.json").read_text(encoding="utf-8"))
    manifest["sources"]["backblaze"].pop("2022-01-05.csv")
    co.check_backblaze_sources(manifest)
    assert "[PASS] backblaze source files in manifest: files=44" in capsys.readouterr().out
    co.check_backblaze_sources(json.loads(tree.path("manifest.json").read_text(encoding="utf-8")))
    assert "[FAIL] backblaze source files in manifest: files=45 expected=44" in capsys.readouterr().out


def test_the_nan_pattern_is_checked_for_both_vendors(tree, capsys):
    filtered = pd.read_parquet(tree.path(co.BACKBLAZE_FILTERED))
    filtered.loc[filtered["model"] == TOSHIBA, "smart_187_raw"] = 1.0
    filtered.loc[filtered["model"] == SEAGATE, "smart_188_raw"] = np.nan
    filtered.to_parquet(tree.path(co.BACKBLAZE_FILTERED), index=False)
    tree.record(co.BACKBLAZE_FILTERED)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith("[FAIL] backblaze NaN pattern"))
    assert f"{TOSHIBA} smart_187_raw populated share 1.0000, expected 0" in line
    assert f"{SEAGATE} smart_188_raw populated share 0.0000" in line


def test_a_label_count_that_differs_from_preprocessing_fails(tree, capsys):
    old = tree.label_lines[0]
    assert old.startswith("labels_h30 model=all positive=")
    tree.rewrite_preprocessing(old, old.replace("positive=", "positive=1"))
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] backblaze label counts vs PREPROCESSING.md: labels all: recorded=" in out


def test_missing_label_lines_fail(tree, capsys):
    tree.label_lines = []
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] backblaze label counts vs PREPROCESSING.md: no label lines recorded" in out


def test_the_label_counts_include_all_three_classes(tree, capsys):
    code, out = run(capsys)
    line = next(l for l in out.splitlines() if l.startswith("[PASS] backblaze label counts"))
    assert "horizon_days=30" in line and "models=2" in line
    assert "positive=" in line and "negative=" in line and "censored=" in line
    positive = int(line.split("positive=")[1].split()[0])
    negative = int(line.split("negative=")[1].split()[0])
    censored = int(line.split("censored=")[1].split()[0])
    assert positive > 0 and negative > 0 and censored > 0


def test_a_timelines_file_that_differs_from_a_recomputation_fails_the_leakage_check(tree, capsys):
    timeline = pd.read_parquet(tree.path(co.BACKBLAZE_TIMELINES))
    timeline.loc[timeline.index[5], "smart_9_raw_diff7d"] = 12345.0
    timeline.to_parquet(tree.path(co.BACKBLAZE_TIMELINES), index=False)
    tree.record(co.BACKBLAZE_TIMELINES)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] leakage cut-and-recompute backblaze_timelines: stored timelines differ" in out


def test_a_missing_leakage_test_file_fails(tree, capsys):
    tree.path(co.LEAKAGE_TEST).unlink()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] leakage test present {co.LEAKAGE_TEST}" in out


def test_committed_only_skips_the_backblaze_data_checks_but_checks_sources_and_the_test_file(tree, capsys):
    shutil.rmtree(tree.root / "data")
    code, out = run(capsys, "--committed-only")
    assert code == 0, out
    assert f"[SKIP] backblaze model set {co.BACKBLAZE_FILTERED}: not tracked in git" in out
    assert "[SKIP] backblaze label counts vs PREPROCESSING.md: not tracked in git" in out
    assert f"[SKIP] backblaze shared column coverage {co.BACKBLAZE_FILTERED}: not tracked in git" in out
    assert "[SKIP] leakage cut-and-recompute backblaze_timelines: not tracked in git" in out
    assert "[PASS] backblaze source files in manifest" in out
    assert f"[PASS] leakage test present {co.LEAKAGE_TEST}" in out


def test_a_model_with_no_smart_12_values_fails_the_shared_column_coverage(tree, capsys):
    filtered = pd.read_parquet(tree.path(co.BACKBLAZE_FILTERED))
    filtered.loc[filtered["model"] == TOSHIBA, "smart_12_raw"] = np.nan
    filtered.to_parquet(tree.path(co.BACKBLAZE_FILTERED), index=False)
    tree.record(co.BACKBLAZE_FILTERED)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert (f"[FAIL] backblaze shared column coverage {co.BACKBLAZE_FILTERED}: "
            f"{TOSHIBA} smart_12_raw populated share 0.0000") in out
    assert f"[PASS] backblaze NaN pattern {co.BACKBLAZE_FILTERED}" in out


def test_the_valid_fixture_passes_the_shared_column_coverage(tree, capsys):
    code, out = run(capsys)
    assert code == 0, out
    assert (f"[PASS] backblaze shared column coverage {co.BACKBLAZE_FILTERED}: "
            f"smart_12_raw populated share >= 0.99 for every model") in out


def test_a_date_column_with_no_values_fails_the_date_range_without_raising(tree, capsys):
    table = pa.Table.from_pandas(pd.read_parquet(tree.path(co.BACKBLAZE_FILTERED)), preserve_index=False)
    position = table.schema.get_field_index("date")
    table = table.set_column(position, "date", pa.nulls(len(table), pa.date32()))
    pq.write_table(table, tree.path(co.BACKBLAZE_FILTERED))
    tree.record(co.BACKBLAZE_FILTERED)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith("[FAIL] backblaze date range"))
    assert "first=none last=none" in line


def test_an_extract_with_no_rows_fails_the_date_range_without_raising(tree, capsys):
    table = pa.Table.from_pandas(pd.read_parquet(tree.path(co.BACKBLAZE_FILTERED)), preserve_index=False)
    position = table.schema.get_field_index("date")
    table = table.set_column(position, "date", pa.nulls(len(table), pa.date32())).slice(0, 0)
    pq.write_table(table, tree.path(co.BACKBLAZE_FILTERED))
    tree.record(co.BACKBLAZE_FILTERED)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert "[FAIL] backblaze date range" in out and "first=none last=none distinct_dates=0" in out


# -- A2: labels and drive split --

def rewrite_parquet(tree, rel, change):
    frame = pd.read_parquet(tree.path(rel))
    frame = change(frame)
    frame.to_parquet(tree.path(rel), index=False)
    tree.record(rel)
    tree.save()


def test_the_valid_tree_passes_the_label_and_split_checks(tree, capsys):
    code, out = run(capsys)
    assert code == 0, out
    for label in (f"backblaze labels rows and key order {co.BACKBLAZE_LABELS}",
                  f"backblaze labels values {co.BACKBLAZE_LABELS}",
                  f"backblaze labels totals {co.BACKBLAZE_LABELS}",
                  f"unique key (serial_number, date) {co.BACKBLAZE_LABELS}",
                  f"backblaze split one row per drive {co.BACKBLAZE_SPLIT}",
                  f"backblaze split values {co.BACKBLAZE_SPLIT}",
                  f"backblaze split drives unique {co.BACKBLAZE_SPLIT}",
                  f"backblaze split failed drives per model {co.BACKBLAZE_SPLIT}",
                  f"backblaze split stratum sizes {co.BACKBLAZE_SPLIT}",
                  f"backblaze split counts vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}",
                  f"sha256 {co.BACKBLAZE_LABELS}", f"sha256 {co.BACKBLAZE_SPLIT}",
                  f"rows vs PREPROCESSING.md {co.BACKBLAZE_LABELS}", f"rows vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}"):
        assert any(line.startswith(f"[PASS] {label}") for line in out.splitlines()), label


def test_a_label_table_with_a_missing_row_fails_the_row_and_key_check(tree, capsys):
    rewrite_parquet(tree, co.BACKBLAZE_LABELS, lambda f: f.iloc[1:])
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze labels rows and key order {co.BACKBLAZE_LABELS}: labels=" in out


def test_a_label_table_in_another_row_order_fails_the_key_order_check(tree, capsys):
    rewrite_parquet(tree, co.BACKBLAZE_LABELS, lambda f: f.iloc[::-1])
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze labels rows and key order {co.BACKBLAZE_LABELS}" in out


def test_a_label_value_outside_the_three_classes_fails(tree, capsys):
    def change(frame):
        frame.loc[frame.index[0], "label"] = 2
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_LABELS, change)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze labels values {co.BACKBLAZE_LABELS}: values=" in out


def test_a_label_changed_to_another_valid_class_fails_the_totals(tree, capsys):
    def change(frame):
        row = frame.index[frame["label"] == 1][0]
        frame.loc[row, "label"] = 0
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_LABELS, change)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze labels totals {co.BACKBLAZE_LABELS}: " in out and "; all: recorded=" in out
    assert f"[PASS] backblaze labels values {co.BACKBLAZE_LABELS}" in out


def test_a_duplicated_label_key_fails_the_unique_key_check(tree, capsys):
    rewrite_parquet(tree, co.BACKBLAZE_LABELS, lambda f: pd.concat([f, f.iloc[:1]], ignore_index=True))
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] unique key (serial_number, date) {co.BACKBLAZE_LABELS}: duplicate_rows=1" in out


def test_a_missing_label_file_is_missing(tree, capsys):
    tree.path(co.BACKBLAZE_LABELS).unlink()
    code, out = run(capsys)
    assert code == 1
    assert f"[MISSING] backblaze labels totals {co.BACKBLAZE_LABELS}" in out
    assert f"[MISSING] backblaze split counts vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}" in out
    assert "[FAIL]" not in out and out.rstrip().endswith("RESULT: INCOMPLETE")


def test_a_drive_missing_from_the_split_fails_the_one_row_per_drive_check(tree, capsys):
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, lambda f: f.iloc[1:])
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze split one row per drive {co.BACKBLAZE_SPLIT}: split_rows=" in out
    assert "missing=1" in out


def test_a_drive_in_two_splits_fails_the_uniqueness_check(tree, capsys):
    def change(frame):
        copy = frame.iloc[:1].copy()
        copy["split"] = "test" if frame.iloc[0]["split"] != "test" else "val"
        return pd.concat([frame, copy], ignore_index=True)
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, change)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze split drives unique {co.BACKBLAZE_SPLIT}: duplicate_drives=1" in out


def test_an_unknown_split_name_fails(tree, capsys):
    def change(frame):
        frame.loc[frame.index[0], "split"] = "holdout"
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, change)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze split values {co.BACKBLAZE_SPLIT}: unknown_values=['holdout']" in out


def test_a_split_without_a_failed_drive_of_some_model_fails(tree, capsys):
    def change(frame):
        frame.loc[(frame["model"] == TOSHIBA) & frame["ever_failed"], "split"] = "train"
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, change)
    code, out = run(capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith("[FAIL] backblaze split failed drives per model"))
    assert f"{TOSHIBA}/val" in line and f"{TOSHIBA}/test" in line
    assert f"[FAIL] backblaze split stratum sizes {co.BACKBLAZE_SPLIT}" in out


def test_a_stratum_more_than_one_drive_off_target_fails_the_size_check(tree, capsys):
    def change(frame):
        frame.loc[(frame["model"] == SEAGATE) & frame["ever_failed"], "split"] = "train"
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, change)
    code, out = run(capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith("[FAIL] backblaze split stratum sizes"))
    assert f"{SEAGATE} failed=True val: 0 vs 1.35" in line and f"{SEAGATE} failed=True train: 9 vs 6.30" in line


def test_split_counts_that_differ_from_preprocessing_fail(tree, capsys):
    def change(frame):
        failed = frame[(frame["model"] == SEAGATE) & frame["ever_failed"]]
        a = failed[failed["split"] == "train"].index[0]
        b = failed[failed["split"] == "val"].index[0]
        frame.loc[[a, b], "split"] = ["val", "train"]
        return frame
    rewrite_parquet(tree, co.BACKBLAZE_SPLIT, change)
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze split counts vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}: " in out
    assert f"[PASS] backblaze split stratum sizes {co.BACKBLAZE_SPLIT}" in out


def test_missing_a2_lines_fail_the_split_count_check(tree, capsys):
    tree.a2_lines = []
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] backblaze split counts vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}" in out


def test_committed_only_skips_the_label_and_split_checks(tree, capsys):
    shutil.rmtree(tree.root / "data")
    code, out = run(capsys, "--committed-only")
    assert code == 0, out
    for label in (f"backblaze labels rows and key order {co.BACKBLAZE_LABELS}",
                  f"backblaze labels totals {co.BACKBLAZE_LABELS}",
                  f"unique key (serial_number, date) {co.BACKBLAZE_LABELS}",
                  f"backblaze split one row per drive {co.BACKBLAZE_SPLIT}",
                  f"backblaze split counts vs PREPROCESSING.md {co.BACKBLAZE_SPLIT}"):
        assert f"[SKIP] {label}: not tracked in git" in out
    assert f"[SKIP] sha256 {co.BACKBLAZE_LABELS}: not tracked in git" in out


def test_parse_split_lines_reads_only_the_a2_section():
    text = ("## B2 — x\n- `a2_split split=train model=ZZ drives=9 failed_drives=9 positive=9 negative=9 censored=9`\n"
            "## A2 — Horizon labels and drive split\n"
            "- `a2_split split=train model=TOSHIBA MG07ACA14TA drives=5 failed_drives=1 positive=2 negative=3 censored=4`\n"
            "- `a2_split split=val model=all drives=1 failed_drives=0 positive=0 negative=0 censored=1`\n"
            "## B6 — y\n")
    assert co.parse_split_lines(text) == {("train", "TOSHIBA MG07ACA14TA"): (5, 1, 2, 3, 4),
                                          ("val", "all"): (1, 0, 0, 0, 1)}
    assert co.parse_split_lines("no section") == {}
