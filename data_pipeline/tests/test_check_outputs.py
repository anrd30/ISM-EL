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
import pytest

from contracts import contracts
from data_pipeline import check_outputs as co

REAL_PROFILES = Path(co.__file__).resolve().parent / "profiles"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
DAY1, DAY2 = datetime.date(2022, 1, 1), datetime.date(2022, 1, 2)
KEYLESS_SUFFIXES = (".png", ".json")


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
        manifest = {"files": files, "sources": {}, "profiles": profiles}
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
            + "\n".join(lines) + "\n\n## B6 — Acceptance\n"
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
    (root / "profiles").mkdir(parents=True)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    for name in co.PROFILE_NAMES:
        shutil.copy(REAL_PROFILES / f"profile_{name}.csv", root / "profiles")
        shutil.copy(REAL_PROFILES / f"profile_{name}.meta.json", root / "profiles")
        pd.read_csv(root / co.PROFILE_CSV[name]).to_parquet(root / co.PROFILE_PARQUET[name], index=False)
    (root / co.PROFILE_PLOT).write_bytes(PNG_BYTES)

    tables = {
        co.BACKBLAZE_FILTERED: pd.DataFrame({"serial_number": ["a", "a", "b"], "date": [DAY1, DAY2, DAY1],
                                             "failure": [0, 0, 1]}),
        co.BACKBLAZE_TIMELINES: pd.DataFrame({"serial_number": ["a", "a", "b"], "date": [DAY1, DAY2, DAY1],
                                              "gap_days_before": [None, 1, None]}),
        co.BACKBLAZE_DRIVE_INDEX: pd.DataFrame({"serial_number": ["a", "b"], "days_observed": [2, 1]}),
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
    assert code == 0, out
    assert f"[PASS] unique key (serial_number, date) {co.BACKBLAZE_FILTERED}: duplicate_rows=0" in out


def test_a_genuine_duplicate_key_is_a_failure(tree, capsys):
    frame = pd.DataFrame({"serial_number": ["a", "a", "b"], "date": [DAY1, DAY1, DAY1], "failure": [0, 0, 1]})
    frame.to_parquet(tree.path(co.BACKBLAZE_TIMELINES), index=False)
    tree.record(co.BACKBLAZE_TIMELINES)
    tree.save()
    code, out = run(capsys)
    assert code == 1
    assert f"[FAIL] unique key (serial_number, date) {co.BACKBLAZE_TIMELINES}: duplicate_rows=1" in out


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
