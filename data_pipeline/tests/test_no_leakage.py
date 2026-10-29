"""
Leakage test for per-drive timelines (B2).

Rolling-window features must be trailing: no column of the timelines table
may use information from a later date than the row it describes.

Method: cut the input at a date `d`, recompute the timelines on the
truncated data, and compare to the same rows computed on the full data.
They must be identical for every row with date <= d. A second test checks
the calendar-window differences against a brute-force reference.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from data_pipeline.backblaze import timelines  # noqa: E402

FILTERED = timelines.INPUT
SAMPLE_DRIVES = 400


def synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=60, freq="D")
    rows = []
    for k in range(8):
        keep = rng.random(len(days)) > 0.15
        keep[: int(rng.integers(0, 10))] = False
        d = days[keep]
        n = len(d)
        base = rng.integers(0, 100, size=len(timelines.SMART_COLUMNS))
        steps = rng.integers(0, 4, size=(n, len(timelines.SMART_COLUMNS))).cumsum(axis=0)
        vals = (base + steps).astype("float64")
        vals[rng.random(vals.shape) < 0.05] = np.nan
        frame = pd.DataFrame(vals, columns=list(timelines.SMART_COLUMNS))
        frame.insert(0, "serial_number", f"S{k}")
        frame.insert(1, "model", "M")
        frame.insert(2, "date", d.date)
        frame.insert(3, "capacity_bytes", 4_000_000_000_000)
        frame.insert(4, "failure", 0)
        if k % 3 == 0 and n:
            frame.loc[frame.index[-1], "failure"] = 1
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)


def assert_prefix_identical(df: pd.DataFrame, cut: pd.Timestamp) -> None:
    full = timelines.build_timelines(df)
    kept = df[pd.to_datetime(df["date"]) <= cut]
    truncated = timelines.build_timelines(kept)
    expected = full[full["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, expected, check_exact=True)
    assert timelines.to_arrow(truncated).equals(timelines.to_arrow(expected))


@pytest.mark.parametrize("cut", ["2022-01-15", "2022-02-01", "2022-02-20"])
def test_truncated_and_full_features_match_for_past_rows(cut):
    assert_prefix_identical(synthetic_frame(), pd.Timestamp(cut))


@pytest.mark.skipif(not FILTERED.exists(), reason="backblaze_filtered.parquet not built")
@pytest.mark.parametrize("cut", ["2022-02-01", "2022-03-01"])
def test_truncated_and_full_features_match_on_extract(cut):
    df = pq.read_table(
        FILTERED, columns=timelines.CORE_IN + list(timelines.SMART_COLUMNS)
    ).to_pandas()
    serials = sorted(df["serial_number"].unique())[:SAMPLE_DRIVES]
    failing = df.loc[df["failure"] == 1, "serial_number"].unique()[:SAMPLE_DRIVES]
    sample = df[df["serial_number"].isin(set(serials) | set(failing))]
    assert_prefix_identical(sample, pd.Timestamp(cut))


def test_window_differences_match_brute_force():
    df = synthetic_frame(seed=3)
    out = timelines.build_timelines(df)
    src = df.assign(date=pd.to_datetime(df["date"])).set_index(["serial_number", "date"])
    for _, row in out.iterrows():
        hist = src.loc[row["serial_number"]]
        for name, col, win in timelines.feature_spec():
            if win is None:
                continue
            w = hist[(hist.index >= row["date"] - pd.Timedelta(days=win)) & (hist.index <= row["date"])]
            w = w.sort_index()
            if len(w) < timelines.MIN_OBS_DIFF:
                want = np.nan
            else:
                want = w[col].iloc[-1] - w[col].iloc[0]
            got = row[name]
            assert (np.isnan(want) and np.isnan(got)) or want == got, (row["serial_number"], row["date"], name)


def test_timelines_have_no_whole_drive_columns():
    out = timelines.build_timelines(synthetic_frame())
    banned = {"last_seen", "days_observed", "failed_in_window", "failure_date"}
    assert not banned & set(out.columns)
