"""
Schema-alignment test for the Backblaze extract (B1).

Asserts that aligning daily files never silently drops a column present
in some files but absent in others: missing columns become NaN and are
counted, never dropped.
"""
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from data_pipeline.backblaze import extract  # noqa: E402

DAY1 = (
    "date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_5_normalized\n"
    "2022-01-01,A,ST4000DM000,4000787030016,0,1,100\n"
    "2022-01-01,B,OTHER,4000787030016,0,2,100\n"
)
DAY2 = (
    "date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_187_raw\n"
    "2022-01-02,A,ST4000DM000,4000787030016,0,1,7\n"
)


def test_aligned_columns_match_union_of_daily_files(tmp_path):
    (tmp_path / "2022-01-01.csv").write_text(DAY1)
    (tmp_path / "2022-01-02.csv").write_text(DAY2)
    files = sorted(tmp_path.glob("*.csv"))
    headers = {f.name: extract.read_header(f) for f in files}

    columns = extract.build_column_list(headers)

    assert columns == extract.CORE_COLUMNS + ["smart_5_raw", "smart_187_raw"]

    day1 = extract.align_frame(extract.read_day(files[0], headers[files[0].name], columns), columns)
    day2 = extract.align_frame(extract.read_day(files[1], headers[files[1].name], columns), columns)
    assert list(day1.columns) == columns
    assert list(day2.columns) == columns
    assert day1["smart_187_raw"].isna().all()
    assert day2["smart_187_raw"].notna().all()
    assert not day1.empty and pd.api.types.is_float_dtype(day1["smart_187_raw"])
