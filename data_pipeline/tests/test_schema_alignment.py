"""
Schema-alignment test for the Backblaze extract (B1).

Asserts that aligning daily files never silently drops a column present
in some files but absent in others: missing columns become NaN and are
counted, never dropped.
"""
import pytest


@pytest.mark.skip(reason="backblaze/extract.py not implemented")
def test_aligned_columns_match_union_of_daily_files():
    # 1. Build the per-file column sets for the quarter under test.
    # 2. Run the alignment step.
    # 3. Assert the output's column set equals the union of all per-file
    #    columns (restricted to the chosen SMART subset + core columns).
    # 4. Assert every column absent from a given day is NaN for that
    #    day's rows, not silently absent from the output.
    raise NotImplementedError
