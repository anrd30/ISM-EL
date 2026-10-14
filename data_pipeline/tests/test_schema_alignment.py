"""
Schema-alignment test for B1 (Backblaze extract).

SMART columns change between quarterly files (CLAUDE.md §4, B1: "Do not
concatenate Backblaze quarters without aligning the SMART schema first").
This test asserts that the aligned-extract step never silently drops a
column present in some daily files but absent in others — missing columns
must become NaN and be counted, never dropped.

Currently a scaffold: fill in once backblaze/extract.py exists and A1 is
confirmed. Intentionally skipped until then so `pytest` stays green on a
fresh checkout.
"""
import pytest


@pytest.mark.skip(reason="B1 extract.py not implemented yet — see CLAUDE.md §8 execution order")
def test_aligned_columns_match_union_of_daily_files():
    # Once implemented:
    #   1. Build the per-file column sets for the quarter under test.
    #   2. Run the real alignment step.
    #   3. Assert the output's column set equals the union of all per-file
    #      columns (restricted to the chosen SMART subset + core columns).
    #   4. Assert every column absent from a given day is NaN for that
    #      day's rows, not silently absent from the output.
    raise NotImplementedError
