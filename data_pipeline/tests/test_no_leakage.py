"""
Leakage test for B2 (per-drive timelines).

CLAUDE.md §7: "never let a feature use information from a later date than
the row it describes... all rolling-window features must be trailing."

The test: cut the raw input at a date `d`, recompute B2's features on the
truncated data, and compare to the same rows' features computed on the
full dataset. They must be byte-identical for every row with
date <= d. If anything differs, a feature is reading the future.

Scaffold only until data_pipeline/backblaze/timelines.py exists.
"""
import pytest


@pytest.mark.skip(reason="B2 timelines.py not implemented yet — see CLAUDE.md §8 execution order")
def test_truncated_and_full_features_match_for_past_rows():
    # Once implemented:
    #   full = build_timelines(full_df)
    #   cut_date = <some date in the middle of the sample>
    #   truncated = build_timelines(full_df[full_df.date <= cut_date])
    #   compare = full[full.date <= cut_date]
    #   pd.testing.assert_frame_equal(truncated, compare)
    raise NotImplementedError
