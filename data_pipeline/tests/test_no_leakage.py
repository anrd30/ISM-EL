"""
Leakage test for per-drive timelines (B2).

Rolling-window features must be trailing: no feature may use information
from a later date than the row it describes.

Method: cut the raw input at a date `d`, recompute features on the
truncated data, and compare to the same rows' features computed on the
full dataset. They must be identical for every row with date <= d.
"""
import pytest


@pytest.mark.skip(reason="backblaze/timelines.py not implemented")
def test_truncated_and_full_features_match_for_past_rows():
    # full = build_timelines(full_df)
    # cut_date = <a date in the middle of the sample>
    # truncated = build_timelines(full_df[full_df.date <= cut_date])
    # compare = full[full.date <= cut_date]
    # pd.testing.assert_frame_equal(truncated, compare)
    raise NotImplementedError
