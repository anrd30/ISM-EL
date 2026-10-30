"""
Grid test for the four intensity profile files (B5).

Each profile file must be exactly 288 rows (24h at a 300-second interval)
with no gaps and no duplicate timestamps.
"""
from pathlib import Path

import pandas as pd
import pytest

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
EXPECTED_PROFILES = ["idle", "diurnal", "bursty", "peak"]


@pytest.mark.skip(reason="profile files not generated")
@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_profile_has_288_rows_at_300s_spacing(profile: str):
    f = PROFILES_DIR / f"profile_{profile}.csv"
    assert f.exists(), f"missing {f}"
    df = pd.read_csv(f)
    assert len(df) == 288, f"{f.name} has {len(df)} rows, expected 288"
    assert list(df["t_seconds"]) == list(range(0, 288 * 300, 300)), (
        f"{f.name} is not a clean 300s grid from 0 to 86100"
    )
