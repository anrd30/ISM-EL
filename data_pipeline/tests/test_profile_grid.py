"""
Grid test for B5 (four intensity profiles).

Each profile file must be exactly 288 rows (24h at a 300-second interval)
with no gaps and no duplicate timestamps. This is part of the literal
B-lane acceptance check: "the four profiles are visibly different when
plotted" and must be "checkable in under a minute" (CLAUDE.md §4).

Scaffold only until data_pipeline/tencent/profiles.py exists and B5 has
produced output files under profiles/.
"""
from pathlib import Path

import pandas as pd
import pytest

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
EXPECTED_PROFILES = ["idle", "diurnal", "bursty", "peak"]


@pytest.mark.skip(reason="B5 profile files not generated yet — see CLAUDE.md §8 execution order")
@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_profile_has_288_rows_at_300s_spacing(profile: str):
    f = PROFILES_DIR / f"profile_{profile}.csv"
    assert f.exists(), f"missing {f}"
    df = pd.read_csv(f)
    assert len(df) == 288, f"{f.name} has {len(df)} rows, expected 288"
    assert list(df["t_seconds"]) == list(range(0, 288 * 300, 300)), (
        f"{f.name} is not a clean 300s grid from 0 to 86100"
    )
