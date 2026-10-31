"""
Grid test for the four intensity profile files (B5).

Each profile file must be exactly 288 rows (24h at a 300-second interval)
with no gaps and no duplicate timestamps, and must satisfy the contract's
replay-frame validator.
"""
from pathlib import Path

import pandas as pd
import pytest

from contracts import contracts
from data_pipeline.tencent import build_profiles
from sim import scenario

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"
EXPECTED_PROFILES = ["idle", "diurnal", "bursty", "peak"]
EXPECTED_SAFE_ACTIONS = {"idle": 4, "diurnal": 3, "bursty": 2, "peak": 1}


@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_profile_has_288_rows_at_300s_spacing(profile: str):
    f = PROFILES_DIR / f"profile_{profile}.csv"
    assert f.exists(), f"missing {f}"
    df = pd.read_csv(f)
    assert len(df) == 288, f"{f.name} has {len(df)} rows, expected 288"
    assert list(df["t_seconds"]) == list(range(0, 288 * 300, 300)), (
        f"{f.name} is not a clean 300s grid from 0 to 86100"
    )


@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_profile_satisfies_replay_contract(profile: str):
    df = pd.read_csv(PROFILES_DIR / f"profile_{profile}.csv")
    assert list(df.columns) == list(contracts.REPLAY_COLUMNS)
    contracts.validate_replay_frame(df)


def committed_ladder(profile: str) -> dict:
    frame = pd.read_csv(PROFILES_DIR / f"profile_{profile}.csv")
    return build_profiles.latency_ladder(frame, profile)


def test_safe_actions_per_profile_on_the_committed_csvs():
    """Number of rebuild allocations whose latency-violating share is within the contract limit."""
    assert {p: committed_ladder(p)["safe_actions"] for p in EXPECTED_PROFILES} == EXPECTED_SAFE_ACTIONS


@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_every_profile_has_at_least_one_safe_action(profile: str):
    assert committed_ladder(profile)["safe_actions"] >= 1


@pytest.mark.parametrize("profile", EXPECTED_PROFILES)
def test_simulator_reads_the_sidecar_as_real_data_with_the_bare_profile_name(profile: str):
    trace = scenario.load_replay_csv(PROFILES_DIR / f"profile_{profile}.csv")
    assert trace.synthetic is False
    assert trace.profile == profile

