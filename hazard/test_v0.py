"""Tests for the v0 heuristic hazard model.

These check interface conformance, internal consistency with the frozen
contract's arithmetic, boundedness and monotonicity of the heuristic. None of
them says anything about predictive accuracy: the model is not fitted to data.
"""

from __future__ import annotations

import math

import pytest

from contracts.contracts import (
    GROUP_SIZE,
    HORIZON_HOURS,
    ContractViolation,
    HazardModel,
    Observation,
    aggregate_group_hazard,
    hazard_p_to_per_hour,
)
from hazard.v0 import (
    ANNUALISED_FAILURE_RATE,
    FEATURE_NAMES,
    HOURS_PER_YEAR,
    MULTIPLIER_CAP,
    HeuristicHazardV0,
    baseline_horizon_probability,
    group_hazard,
    severity_multiplier,
)

ERROR_FEATURES = ("smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw")
STRONG_FEATURES = ("smart_187_raw", "smart_197_raw")

# The contract's per-drive-hour hazard, 0.0211 / 8760, quoted as "about 2.41e-6".
CONTRACT_HAZARD_PER_HOUR = 2.41e-6

MODEL = HeuristicHazardV0()


def healthy() -> dict[str, float]:
    """All five features present, no degradation."""
    return {name: 0.0 for name in FEATURE_NAMES}


def p_of(features: dict[str, float], horizon_hours: float = HORIZON_HOURS) -> float:
    return MODEL(features, horizon_hours)[0]


def test_satisfies_protocol_and_returns_builtin_float_pair() -> None:
    assert isinstance(MODEL, HazardModel)
    out = MODEL(healthy())
    assert isinstance(out, tuple) and len(out) == 2
    assert type(out[0]) is float and type(out[1]) is float
    # Also with a degraded and an empty record.
    for feats in ({"smart_197_raw": 3.0}, {}):
        out = MODEL(feats)
        assert type(out[0]) is float and type(out[1]) is float


def test_default_horizon_is_the_contract_horizon() -> None:
    assert MODEL(healthy()) == MODEL(healthy(), HORIZON_HOURS)


def test_baseline_matches_contract_per_hour_hazard() -> None:
    # The contract derives lambda = 0.0211 / 8760 directly (annual -> hourly).
    # This model goes annual -> horizon (p = 1 - exp(-lambda * H)) and the
    # contract's hazard_p_to_per_hour then goes horizon -> hourly
    # (1 - (1 - p) ** (1 / H) = 1 - exp(-lambda)). Both assume a constant rate,
    # so the two routes differ only by the discrete-versus-continuous gap
    # lambda - (1 - exp(-lambda)) ~ lambda^2 / 2, a relative difference of
    # about lambda / 2 = 1.2e-6.
    direct = ANNUALISED_FAILURE_RATE / HOURS_PER_YEAR
    via_horizon = hazard_p_to_per_hour(baseline_horizon_probability())
    assert via_horizon == pytest.approx(direct, rel=1e-5)
    # And against the rounded figure quoted in the contract (2.4087e-6 vs 2.41e-6).
    assert via_horizon == pytest.approx(CONTRACT_HAZARD_PER_HOUR, rel=2e-3)


def test_baseline_follows_afr_and_horizon() -> None:
    # No precomputed literal: changing either input changes the result consistently.
    p_double_afr = baseline_horizon_probability(afr=2 * ANNUALISED_FAILURE_RATE)
    assert hazard_p_to_per_hour(p_double_afr) == pytest.approx(
        2 * ANNUALISED_FAILURE_RATE / HOURS_PER_YEAR, rel=1e-5
    )
    p_short = baseline_horizon_probability(horizon_hours=24.0)
    assert hazard_p_to_per_hour(p_short, 24.0) == pytest.approx(
        ANNUALISED_FAILURE_RATE / HOURS_PER_YEAR, rel=1e-5
    )
    assert HeuristicHazardV0(afr=0.04)(healthy())[0] > MODEL(healthy())[0]


def test_healthy_drive_returns_baseline() -> None:
    p, u = MODEL(healthy())
    assert p == pytest.approx(baseline_horizon_probability(), rel=1e-12)
    assert 0.0 <= p < 1.0
    assert u > 0.0
    assert severity_multiplier(healthy()) == (1.0, 0)


@pytest.mark.parametrize("name", ERROR_FEATURES)
def test_error_counts_are_monotone_and_raise_p(name: str) -> None:
    base = p_of(healthy())
    previous = base
    for count in (1, 2, 5, 10, 100, 1000, 1e6):
        feats = healthy()
        feats[name] = float(count)
        p = p_of(feats)
        assert p >= previous
        previous = p
    # Any nonzero count strictly raises p (below the feature's saturation).
    feats = healthy()
    feats[name] = 1.0
    assert p_of(feats) > base
    # A large count raises p measurably.
    feats[name] = 1000.0
    assert p_of(feats) > 1.05 * base


def test_age_is_monotone() -> None:
    previous = p_of(healthy())
    for hours in (1000, 20000, 43800, 1e7):
        feats = healthy()
        feats["smart_9_raw"] = float(hours)
        p = p_of(feats)
        assert p >= previous
        previous = p


def test_strong_features_outweigh_weak_ones() -> None:
    def p_with(name: str, count: float) -> float:
        feats = healthy()
        feats[name] = count
        return p_of(feats)

    for strong in STRONG_FEATURES:
        for weak in ("smart_5_raw", "smart_188_raw"):
            assert p_with(strong, 10.0) > p_with(weak, 10.0)


def test_multiplier_is_bounded_for_absurd_values() -> None:
    cap_p = baseline_horizon_probability()
    cap_p = -math.expm1(-MULTIPLIER_CAP * -math.log1p(-cap_p))
    absurd = {name: 1e300 for name in FEATURE_NAMES}
    inf = {name: math.inf for name in FEATURE_NAMES}
    for feats in (absurd, inf):
        m, missing = severity_multiplier(feats)
        assert m == pytest.approx(MULTIPLIER_CAP)
        assert missing == 0
        p, u = MODEL(feats)
        assert p == pytest.approx(cap_p, rel=1e-9)
        assert p <= cap_p * (1 + 1e-9)
        assert 0.0 <= p < 1.0
        assert 0.0 < u <= 0.5


def test_multiplier_range_over_plausible_inputs() -> None:
    samples = [
        healthy(),
        {**healthy(), "smart_9_raw": 20000.0},
        {**healthy(), "smart_197_raw": 1.0},
        {**healthy(), "smart_197_raw": 10.0},
        {**healthy(), "smart_5_raw": 8.0, "smart_9_raw": 30000.0},
        {**healthy(), "smart_187_raw": 10.0, "smart_197_raw": 10.0},
        {**healthy(), "smart_187_raw": 50.0, "smart_197_raw": 50.0, "smart_5_raw": 200.0},
        {name: 1e9 for name in FEATURE_NAMES},
    ]
    multipliers = [severity_multiplier(s)[0] for s in samples]
    assert min(multipliers) == 1.0
    assert max(multipliers) == pytest.approx(MULTIPLIER_CAP)
    # Not constant and not routinely saturated: most samples sit strictly inside.
    assert sum(m < MULTIPLIER_CAP - 1e-9 for m in multipliers) >= len(samples) - 2
    assert len({round(m, 6) for m in multipliers}) >= 6
    assert all(1.0 <= m <= MULTIPLIER_CAP for m in multipliers)


def test_missing_features_are_no_evidence_and_widen_uncertainty() -> None:
    full = healthy()
    p_full, u_full = MODEL(full)
    p_empty, u_empty = MODEL({})
    # No evidence of degradation: baseline, not an imputed value, and no crash.
    assert p_empty == pytest.approx(p_full, rel=1e-12)
    assert severity_multiplier({}) == (1.0, len(FEATURE_NAMES))
    assert u_empty > u_full
    # Removing one zero-valued feature at a time widens the uncertainty monotonically.
    previous = u_full
    feats = dict(full)
    for name in FEATURE_NAMES:
        del feats[name]
        u = MODEL(feats)[1]
        assert u > previous
        previous = u
    # Missing contributes nothing, even next to a present degraded feature.
    p_one = MODEL({"smart_197_raw": 5.0})[0]
    p_one_full = MODEL({**healthy(), "smart_197_raw": 5.0})[0]
    assert p_one == pytest.approx(p_one_full, rel=1e-12)


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0, "abc", True])
def test_unusable_values_count_as_missing(bad: object) -> None:
    feats = {**healthy(), "smart_197_raw": bad}
    m, missing = severity_multiplier(feats)  # type: ignore[arg-type]
    assert m == 1.0
    assert missing == 1


def test_uncertainty_within_contract_bound_for_all_inputs() -> None:
    inputs: list[dict[str, float]] = [
        {},
        healthy(),
        {name: 1e300 for name in FEATURE_NAMES},
        {name: math.inf for name in FEATURE_NAMES},
        {"smart_197_raw": 1e12},
        {"smart_5_raw": float("nan")},
    ]
    for horizon in (1.0, HORIZON_HOURS, 1e5, 1e9):
        for feats in inputs:
            p, u = MODEL(feats, horizon)
            assert 0.0 <= p < 1.0
            assert 0.0 < u <= 0.5


def test_horizon_is_honoured() -> None:
    p_default = p_of(healthy())
    p_short = p_of(healthy(), 72.0)
    p_long = p_of(healthy(), 2 * HORIZON_HOURS)
    assert p_short < p_default < p_long
    assert p_short == pytest.approx(baseline_horizon_probability(72.0))
    # The per-hour rate recovered through the contract is horizon-independent.
    assert hazard_p_to_per_hour(p_short, 72.0) == pytest.approx(
        hazard_p_to_per_hour(p_default), rel=1e-9
    )
    # And so are degraded drives.
    sick = {**healthy(), "smart_197_raw": 10.0}
    assert hazard_p_to_per_hour(p_of(sick, 72.0), 72.0) == pytest.approx(
        hazard_p_to_per_hour(p_of(sick)), rel=1e-9
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_invalid_horizon_raises_contract_violation(bad: float) -> None:
    with pytest.raises(ContractViolation):
        MODEL(healthy(), bad)


@pytest.mark.parametrize("n", [1, GROUP_SIZE - 1])
def test_pair_accepted_by_aggregate_group_hazard(n: int) -> None:
    pair = MODEL({**healthy(), "smart_187_raw": 2.0})
    hazard, unc = aggregate_group_hazard([pair] * n)
    assert hazard > 0.0 and unc > 0.0
    assert type(hazard) is float and type(unc) is float


def test_nine_healthy_drives_tie_to_contract_table() -> None:
    per_drive_rate = hazard_p_to_per_hour(MODEL(healthy())[0])
    hazard, unc = group_hazard(MODEL, [healthy()] * (GROUP_SIZE - 1))
    assert hazard == pytest.approx((GROUP_SIZE - 1) * per_drive_rate, rel=1e-9)
    assert hazard == pytest.approx(2.17e-5, rel=5e-3)  # contract table, 1x row
    obs = Observation(
        fg_read_iops=0.0,
        fg_write_iops=0.0,
        fg_bw=0.0,
        latency_now=10.0,
        latency_target=10.0,
        hazard=hazard,
        hazard_uncertainty=unc,
        redundancy_left=1.0,
        progress=0.0,
        work_remaining=12000.0,
        alloc_current=20.0,
        bw_available=200.0,
    )
    obs.validate()
    field5 = float(obs.to_vector(normalized=True)[5])
    assert field5 == pytest.approx(0.12, abs=0.005)  # contract table: 1x -> 0.12


def test_group_hazard_uses_contract_aggregation() -> None:
    feats = [healthy(), {**healthy(), "smart_197_raw": 4.0}, {}]
    expected = aggregate_group_hazard([MODEL(f) for f in feats])
    assert group_hazard(MODEL, feats) == expected
    with pytest.raises(ContractViolation):
        group_hazard(MODEL, [])
    with pytest.raises(ContractViolation):
        group_hazard(MODEL, [healthy()] * GROUP_SIZE)


def test_single_sick_drive_moves_field5_but_does_not_saturate_it() -> None:
    # A capped drive among nine: (8 + 10) / 9 = 2x baseline group hazard.
    sick = {name: 1e9 for name in FEATURE_NAMES}
    hazard, _ = group_hazard(MODEL, [sick] + [healthy()] * (GROUP_SIZE - 2))
    baseline, _ = group_hazard(MODEL, [healthy()] * (GROUP_SIZE - 1))
    assert hazard / baseline == pytest.approx(2.0, rel=1e-3)
    # Nine capped drives sit above the contract's saturation point (8.3x).
    all_sick, _ = group_hazard(MODEL, [sick] * (GROUP_SIZE - 1))
    assert all_sick / baseline == pytest.approx(MULTIPLIER_CAP, rel=1e-3)
    assert all_sick > 1.8e-4
