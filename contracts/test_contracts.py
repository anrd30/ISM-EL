"""Tests for the frozen interface contract (``contracts.contracts``).

Every workstream codes against the names, orders, units, scales and error
behaviour pinned here, so a failure in this module means a consumer of the
interface would silently receive different data, not merely a changed
implementation detail.
"""

import dataclasses
import enum
import inspect
import math
import re
import subprocess
import sys
import typing
import warnings
from pathlib import Path

import numpy as np
import pytest

from contracts import contracts as C
from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    CONTRACT_VERSION,
    CONTROL_INTERVAL_S,
    DISK_INDEX_COLUMNS,
    DRIVE_CAPACITY_GB,
    GROUP_DATA_DRIVES,
    GROUP_PARITY_DRIVES,
    GROUP_SIZE,
    GROUP_TOLERATED_FAILURES,
    HORIZON_DAYS,
    HORIZON_HOURS,
    INFO_KEYS,
    LATENCY_LIMIT_FACTOR,
    LATENCY_VIOLATION_FRACTION_MAX,
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    OBS_DIM,
    OBS_FIELDS,
    OBS_SCALES,
    REBUILD_WORK_GB,
    REPLAY_COLUMNS,
    REPLAY_DTYPES,
    RISK_BUDGET_EPS,
    ContractViolation,
    HazardModel,
    Observation,
    RewardConfig,
    TerminationReason,
    action_to_mbps,
    aggregate_group_hazard,
    compute_reward,
    degraded_cost,
    hazard_p_to_per_hour,
    hazard_uncertainty_to_per_hour,
    kbps_to_mbps,
    latency_degradation,
    latency_violation,
    validate_info,
    termination_flags,
    validate_replay_frame,
    validate_replay_grid,
)

EXPECTED_FIELDS = (
    "fg_read_iops",
    "fg_write_iops",
    "fg_bw",
    "latency_now",
    "latency_target",
    "hazard",
    "hazard_uncertainty",
    "redundancy_left",
    "progress",
    "work_remaining",
    "alloc_current",
    "bw_available",
)

EXPECTED_SCALES = {
    "fg_read_iops": 20000.0,
    "fg_write_iops": 20000.0,
    "fg_bw": 200.0,
    "latency_now": 50.0,
    "latency_target": 50.0,
    "hazard": 1.8e-4,
    "hazard_uncertainty": 1.8e-4,
    "redundancy_left": 2.0,
    "progress": 1.0,
    "work_remaining": 12000.0,
    "alloc_current": 120.0,
    "bw_available": 200.0,
}

# A mid-range observation.  Every value is exactly representable in float32
# (dyadic rationals), so a float32 round-trip must reproduce it bit-for-bit.
EXACT_OBS_KWARGS = dict(
    fg_read_iops=8192.0,
    fg_write_iops=3072.0,
    fg_bw=90.5,
    latency_now=22.5,
    latency_target=20.0,
    hazard=2.0 ** -14,
    hazard_uncertainty=2.0 ** -16,
    redundancy_left=1.0,
    progress=0.375,
    work_remaining=7500.0,
    alloc_current=80.0,
    bw_available=110.0,
)

# The same shape of observation with decimal values that float32 cannot hold
# exactly (e.g. 5e-5); only approximate round-trips are meaningful for these.
REALISTIC_OBS_KWARGS = dict(
    fg_read_iops=8123.0,
    fg_write_iops=3011.0,
    fg_bw=91.3,
    latency_now=22.7,
    latency_target=20.0,
    hazard=5e-5,
    hazard_uncertainty=2e-5,
    redundancy_left=1.0,
    progress=0.35,
    work_remaining=7800.0,
    alloc_current=80.0,
    bw_available=110.0,
)


def make_obs(**overrides) -> Observation:
    """Return a valid, mid-range observation with ``overrides`` applied."""
    kwargs = dict(REALISTIC_OBS_KWARGS)
    kwargs.update(overrides)
    return Observation(**kwargs)


def make_info(**overrides) -> dict:
    """Return a valid info dict with ``overrides`` applied."""
    info = {
        "latency_p95": 40.0,
        "latency_p99": 55.0,
        "violated_latency": False,
        "p_failure_before_complete": 1e-4,
        "violated_risk_budget": False,
        "alloc_clipped_by_filter": False,
        "recovery_complete": False,
    }
    info.update(overrides)
    return info


def _version_tuple(version):
    return tuple(int(part) for part in version.split("."))


def _changelog_rows():
    """Return ``(version, text)`` for each row of the changelog table in INTERFACES.md."""
    text = (Path(__file__).resolve().parent / "INTERFACES.md").read_text(encoding="utf-8")
    heading = re.search(r"^#+\s*(?:\d+\.\s*)?Changelog\s*$", text, re.MULTILINE | re.IGNORECASE)
    assert heading, "INTERFACES.md has no Changelog section"
    rows = []
    for line in text[heading.end():].splitlines():
        m = re.match(r"^\|\s*(\d+\.\d+\.\d+)\s*\|(.*)\|\s*$", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


_VERSION_SHAPE = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])")


def _strictly_ascending(versions):
    """True for a non-empty sequence of version tuples in strictly increasing order.

    A single row is trivially ascending; an empty table is not valid, because it
    would leave the drift test with no most-recent row to compare.
    """
    return bool(versions) and all(a < b for a, b in zip(versions, versions[1:]))


def _doc_version_strings_outside_changelog():
    """Return ``(line_number, version)`` for each version-shaped string in
    INTERFACES.md that is not inside the changelog section."""
    text = (Path(__file__).resolve().parent / "INTERFACES.md").read_text(encoding="utf-8")
    heading = re.search(r"^#+\s*(?:\d+\.\s*)?Changelog\s*$", text, re.MULTILINE | re.IGNORECASE)
    assert heading, "INTERFACES.md has no Changelog section"
    before = text[: heading.start()]
    found = []
    for lineno, line in enumerate(before.splitlines(), start=1):
        for m in _VERSION_SHAPE.finditer(line):
            found.append((lineno, m.group(0)))
    return found


def _expect_contract_violation(fn, *args, why):
    """Call ``fn``; pass if it raises ``ContractViolation``, fail with ``why`` otherwise.

    A different exception type (for example ``TypeError``) propagates and fails the
    test, which is the point: the named contract exception is the only boundary
    signal callers are told to catch.
    """
    try:
        out = fn(*args)
    except ContractViolation:
        return
    pytest.fail(f"{fn.__name__}{args!r} returned {out!r} instead of raising ContractViolation: {why}")


BOOL_KEYS = (
    "violated_latency",
    "violated_risk_budget",
    "alloc_clipped_by_filter",
    "recovery_complete",
)
NUMERIC_KEYS = ("latency_p95", "latency_p99", "p_failure_before_complete")


class TestConstants:
    def test_numeric_constants_are_frozen(self):
        """Pins every scalar constant; they define the problem all agents solve."""
        assert CONTROL_INTERVAL_S == 300
        assert HORIZON_DAYS == 30
        assert HORIZON_HOURS == 720.0
        assert RISK_BUDGET_EPS == 5e-3
        assert LATENCY_LIMIT_FACTOR == 1.5
        assert LATENCY_VIOLATION_FRACTION_MAX == 0.05
        assert MAX_EPISODE_STEPS == 2400
        assert ACTIONS_MBPS == (20, 50, 80, 120)
        assert N_ACTIONS == 4
        assert GROUP_DATA_DRIVES == 8
        assert GROUP_PARITY_DRIVES == 2
        assert GROUP_TOLERATED_FAILURES == 2
        assert DRIVE_CAPACITY_GB == 12000.0
        assert REBUILD_WORK_GB == 12000.0
        assert BW_TOTAL_MBPS == 200.0
        assert OBS_DIM == 12

    def test_group_size_value_and_derivation(self):
        """The group is the data drives plus the parity drives; one drive is already
        failed when an episode starts, leaving ``GROUP_SIZE - 1`` survivors."""
        assert GROUP_SIZE == 10
        assert GROUP_SIZE == GROUP_DATA_DRIVES + GROUP_PARITY_DRIVES
        assert GROUP_SIZE - 1 == 9

    def test_hazard_scales_are_group_level_aggregates(self):
        """Fields 5 and 6 are sums over the surviving drives, so their scale is
        ``(GROUP_SIZE - 1)`` times the per-drive saturation hazard of 2e-5.  A silent
        reversion to per-drive semantics would leave a scale that is wrong by 9x
        and feed the policy inputs that are 9x too large."""
        per_drive_scale = 2e-5
        for name in ("hazard", "hazard_uncertainty"):
            assert OBS_SCALES[name] == pytest.approx((GROUP_SIZE - 1) * per_drive_scale, rel=1e-9), (
                f"{name} scale is no longer (GROUP_SIZE - 1) * 2e-5: fields 5 and 6 "
                "must be group-level aggregates, not one drive's hazard"
            )

    def test_contract_version_is_semver(self):
        """The version is the key by which consumers and the changelog identify the
        frozen interface, so it must be a plain MAJOR.MINOR.PATCH string."""
        assert isinstance(CONTRACT_VERSION, str)
        assert re.fullmatch(r"\d+\.\d+\.\d+", CONTRACT_VERSION), CONTRACT_VERSION

    def test_changelog_rows_are_in_ascending_version_order(self):
        """The drift test below reads the last row as the most recent one, which is
        only valid while rows are appended in increasing version order."""
        versions = [_version_tuple(v) for v, _ in _changelog_rows()]
        assert versions, "no changelog rows found in INTERFACES.md"
        assert _strictly_ascending(versions), (
            f"changelog rows are not in strictly ascending version order: {versions}"
        )

    @pytest.mark.parametrize(
        "versions, expected",
        [
            ([(1, 0, 0)], True),  # a single row is trivially ordered
            ([(1, 0, 0), (1, 1, 0)], True),
            ([(1, 9, 0), (1, 10, 0)], True),  # numeric, not lexicographic, comparison
            ([(1, 1, 0), (1, 0, 0)], False),  # disorder
            ([(1, 0, 0), (1, 0, 0)], False),  # duplicate
            ([(1, 0, 0), (1, 2, 0), (1, 1, 0)], False),
            ([], False),  # an empty table cannot supply a most-recent row
        ],
    )
    def test_ordering_check_detects_disorder_and_accepts_single_row(self, versions, expected):
        """The ascending-order check must pass on one row and still fail on disorder,
        duplicates or an empty table once more rows exist, so it keeps guarding the
        assumption that the last row is the most recent."""
        assert _strictly_ascending(versions) is expected

    def test_code_and_changelog_versions_agree(self):
        """The code and its interface document are edited separately; if the latest
        changelog row and ``CONTRACT_VERSION`` differ, one author bumped without the other."""
        doc_version = _changelog_rows()[-1][0]
        if doc_version == CONTRACT_VERSION:
            return
        if _version_tuple(doc_version) > _version_tuple(CONTRACT_VERSION):
            behind = "contracts.py is behind INTERFACES.md"
        else:
            behind = "INTERFACES.md is behind contracts.py"
        raise AssertionError(
            f"{behind}: CONTRACT_VERSION={CONTRACT_VERSION!r} but the latest "
            f"changelog row is {doc_version!r}"
        )

    def test_every_version_string_outside_the_changelog_matches_the_code(self):
        """INTERFACES.md states the version in more than one place (the header table and
        the constants table, besides the changelog).  A bump that updates only some of
        them leaves the document contradicting itself, so every such string must equal
        ``CONTRACT_VERSION``."""
        found = _doc_version_strings_outside_changelog()
        assert len(found) >= 2, (
            f"expected the version in at least the header and constants tables, found {found}"
        )
        stale = [(line, v) for line, v in found if v != CONTRACT_VERSION]
        assert not stale, (
            f"INTERFACES.md has version strings that differ from CONTRACT_VERSION="
            f"{CONTRACT_VERSION!r} outside the changelog (line, value): {stale}; "
            "a bump updated some places and not others"
        )

    def test_derived_relations_hold(self):
        """Constants that are defined in terms of one another must not drift apart."""
        assert N_ACTIONS == len(ACTIONS_MBPS)
        assert GROUP_TOLERATED_FAILURES == GROUP_PARITY_DRIVES
        assert HORIZON_HOURS == HORIZON_DAYS * 24
        assert all(a <= BW_TOTAL_MBPS for a in ACTIONS_MBPS)
        assert list(ACTIONS_MBPS) == sorted(ACTIONS_MBPS)

    def test_episode_horizon_lets_the_slowest_baseline_complete(self):
        """The slowest constant allocation must be able to finish the rebuild inside
        the episode.  Otherwise the minimum-bandwidth baseline can never reach
        RECOVERY_COMPLETE, which invalidates every baseline comparison."""
        slowest_mbps = min(ACTIONS_MBPS)
        # Decimal units throughout, matching kbps_to_mbps: 1 GB = 1000 MB.
        seconds_needed = REBUILD_WORK_GB * 1000.0 / slowest_mbps
        seconds_available = MAX_EPISODE_STEPS * CONTROL_INTERVAL_S
        assert seconds_available > seconds_needed, (
            f"episode horizon of {seconds_available} s is shorter than the "
            f"{seconds_needed:.0f} s a constant {slowest_mbps} MB/s rebuild needs for "
            f"{REBUILD_WORK_GB} GB: the slowest baseline can never complete, so every "
            "baseline comparison is invalid"
        )

    def test_scales_track_the_physical_quantity_they_normalise(self):
        """A scale that disagrees with its physical bound would saturate or waste range."""
        assert OBS_SCALES["redundancy_left"] == GROUP_TOLERATED_FAILURES
        assert OBS_SCALES["work_remaining"] == REBUILD_WORK_GB
        assert OBS_SCALES["bw_available"] == BW_TOTAL_MBPS
        assert OBS_SCALES["alloc_current"] == max(ACTIONS_MBPS)

    def test_replay_column_tuples_are_frozen(self):
        """Column names and order are the schema the data pipeline must emit."""
        assert REPLAY_COLUMNS == (
            "t_seconds",
            "read_iops",
            "write_iops",
            "read_bw_kbps",
            "write_bw_kbps",
            "disk_usage_mb",
        )
        assert DISK_INDEX_COLUMNS == (
            "disk_uid",
            "disk_attr",
            "disk_type",
            "vm_cpu",
            "vm_memory",
            "disk_capacity",
        )

    def test_contract_violation_is_an_exception(self):
        """Callers catch ``ContractViolation`` as an ordinary exception type."""
        assert issubclass(ContractViolation, Exception)


class TestFieldOrder:
    def test_obs_fields_exact_order(self):
        """Index position is the only thing tying a policy weight to a physical
        quantity, so any reordering silently scrambles every policy input."""
        assert OBS_FIELDS == EXPECTED_FIELDS, (
            "OBS_FIELDS order/contents changed: a silent reordering scrambles "
            "every policy input (index i no longer means the same quantity)"
        )

    def test_obs_dim_matches_field_count(self):
        """``OBS_DIM`` sizes the policy input layer and must equal the tuple length."""
        assert OBS_DIM == len(OBS_FIELDS) == 12

    def test_obs_fields_is_a_tuple_of_unique_names(self):
        """Duplicate names would make ``OBS_SCALES`` lose an entry."""
        assert isinstance(OBS_FIELDS, tuple)
        assert len(set(OBS_FIELDS)) == len(OBS_FIELDS)

    def test_dataclass_field_order_matches_obs_fields(self):
        """``to_vector`` and ``from_vector`` rely on declaration order; if it
        diverges from ``OBS_FIELDS`` a silent reordering scrambles every policy input."""
        names = tuple(f.name for f in dataclasses.fields(Observation))
        assert names == OBS_FIELDS, (
            "Observation dataclass field order differs from OBS_FIELDS: a silent "
            "reordering scrambles every policy input"
        )

    def test_vector_index_follows_field_order(self):
        """Each field is given a distinct fraction of its scale, so a vector whose
        entry i is not field i's value is detected, not just a wrong total."""
        fractions = [(i + 1) / 20.0 for i in range(OBS_DIM)]
        kwargs = {
            name: OBS_SCALES[name] * frac for name, frac in zip(OBS_FIELDS, fractions)
        }
        obs = Observation(**kwargs)
        np.testing.assert_allclose(
            obs.to_vector(normalized=True), np.array(fractions, dtype=np.float32), rtol=1e-6
        )
        raw = obs.to_vector(normalized=False)
        for i, name in enumerate(OBS_FIELDS):
            assert raw[i] == pytest.approx(kwargs[name], rel=1e-6), name


class TestScales:
    def test_scales_have_exactly_the_observation_fields(self):
        """A missing or extra entry means normalisation is undefined for a field."""
        assert set(OBS_SCALES) == set(OBS_FIELDS)
        assert len(OBS_SCALES) == OBS_DIM

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_scale_value_is_frozen(self, name):
        """Scales are fixed physical bounds, not statistics of any dataset: a
        data-derived scale would change between training and deployment."""
        assert OBS_SCALES[name] == EXPECTED_SCALES[name]

    def test_scales_are_positive_and_finite(self):
        """Division by a zero or non-finite scale would poison the vector."""
        for name, scale in OBS_SCALES.items():
            assert math.isfinite(scale) and scale > 0, name

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_value_at_scale_normalises_to_one(self, name):
        """Normalisation is ``value / scale``; the scale itself maps to exactly 1."""
        obs = make_obs(**{name: OBS_SCALES[name]})
        idx = OBS_FIELDS.index(name)
        assert obs.to_vector(normalized=True)[idx] == pytest.approx(1.0)

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_half_scale_normalises_to_half(self, name):
        """Checks the divisor per field, independent of any other field."""
        obs = make_obs(**{name: OBS_SCALES[name] / 2.0})
        idx = OBS_FIELDS.index(name)
        assert obs.to_vector(normalized=True)[idx] == pytest.approx(0.5, rel=1e-6)


class TestObservationStructure:
    def test_observation_is_frozen(self):
        """Observations are shared between agents; mutation would corrupt another reader."""
        obs = make_obs()
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.progress = 0.9  # type: ignore[misc]

    def test_observation_has_no_defaults(self):
        """A defaulted field would let a producer omit a quantity without error."""
        assert dataclasses.is_dataclass(Observation)
        for f in dataclasses.fields(Observation):
            assert f.default is dataclasses.MISSING, f.name
            assert f.default_factory is dataclasses.MISSING, f.name
        with pytest.raises(TypeError):
            Observation()  # type: ignore[call-arg]

    def test_observation_rejects_missing_field(self):
        """Constructing with 11 of 12 fields must fail at construction time."""
        kwargs = dict(REALISTIC_OBS_KWARGS)
        kwargs.pop("bw_available")
        with pytest.raises(TypeError):
            Observation(**kwargs)

    def test_observation_equality_is_by_value(self):
        """Dataclass value equality is what the exact round-trip tests rely on."""
        assert make_obs() == make_obs()
        assert make_obs() != make_obs(progress=0.5)


class TestVectorProperties:
    @pytest.mark.parametrize("normalized", [True, False])
    def test_dtype_and_shape(self, normalized):
        """Policy networks consume float32 vectors of fixed length 12."""
        vec = make_obs().to_vector(normalized=normalized)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (12,)

    def test_default_is_normalised(self):
        """``normalized=True`` is the documented default."""
        obs = make_obs()
        np.testing.assert_array_equal(obs.to_vector(), obs.to_vector(normalized=True))

    def test_normalised_vector_in_unit_interval(self):
        """A realistic observation stays inside the unit cube after normalisation."""
        vec = make_obs().to_vector(normalized=True)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    @pytest.mark.parametrize("huge", [1e6, 1e30, 1e300])
    def test_normalised_vector_clipped_for_absurd_inputs(self, huge):
        """Out-of-range telemetry must saturate at 1.0 rather than leak into the
        policy input; ``clip(value / scale, 0, 1)`` bounds every element."""
        kwargs = {name: huge for name in OBS_FIELDS}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # float32 overflow is acceptable here
            vec = Observation(**kwargs).to_vector(normalized=True)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
        np.testing.assert_array_equal(vec, np.ones(12, dtype=np.float32))

    def test_negative_value_clips_to_zero_in_normalised_vector(self):
        """The lower bound of ``clip(value / scale, 0, 1)`` applies to the vector
        even though ``validate`` is the component that rejects negatives."""
        vec = make_obs(fg_bw=-50.0).to_vector(normalized=True)
        assert vec[OBS_FIELDS.index("fg_bw")] == 0.0

    def test_unnormalised_vector_is_not_clipped(self):
        """``normalized=False`` carries raw physical values, including those above scale."""
        vec = make_obs(fg_read_iops=50000.0).to_vector(normalized=False)
        assert vec[0] == np.float32(50000.0)

    def test_normalised_vector_saturates_above_scale_and_clips_negative(self):
        """``normalized=True`` bounds both ends: above-scale values become exactly 1.0
        and negatives become exactly 0.0."""
        vec = make_obs(fg_read_iops=50000.0, fg_bw=-50.0).to_vector(normalized=True)
        assert vec[OBS_FIELDS.index("fg_read_iops")] == 1.0
        assert vec[OBS_FIELDS.index("fg_bw")] == 0.0

    def test_unnormalised_vector_passes_negative_through(self):
        """The raw form performs no range handling at all."""
        vec = make_obs(fg_bw=-50.0).to_vector(normalized=False)
        assert vec[OBS_FIELDS.index("fg_bw")] == -50.0

    def test_to_vector_does_not_validate_so_nan_survives_it(self):
        """``to_vector`` neither validates nor sanitises: a NaN passes through the raw
        form and, because ``np.clip`` propagates NaN, through the normalised form
        too.  Only ``validate()`` catches it, so a producer that skips ``validate()``
        feeds NaN straight into the policy."""
        obs = make_obs(latency_now=float("nan"))
        idx = OBS_FIELDS.index("latency_now")
        assert np.isnan(obs.to_vector(normalized=False)[idx])
        assert np.isnan(obs.to_vector(normalized=True)[idx])
        with pytest.raises(ContractViolation):
            obs.validate()


class TestRoundTrip:
    def test_raw_round_trip_is_exact(self):
        """Raw units pass through float32 unchanged when the values are float32-exact."""
        obs = Observation(**EXACT_OBS_KWARGS)
        back = Observation.from_vector(obs.to_vector(normalized=False), normalized=False)
        assert back == obs

    def test_raw_round_trip_realistic_values_within_float32(self):
        """Decimal values such as 5e-5 are not float32-representable; the round-trip
        error must stay at float32 precision (relative ~6e-8)."""
        obs = make_obs()
        back = Observation.from_vector(obs.to_vector(normalized=False), normalized=False)
        for name in OBS_FIELDS:
            assert getattr(back, name) == pytest.approx(getattr(obs, name), rel=1e-6), name

    def test_normalised_round_trip_below_scale(self):
        """For values under their scale, ``value / scale * scale`` recovers the value
        up to float32 rounding."""
        obs = make_obs()
        for name in OBS_FIELDS:
            assert getattr(obs, name) < OBS_SCALES[name], name
        back = Observation.from_vector(obs.to_vector(normalized=True), normalized=True)
        for name in OBS_FIELDS:
            assert getattr(back, name) == pytest.approx(getattr(obs, name), rel=1e-6), name

    def test_normalised_round_trip_exact_values(self):
        """With float32-exact inputs the normalised round-trip should agree to
        within about one float32 ulp."""
        obs = Observation(**EXACT_OBS_KWARGS)
        back = Observation.from_vector(obs.to_vector(normalized=True), normalized=True)
        for name in OBS_FIELDS:
            assert getattr(back, name) == pytest.approx(getattr(obs, name), rel=2e-7), name

    def test_value_above_scale_is_clipped_and_does_not_round_trip(self):
        """Clipping is intended: the normalised vector deliberately discards how far
        a value exceeds its scale, so the inverse returns the scale, not the input.
        A policy therefore sees a saturated 1.0 for any overload."""
        obs = make_obs(fg_read_iops=50000.0)  # 2.5x the 20000 scale
        vec = obs.to_vector(normalized=True)
        assert vec[0] == 1.0
        back = Observation.from_vector(vec, normalized=True)
        assert back.fg_read_iops == pytest.approx(OBS_SCALES["fg_read_iops"])
        assert back.fg_read_iops != pytest.approx(50000.0)
        assert back != obs

    def test_from_vector_returns_observation(self):
        """The inverse yields the frozen dataclass, not a dict or array."""
        back = Observation.from_vector(np.zeros(12, dtype=np.float32), normalized=True)
        assert isinstance(back, Observation)

    def test_from_vector_accepts_plain_list(self):
        """Callers may hold the vector as a Python sequence (e.g. from JSON)."""
        back = Observation.from_vector([0.5] * 12, normalized=True)
        assert back.fg_bw == pytest.approx(0.5 * OBS_SCALES["fg_bw"])
        assert back.progress == pytest.approx(0.5)

    def test_from_vector_default_is_normalised(self):
        """``normalized=True`` is the default for the inverse as well."""
        vec = np.full(12, 0.5, dtype=np.float32)
        assert Observation.from_vector(vec) == Observation.from_vector(vec, normalized=True)

    @pytest.mark.parametrize("length", [0, 11, 13, 24])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_from_vector_wrong_length_raises(self, length, normalized):
        """A vector of the wrong length means producer and consumer disagree on the
        layout; it must fail loudly rather than truncate or pad."""
        with pytest.raises(ContractViolation):
            Observation.from_vector(np.zeros(length, dtype=np.float32), normalized=normalized)


class TestObservationValidation:
    def test_valid_observation_passes_silently(self):
        """``validate`` returns None and raises nothing for an in-range observation."""
        assert make_obs().validate() is None
        assert Observation(**EXACT_OBS_KWARGS).validate() is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(progress=0.0),
            dict(progress=1.0),
            dict(redundancy_left=0.0),
            dict(redundancy_left=2.0),
            dict(latency_target=1e-9),
            dict(fg_read_iops=0.0, fg_write_iops=0.0, fg_bw=0.0),
            dict(fg_read_iops=1e9),  # above scale is legal; the vector clips it
        ],
    )
    def test_boundary_values_are_valid(self, kwargs):
        """Closed interval endpoints are legal; only values outside them are breaches."""
        make_obs(**kwargs).validate()

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_value_in_any_field_raises(self, name, bad):
        """NaN or inf in any position poisons downstream arithmetic."""
        with pytest.raises(ContractViolation):
            make_obs(**{name: bad}).validate()

    @pytest.mark.parametrize(
        "name",
        [
            "fg_read_iops",
            "fg_write_iops",
            "fg_bw",
            "latency_now",
            "hazard",
            "hazard_uncertainty",
            "work_remaining",
            "alloc_current",
            "bw_available",
        ],
    )
    def test_negative_value_raises(self, name):
        """Rates, latencies, hazards, work and bandwidth are non-negative quantities."""
        with pytest.raises(ContractViolation):
            make_obs(**{name: -1e-3}).validate()

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_progress_outside_unit_interval_raises(self, bad):
        """Progress is a completed fraction and must lie in [0, 1]."""
        with pytest.raises(ContractViolation):
            make_obs(progress=bad).validate()

    @pytest.mark.parametrize("bad", [-1.0, 3.0])
    def test_redundancy_left_outside_range_raises(self, bad):
        """The group tolerates two failures, so redundancy_left is confined to [0, 2]."""
        with pytest.raises(ContractViolation):
            make_obs(redundancy_left=bad).validate()

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_latency_target_raises(self, bad):
        """The latency target is a divisor in the degradation metric and must be > 0."""
        with pytest.raises(ContractViolation):
            make_obs(latency_target=bad).validate()


class TestActions:
    @pytest.mark.parametrize(
        "index, expected", [(0, 20.0), (1, 50.0), (2, 80.0), (3, 120.0)]
    )
    def test_valid_indices_map_to_mbps(self, index, expected):
        """The discrete action set is the only way an agent expresses an allocation."""
        out = action_to_mbps(index)
        assert out == expected
        assert isinstance(out, float)

    def test_mapping_matches_constant(self):
        """``ACTIONS_MBPS`` and ``action_to_mbps`` must describe the same table."""
        assert [action_to_mbps(i) for i in range(N_ACTIONS)] == [float(a) for a in ACTIONS_MBPS]

    @pytest.mark.parametrize("bad", [-1, 4, 100, 0.0, 1.0, True, False, "0", None])
    def test_invalid_actions_raise(self, bad):
        """Out-of-range indices and non-int types are breaches.  ``True`` is a
        subclass of ``int`` in Python and would otherwise be accepted as action 1."""
        with pytest.raises(ContractViolation):
            action_to_mbps(bad)


class TestInfoValidation:
    def test_info_keys_frozen(self):
        """The key set and order are the logging schema evaluation consumes."""
        assert INFO_KEYS == (
            "latency_p95",
            "latency_p99",
            "violated_latency",
            "p_failure_before_complete",
            "violated_risk_budget",
            "alloc_clipped_by_filter",
            "recovery_complete",
        )

    def test_complete_valid_info_passes(self):
        """A fully populated, correctly typed dict is accepted and returns None."""
        assert validate_info(make_info()) is None

    @pytest.mark.parametrize("key", INFO_KEYS)
    def test_missing_key_raises(self, key):
        """Each of the seven keys is mandatory; evaluation indexes them unguarded."""
        info = make_info()
        del info[key]
        with pytest.raises(ContractViolation):
            validate_info(info)

    @pytest.mark.parametrize("key", BOOL_KEYS)
    @pytest.mark.parametrize(
        "bad", [np.bool_(True), np.bool_(False), 1, 0, "yes"], ids=repr
    )
    def test_boolean_slots_require_real_bool(self, key, bad):
        """Constraint accounting sums these flags; ``np.bool_``, ints and truthy
        strings behave differently under serialisation and ``is True`` checks, so
        only the genuine ``bool`` singletons are accepted."""
        with pytest.raises(ContractViolation):
            validate_info(make_info(**{key: bad}))

    @pytest.mark.parametrize("key", BOOL_KEYS)
    @pytest.mark.parametrize("good", [True, False])
    def test_boolean_slots_accept_real_bool(self, key, good):
        """The strictness above must not reject the legitimate values."""
        validate_info(make_info(**{key: good}))

    @pytest.mark.parametrize("key", NUMERIC_KEYS)
    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
    def test_numeric_slots_reject_negative_and_non_finite(self, key, bad):
        """Latencies and probabilities are finite and non-negative."""
        with pytest.raises(ContractViolation):
            validate_info(make_info(**{key: bad}))

    def test_probability_above_one_raises(self):
        """A probability above 1 is meaningless and indicates a units error."""
        with pytest.raises(ContractViolation):
            validate_info(make_info(p_failure_before_complete=1.5))

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_probability_endpoints_are_valid(self, p):
        """The probability interval is closed at both ends."""
        validate_info(make_info(p_failure_before_complete=p))

    def test_extra_diagnostic_keys_are_permitted(self):
        """Implementations may log additional diagnostics; the contract only
        constrains the seven mandatory keys."""
        info = make_info(queue_depth=7, note="free text", arr=np.zeros(3), nested={"a": 1})
        validate_info(info)

    def test_termination_reason_absent_is_valid(self):
        """Mid-episode steps carry no reason."""
        assert "termination_reason" not in make_info()
        validate_info(make_info())

    def test_termination_reason_none_is_valid(self):
        """An explicit None is equivalent to absence mid-episode."""
        validate_info(make_info(termination_reason=None))

    @pytest.mark.parametrize("reason", list(TerminationReason), ids=lambda r: r.name)
    def test_termination_reason_value_string_is_valid(self, reason):
        """The ``.value`` string is the serialisable form carried in ``info``."""
        validate_info(make_info(termination_reason=reason.value))

    @pytest.mark.parametrize("reason", list(TerminationReason), ids=lambda r: r.name)
    def test_termination_reason_enum_member_is_rejected(self, reason):
        """The member itself is not JSON-serialisable and would make logged info
        differ between in-process and persisted forms."""
        with pytest.raises(ContractViolation):
            validate_info(make_info(termination_reason=reason))

    @pytest.mark.parametrize(
        "bad",
        ["banana", "RECOVERY_COMPLETE", "", 0, 1, 1.5, True, b"recovery_complete", ["recovery_complete"]],
    )
    def test_termination_reason_arbitrary_values_are_rejected(self, bad):
        """Only the exact ``.value`` strings identify a cause; names, other strings and
        non-strings must not slip through."""
        with pytest.raises(ContractViolation):
            validate_info(make_info(termination_reason=bad))

    def test_latencies_above_one_are_valid(self):
        """Only the probability is bounded above; latencies are in milliseconds."""
        validate_info(make_info(latency_p95=500.0, latency_p99=2000.0))


class TestTermination:
    def test_members_exist(self):
        """The three terminal causes are the complete episode-ending taxonomy."""
        assert issubclass(TerminationReason, enum.Enum)
        names = {m.name for m in TerminationReason}
        assert {"RECOVERY_COMPLETE", "TERMINAL_FAILURE", "HORIZON_EXPIRED"} <= names

    def test_members_are_distinct(self):
        """Evaluation reports outcome rates per cause; aliased members would merge them."""
        members = [
            TerminationReason.RECOVERY_COMPLETE,
            TerminationReason.TERMINAL_FAILURE,
            TerminationReason.HORIZON_EXPIRED,
        ]
        assert len(set(members)) == 3
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                assert a is not b
                assert a != b

    def test_values_are_distinguishable(self):
        """Values are what appears in logged info dicts, so they must differ even
        after serialisation."""
        values = [m.value for m in TerminationReason]
        assert len(set(values)) == len(values)
        assert len({str(v) for v in values}) == len(values)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("RECOVERY_COMPLETE", (True, False)),
            ("TERMINAL_FAILURE", (True, False)),
            ("HORIZON_EXPIRED", (False, True)),
        ],
    )
    def test_termination_flags_mapping(self, name, expected):
        """Fixes the Gymnasium ``(terminated, truncated)`` pair for each cause."""
        out = termination_flags(TerminationReason[name])
        assert out == expected
        assert type(out[0]) is bool and type(out[1]) is bool

    def test_only_horizon_expiry_truncates(self):
        """The horizon is an artificial cutoff, not an absorbing state.  Flagging it
        as ``terminated`` makes a value-based learner bootstrap V(next state) = 0 at
        the cutoff, which under-values states leading to long rebuilds and biases the
        agent against slow allocations (value-bootstrapping bias)."""
        for reason in TerminationReason:
            terminated, truncated = termination_flags(reason)
            if reason is TerminationReason.HORIZON_EXPIRED:
                assert truncated and not terminated, (
                    "HORIZON_EXPIRED must be truncated, not terminated: terminating at "
                    "the horizon makes the learner bootstrap V(next)=0 and biases it "
                    "against slow allocations (value-bootstrapping bias)"
                )
            else:
                assert terminated and not truncated, (
                    f"{reason.name} must be terminated and only HORIZON_EXPIRED may "
                    "set truncated, otherwise value bootstrapping is biased"
                )
        assert [r for r in TerminationReason if termination_flags(r)[1]] == [
            TerminationReason.HORIZON_EXPIRED
        ]

    @pytest.mark.parametrize(
        "bad",
        [
            "recovery_complete",
            "horizon_expired",
            TerminationReason.HORIZON_EXPIRED.value,
            "RECOVERY_COMPLETE",
            None,
            0,
            1,
            True,
            (True, False),
        ],
    )
    def test_termination_flags_rejects_non_members(self, bad):
        """Passing the logged ``.value`` string instead of the member is a likely
        mistake; accepting it would let a typo choose the flags silently."""
        with pytest.raises(ContractViolation):
            termination_flags(bad)

    def test_value_lookup_round_trips(self):
        """A logged value must map back to exactly one member."""
        for m in TerminationReason:
            assert TerminationReason(m.value) is m


    def test_failure_probability_is_not_a_termination_reason(self):
        """``p_failure_before_complete`` is P(at least one further failure before the
        rebuild completes); ``TERMINAL_FAILURE`` is data loss (redundancy exhausted
        and another failure occurs).  They are different quantities: the probability
        key does not name a termination cause, no cause value reuses the key, and a
        probability of 1.0 is a valid info value that implies no termination."""
        assert "p_failure_before_complete" in INFO_KEYS
        causes = {m.name for m in TerminationReason} | {str(m.value) for m in TerminationReason}
        assert "p_failure_before_complete" not in causes
        assert all("termination_reason" != k for k in INFO_KEYS)
        validate_info(make_info(p_failure_before_complete=1.0, recovery_complete=False))


class TestReward:
    def test_default_coefficients(self):
        """The coefficient defaults encode the agreed priority of the three terms."""
        cfg = RewardConfig()
        assert (cfg.alpha, cfg.beta, cfg.gamma) == (1.0, 0.1, 10.0)

    def test_config_is_frozen(self):
        """A shared config mutated mid-run would silently change the objective."""
        cfg = RewardConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.alpha = 5.0  # type: ignore[misc]

    def test_zero_inputs_give_zero_reward(self):
        """With no degradation, cost or violation the controller is not penalised."""
        assert compute_reward(0, 0, 0) == 0.0
        assert compute_reward(0.0, 0.0, 0.0) == 0.0

    @pytest.mark.parametrize("d", [0.0, 0.5, 3.0])
    @pytest.mark.parametrize("c", [0.0, 1.0, 10.0])
    @pytest.mark.parametrize("v", [0.0, 1.0, 5.0])
    def test_reward_never_positive_for_non_negative_terms(self, d, c, v):
        """Every term is a penalty; a positive reward would reward violations."""
        assert compute_reward(d, c, v) <= 0.0

    def test_matches_formula(self):
        """Reward is exactly ``-(alpha*D + beta*C + gamma*V)``."""
        assert compute_reward(1.0, 2.0, 3.0) == pytest.approx(-(1.0 * 1.0 + 0.1 * 2.0 + 10.0 * 3.0))

    def test_custom_config_is_used(self):
        """An explicit config overrides the default coefficients."""
        cfg = RewardConfig(alpha=2.0, beta=0.0, gamma=0.5)
        assert compute_reward(1.0, 100.0, 4.0, cfg) == pytest.approx(-(2.0 * 1.0 + 0.5 * 4.0))

    def test_gamma_term_dominates_alpha_term(self):
        """With default coefficients a unit violation costs more than a unit of
        latency degradation, which is what makes violating the constraint unattractive."""
        assert compute_reward(0.0, 0.0, 1.0) < compute_reward(1.0, 0.0, 0.0)

    def test_reward_is_monotone_in_each_term(self):
        """More degradation, cost or violation never improves the reward."""
        base = compute_reward(1.0, 1.0, 1.0)
        assert compute_reward(2.0, 1.0, 1.0) < base
        assert compute_reward(1.0, 2.0, 1.0) < base
        assert compute_reward(1.0, 1.0, 2.0) < base

    def test_from_dict_full(self):
        """A complete mapping builds the corresponding config."""
        cfg = RewardConfig.from_dict({"alpha": 2.0, "beta": 0.5, "gamma": 20.0})
        assert (cfg.alpha, cfg.beta, cfg.gamma) == (2.0, 0.5, 20.0)

    def test_from_dict_partial_keeps_defaults(self):
        """Omitted keys fall back to the dataclass defaults."""
        cfg = RewardConfig.from_dict({"alpha": 2.0})
        assert (cfg.alpha, cfg.beta, cfg.gamma) == (2.0, 0.1, 10.0)

    def test_from_dict_unknown_key_raises(self):
        """A misspelled coefficient in a config file must not be silently ignored."""
        with pytest.raises(ContractViolation):
            RewardConfig.from_dict({"delta": 1.0})

    def test_from_dict_unknown_key_alongside_valid_raises(self):
        """One bad key invalidates the whole mapping."""
        with pytest.raises(ContractViolation):
            RewardConfig.from_dict({"alpha": 2.0, "delta": 1.0})


class TestLatencyDegradation:
    @pytest.mark.parametrize(
        "now, target, expected",
        [(10, 10, 0.0), (15, 10, 0.5), (5, 10, 0.0), (30, 10, 2.0), (0, 10, 0.0)],
    )
    def test_cases(self, now, target, expected):
        """Degradation is the relative excess over target, floored at zero so that
        being faster than target earns no credit."""
        assert latency_degradation(now, target) == pytest.approx(expected)

    @pytest.mark.parametrize("target", [0, 0.0, -1, -10.0])
    def test_non_positive_target_raises(self, target):
        """The target is the divisor; zero or negative has no meaning."""
        with pytest.raises(ContractViolation):
            latency_degradation(10, target)

    def test_result_is_never_negative(self):
        """Non-negativity keeps the reward penalty one-sided."""
        for now in [0.0, 1.0, 9.99, 10.0, 50.0]:
            assert latency_degradation(now, 10.0) >= 0.0


class TestDegradedCost:
    def test_values(self):
        """The cost counts one per degraded control interval, so summing it over an
        episode yields the number of degraded intervals."""
        assert degraded_cost(True) == 1.0
        assert degraded_cost(False) == 0.0
        assert isinstance(degraded_cost(True), float)
        assert isinstance(degraded_cost(False), float)

    def test_sum_counts_degraded_intervals(self):
        """Additivity over an episode is the defining use of this term."""
        flags = [True, False, True, True, False]
        assert sum(degraded_cost(f) for f in flags) == 3.0

    @pytest.mark.parametrize("flag, expected", [(np.bool_(True), 1.0), (np.bool_(False), 0.0)])
    def test_numpy_bool_accepted(self, flag, expected):
        """Simulator state is often held in numpy arrays, so a numpy boolean flag is a
        legitimate input to the computation and must give the same cost as ``bool``."""
        out = degraded_cost(flag)
        assert out == expected
        assert isinstance(out, float)

    @pytest.mark.parametrize("bad", [1, 0, 1.0, "yes", None])
    def test_non_bool_rejected(self, bad):
        """Ints, floats, strings and None convert to a count silently and would
        corrupt the cost accounting; only boolean types are accepted."""
        with pytest.raises(ContractViolation):
            degraded_cost(bad)

    def test_strict_at_recording_boundary_permissive_inside_computation(self):
        """Deliberate asymmetry: ``validate_info`` guards what is *recorded* and
        demands a real ``bool`` (a numpy bool serialises differently and breaks
        ``is True`` checks), while ``degraded_cost`` is *computation* and accepts
        ``numpy.bool_`` as well.  The two are not inconsistent; they sit on
        different sides of the recording boundary."""
        flag = np.bool_(True)
        assert degraded_cost(flag) == 1.0
        with pytest.raises(ContractViolation):
            validate_info(make_info(violated_latency=flag))


class TestLatencyViolation:
    @pytest.mark.parametrize(
        "now, expected", [(14.9, 0.0), (15.0, 0.0), (15.1, 1.0), (0.0, 0.0), (100.0, 1.0)]
    )
    def test_cases_with_ten_ms_target(self, now, expected):
        """With the 1.5 limit factor and a 10 ms target the limit is 15 ms."""
        assert LATENCY_LIMIT_FACTOR == 1.5
        out = latency_violation(now, 10.0)
        assert out == expected
        assert isinstance(out, float)

    def test_boundary_is_not_a_violation(self):
        """The comparison is strictly greater-than: a latency exactly at
        LATENCY_LIMIT_FACTOR * target is within the limit.  Using >= would count
        every at-limit sample as a violation and shift the 5% violation budget."""
        for target in (10.0, 20.0, 50.0):
            limit = LATENCY_LIMIT_FACTOR * target
            assert latency_violation(limit, target) == 0.0
            assert latency_violation(limit * 1.0001, target) == 1.0

    def test_agrees_with_degradation_away_from_boundary(self):
        """A violation is a degradation above LATENCY_LIMIT_FACTOR - 1."""
        for now in (11.0, 14.0, 16.0, 40.0):
            expected = 1.0 if latency_degradation(now, 10.0) > LATENCY_LIMIT_FACTOR - 1 else 0.0
            assert latency_violation(now, 10.0) == expected

    @pytest.mark.parametrize("target", [0, 0.0, -1, -10.0])
    def test_non_positive_target_raises(self, target):
        """The target scales the limit; zero or negative has no meaning."""
        with pytest.raises(ContractViolation):
            latency_violation(10.0, target)


_NON_FINITE = [float("nan"), float("inf"), float("-inf")]


class TestNonFiniteInputsDoNotBecomeBenign:
    """A non-finite input must raise, or stay visibly non-finite; it must never pass
    through a comparison, ``max`` or ``min`` and emerge as an ordinary number."""

    @pytest.mark.parametrize("fn", [latency_degradation, latency_violation], ids=lambda f: f.__name__)
    @pytest.mark.parametrize("field", ["latency_now", "latency_target"])
    @pytest.mark.parametrize("bad", _NON_FINITE, ids=repr)
    def test_non_finite_latency_raises(self, fn, field, bad):
        """``nan > 0`` is False, so ``max(0, nan)`` style arithmetic silently reports
        zero degradation: a broken measurement would read as a perfectly healthy
        interval and the agent would be rewarded for a state nobody observed."""
        args = {"latency_now": 15.0, "latency_target": 10.0}
        args[field] = bad
        try:
            out = fn(**args)
        except ContractViolation:
            return
        pytest.fail(
            f"{fn.__name__}({field}={bad!r}) returned {out!r} instead of raising: a "
            "NaN or infinite latency silently reporting a finite degradation or "
            "violation means a broken measurement reads as a perfectly healthy "
            "interval, so the agent is rewarded for a state nobody observed"
        )

    @pytest.mark.parametrize("fn", [latency_degradation, latency_violation], ids=lambda f: f.__name__)
    def test_finite_latencies_are_still_accepted(self, fn):
        """The finiteness guard must not reject ordinary measurements, including zero latency."""
        fn(0.0, 10.0)
        fn(15.0, 10.0)
        fn(1e6, 10.0)

    @pytest.mark.parametrize("position", [0, 1, 2], ids=["d_latency", "c_degraded", "v_violation"])
    @pytest.mark.parametrize("bad", _NON_FINITE, ids=repr)
    def test_reward_rejects_non_finite_terms(self, position, bad):
        """A NaN term would turn the reward into NaN and poison every return and
        gradient computed from it; an infinite term would dominate it."""
        args = [0.5, 1.0, 0.0]
        args[position] = bad
        with pytest.raises(ContractViolation):
            compute_reward(*args)

    def test_non_finite_reward_coefficient_never_gives_a_benign_reward(self):
        """``RewardConfig`` accepts any float, so a NaN coefficient must at least stay
        visible: the reward is NaN (or the call raises), never an ordinary number that
        hides the misconfiguration."""
        cfg = RewardConfig.from_dict({"alpha": float("nan")})
        try:
            out = compute_reward(0.5, 1.0, 0.0, cfg)
        except ContractViolation:
            return
        assert not math.isfinite(out), (
            f"reward {out!r} from a NaN coefficient looks like an ordinary number"
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=repr)
    def test_kbps_to_mbps_does_not_hide_non_finite_input(self, bad):
        """A NaN or infinite bandwidth sample must raise or stay non-finite; mapping it
        to a finite MB/s figure would turn a missing measurement into a real load."""
        try:
            out = kbps_to_mbps(bad)
        except ContractViolation:
            return
        assert not math.isfinite(out), f"kbps_to_mbps({bad!r}) returned finite {out!r}"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), np.nan], ids=repr)
    def test_action_index_rejects_non_finite(self, bad):
        """A float action index is invalid whatever its value; NaN must not slip through
        a range comparison that evaluates False."""
        with pytest.raises(ContractViolation):
            action_to_mbps(bad)

    @pytest.mark.parametrize(
        "grid",
        [[0, float("nan")], [0, 300, float("nan")], [0, float("inf")], [0.0, 300.0]],
        ids=repr,
    )
    def test_replay_grid_rejects_non_integer_timestamps(self, grid):
        """``nan != 300`` and ``nan < 0`` are both False/True in surprising ways; the grid
        is integral seconds, so floats (and therefore NaN and inf) are rejected."""
        with pytest.raises(ContractViolation):
            validate_replay_grid(grid)

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_validate_is_the_gate_for_nan_and_inf_in_every_field(self, name):
        """``to_vector`` neither validates nor sanitises, so ``validate()`` is the only
        gate; a NaN or infinity in any field must be caught there."""
        for bad in _NON_FINITE:
            with pytest.raises(ContractViolation):
                make_obs(**{name: bad}).validate()

    def test_to_vector_nan_pass_through_is_deliberate(self):
        """Documented behaviour, asserted so it stays that way: NaN survives both
        forms of ``to_vector`` (``validate()`` is the gate), while the normalised form
        saturates +inf at 1.0 and -inf at 0.0 like any out-of-range value."""
        idx = OBS_FIELDS.index("fg_bw")
        nan_obs = make_obs(fg_bw=float("nan"))
        assert np.isnan(nan_obs.to_vector(normalized=True)[idx])
        assert np.isnan(nan_obs.to_vector(normalized=False)[idx])
        assert make_obs(fg_bw=float("inf")).to_vector(normalized=True)[idx] == 1.0
        assert make_obs(fg_bw=float("-inf")).to_vector(normalized=True)[idx] == 0.0
        with pytest.raises(ContractViolation):
            nan_obs.validate()
        with pytest.raises(ContractViolation):
            make_obs(fg_bw=float("inf")).validate()

    def test_from_vector_nan_is_caught_by_validate(self):
        """Whether ``from_vector`` raises or passes a NaN vector through, the resulting
        observation must not validate."""
        vec = np.full(OBS_DIM, 0.5, dtype=np.float32)
        vec[OBS_FIELDS.index("latency_now")] = np.nan
        try:
            obs = Observation.from_vector(vec)
        except ContractViolation:
            return
        with pytest.raises(ContractViolation):
            obs.validate()


class TestHazardConversion:
    def test_zero_probability_gives_exactly_zero(self):
        """No horizon risk must map to exactly zero per-hour hazard, not a tiny residual."""
        assert hazard_p_to_per_hour(0.0) == 0.0
        assert hazard_p_to_per_hour(0) == 0.0

    def test_small_p_is_close_to_p_over_horizon(self):
        """Write q = 1-(1-p)^(1/h).  Since (1-p)^(1/h) = exp(ln(1-p)/h) and
        ln(1-p) = -(p + p^2/2 + ...), we get q = x - x^2/2 + ... with
        x = (p + p^2/2 + ...)/h, hence q = (p/h)(1 + p/2 + O(p/h) + O(p^2)).
        For p = 1e-4 and h = 720 the relative deviation from p/h is about
        p/2 = 5e-5, so rel=2e-4 leaves a 4x margin."""
        p = 1e-4
        assert hazard_p_to_per_hour(p) == pytest.approx(p / 720.0, rel=2e-4)

    def test_default_horizon_is_horizon_hours(self):
        """The default argument must be the contract horizon, not a private copy."""
        assert hazard_p_to_per_hour(0.05) == hazard_p_to_per_hour(0.05, HORIZON_HOURS)

    def test_matches_closed_form(self):
        """Direct evaluation of the specified expression."""
        for p, h in [(0.01, 720.0), (0.3, 720.0), (0.5, 100.0)]:
            assert hazard_p_to_per_hour(p, h) == pytest.approx(1 - (1 - p) ** (1 / h), rel=1e-9)

    def test_strictly_increasing_in_p(self):
        """A riskier horizon must never map to a lower or equal hourly hazard."""
        ps = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.999]
        qs = [hazard_p_to_per_hour(p) for p in ps]
        assert all(b > a for a, b in zip(qs, qs[1:])), qs

    def test_result_in_unit_interval(self):
        """A per-hour probability lies in [0, 1)."""
        for p in [0.0, 1e-4, 0.5, 0.999]:
            assert 0.0 <= hazard_p_to_per_hour(p) < 1.0

    def test_compounding_over_horizon_recovers_p(self):
        """Survival over h independent hours is (1-q)^h, so 1-(1-q)^h must return the
        original horizon probability; this is the defining property of the conversion."""
        for p in [1e-4, 1e-2, 0.3, 0.9]:
            q = hazard_p_to_per_hour(p)
            assert 1 - (1 - q) ** 720 == pytest.approx(p, rel=1e-7)

    def test_compounding_with_custom_horizon(self):
        """The same property holds for a non-default horizon."""
        for h in [24.0, 100.0, 2000.0]:
            q = hazard_p_to_per_hour(0.2, h)
            assert 1 - (1 - q) ** h == pytest.approx(0.2, rel=1e-7)

    @pytest.mark.parametrize("p", [1.0, -0.1, 1.5, float("nan")])
    def test_invalid_probability_raises(self, p):
        """p = 1 has no finite hazard; negative, above-one and NaN are not probabilities."""
        with pytest.raises(ContractViolation):
            hazard_p_to_per_hour(p)

    @pytest.mark.parametrize("h", [0, 0.0, -1.0, -720.0])
    def test_non_positive_horizon_raises(self, h):
        """The horizon is an exponent divisor and must be positive."""
        with pytest.raises(ContractViolation):
            hazard_p_to_per_hour(0.01, h)


def _delta(sigma, p, h=HORIZON_HOURS):
    """Reference delta-method value, written out independently of the module."""
    return sigma * (1.0 / h) * (1.0 - p) ** (1.0 / h - 1.0)


class TestHazardUncertaintyConversion:
    def test_matches_formula(self):
        """Hand-computed cases of ``sigma * (1/H) * (1-p)**(1/H - 1)``.  With H = 1 the
        map is the identity (h = p), so the derivative is 1 and sigma is unchanged;
        with H = 2, p = 0.75 gives h = 0.5 and derivative (1/2) * 0.25**-0.5 = 1;
        with p = 0.36 the derivative is 0.5 / 0.8 = 0.625."""
        assert hazard_uncertainty_to_per_hour(0.1, 0.3, 1.0) == pytest.approx(0.1)
        assert hazard_uncertainty_to_per_hour(0.1, 0.75, 2.0) == pytest.approx(0.1)
        assert hazard_uncertainty_to_per_hour(0.08, 0.36, 2.0) == pytest.approx(0.08 * 0.625)
        for sigma, p in [(0.01, 0.05), (0.1, 0.5), (0.3, 0.9)]:
            assert hazard_uncertainty_to_per_hour(sigma, p) == pytest.approx(_delta(sigma, p), rel=1e-12)

    def test_default_horizon_is_horizon_hours(self):
        """The default must be the contract horizon, as for the probability map."""
        assert hazard_uncertainty_to_per_hour(0.1, 0.4) == hazard_uncertainty_to_per_hour(
            0.1, 0.4, HORIZON_HOURS
        )

    def test_derivative_agrees_with_finite_difference_of_probability_map(self):
        """The delta method needs d/dp of ``hazard_p_to_per_hour``; a central finite
        difference of the module's own probability map at p = 0.5 must reproduce
        the analytic derivative, tying the two functions to the same map."""
        p, eps = 0.5, 1e-5
        numeric = (hazard_p_to_per_hour(p + eps) - hazard_p_to_per_hour(p - eps)) / (2 * eps)
        assert hazard_uncertainty_to_per_hour(0.1, p) == pytest.approx(0.1 * numeric, rel=1e-6)

    def test_zero_sigma_gives_exactly_zero(self):
        """No uncertainty in, none out, regardless of p."""
        for p in (0.0, 0.3, 0.99):
            assert hazard_uncertainty_to_per_hour(0.0, p) == 0.0

    def test_small_p_is_close_to_sigma_over_horizon(self):
        """h(p) = 1 - (1-p)^(1/H) has derivative (1/H)(1-p)^(1/H - 1).  As p -> 0 the
        factor (1-p)^(1/H - 1) -> 1, so the derivative tends to 1/H and
        sigma_h -> sigma/H.  The factor equals 1 + p(1 - 1/H) + O(p^2), so for
        p = 1e-4 the relative deviation is about 1e-4 and rel=5e-4 leaves margin."""
        sigma = 0.01
        assert hazard_uncertainty_to_per_hour(sigma, 1e-4) == pytest.approx(sigma / 720.0, rel=5e-4)
        assert hazard_uncertainty_to_per_hour(sigma, 0.0) == pytest.approx(sigma / 720.0, rel=1e-12)

    def test_sigma_upper_bound_is_inclusive(self):
        """A standard deviation of a quantity in [0, 1] cannot exceed 0.5; 0.5 itself is allowed."""
        assert hazard_uncertainty_to_per_hour(0.5, 0.1) > 0.0

    @pytest.mark.parametrize("bad", [0.6, 1.0, -0.1, -1e-12, float("nan"), float("inf"), float("-inf")])
    def test_sigma_outside_range_raises(self, bad):
        """Larger sigma values indicate a broken or mis-scaled risk model."""
        with pytest.raises(ContractViolation):
            hazard_uncertainty_to_per_hour(bad, 0.1)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan"), float("inf")])
    def test_p_outside_range_raises(self, bad):
        """The derivative is evaluated at p, which must be a valid probability below 1."""
        with pytest.raises(ContractViolation):
            hazard_uncertainty_to_per_hour(0.1, bad)

    @pytest.mark.parametrize("h", [0, 0.0, -1.0, -720.0])
    def test_non_positive_horizon_raises(self, h):
        """The horizon divides the derivative and must be positive."""
        with pytest.raises(ContractViolation):
            hazard_uncertainty_to_per_hour(0.1, 0.1, h)

    def test_uncertainty_is_not_converted_with_the_probability_map(self):
        """Regression lock: a standard deviation has been pushed through a probability
        map if the result equals ``hazard_p_to_per_hour(sigma)``.  The probability
        map converts a probability over H to a probability over one hour; a
        dispersion does not convert that way.  At p = 0.5, sigma = 0.1 the correct
        delta-method value is about twice the (wrong) probability-map value."""
        sigma, p = 0.1, 0.5
        got = hazard_uncertainty_to_per_hour(sigma, p)
        wrong = hazard_p_to_per_hour(sigma)
        assert got == pytest.approx(_delta(sigma, p), rel=1e-12)
        assert got != pytest.approx(wrong, rel=0.2), (
            "uncertainty matches hazard_p_to_per_hour(sigma): a standard deviation has "
            "been pushed through a probability map instead of the delta method"
        )
        assert got / wrong == pytest.approx(1.9, abs=0.1)


class TestConversionInputValidation:
    """Non-finite horizons and non-real arguments to the two conversion functions."""

    CONVERSIONS = [
        pytest.param(lambda h: hazard_p_to_per_hour(0.1, h), id="hazard_p_to_per_hour"),
        pytest.param(lambda h: hazard_uncertainty_to_per_hour(0.1, 0.1, h), id="hazard_uncertainty_to_per_hour"),
    ]

    @pytest.mark.parametrize("convert", CONVERSIONS)
    @pytest.mark.parametrize("h", [float("inf"), float("-inf"), float("nan")], ids=repr)
    def test_non_finite_horizon_is_rejected(self, convert, h):
        """An infinite horizon is dangerous, not merely wrong: ``1/H`` becomes 0, the
        exponent in ``(1-p)**(1/H)`` becomes 0, and the hazard is exactly zero.  A
        broken horizon input would then read as a perfectly safe group and nothing
        would raise."""
        try:
            out = convert(h)
        except ContractViolation:
            return
        pytest.fail(
            f"horizon_hours={h!r} returned {out!r}: an infinite horizon zeroes the 1/H "
            "exponent and yields a hazard of exactly 0, so a broken input reads as a "
            "perfectly safe group with nothing raising"
        )

    @pytest.mark.parametrize("convert", CONVERSIONS)
    @pytest.mark.parametrize("bad", [None, True, False, np.bool_(True), "720"], ids=repr)
    def test_non_real_horizon_raises_contract_violation_not_type_error(self, convert, bad):
        """``None`` or a bool as the horizon must surface as ``ContractViolation``; a
        leaked ``TypeError`` would bypass callers that catch the named exception."""
        _expect_contract_violation(convert, bad, why="non-real horizon_hours")

    @pytest.mark.parametrize("bad", [None, True, False, np.bool_(True), "0.1"], ids=repr)
    def test_probability_map_rejects_non_real_p(self, bad):
        """A bool is an ``int`` subclass and would silently be treated as 0 or 1."""
        _expect_contract_violation(hazard_p_to_per_hour, bad, why="non-real p_horizon")

    @pytest.mark.parametrize("bad", [None, True, False, np.bool_(True), "0.1"], ids=repr)
    def test_delta_method_rejects_non_real_sigma(self, bad):
        """sigma must be a real number; bools and None must not reach the arithmetic."""
        _expect_contract_violation(hazard_uncertainty_to_per_hour, bad, 0.1, why="non-real sigma_horizon")

    @pytest.mark.parametrize("bad", [None, True, False, np.bool_(True), "0.1"], ids=repr)
    def test_delta_method_rejects_non_real_p(self, bad):
        """p must be a real number; bools and None must not reach the arithmetic."""
        _expect_contract_violation(hazard_uncertainty_to_per_hour, 0.1, bad, why="non-real p_horizon")

    @pytest.mark.parametrize("fn_args", [(0.1,), (0.1, 0.1)], ids=["p_only", "sigma_and_p"])
    def test_numpy_floats_are_real_numbers(self, fn_args):
        """numpy floating scalars are accepted wherever a float is."""
        np_args = tuple(np.float64(a) for a in fn_args)
        if len(fn_args) == 1:
            assert hazard_p_to_per_hour(*np_args) == pytest.approx(hazard_p_to_per_hour(*fn_args), rel=1e-12)
        else:
            assert hazard_uncertainty_to_per_hour(*np_args) == pytest.approx(
                hazard_uncertainty_to_per_hour(*fn_args), rel=1e-12
            )


class TestAggregateGroupHazard:
    def test_hand_computed_sum_and_quadrature(self):
        """With H = 1 the conversions are the identity, so the hazard is the plain sum
        of the p values and the uncertainty is sqrt(0.01^2 + 0.02^2) = 0.0223607."""
        hazard, unc = aggregate_group_hazard([(0.1, 0.01), (0.2, 0.02)], horizon_hours=1.0)
        assert hazard == pytest.approx(0.3, rel=1e-12)
        assert unc == pytest.approx(0.0223606797749979, rel=1e-12)

    def test_three_four_five_quadrature(self):
        """sqrt(3^2 + 4^2) = 5: with H = 1 and sigmas 0.03 and 0.04 the result is 0.05."""
        _, unc = aggregate_group_hazard([(0.1, 0.03), (0.1, 0.04)], horizon_hours=1.0)
        assert unc == pytest.approx(0.05, rel=1e-12)

    def test_matches_per_drive_conversions_at_default_horizon(self):
        """Hazard is the sum of ``hazard_p_to_per_hour`` over drives; uncertainty is the
        root-sum-square of the delta-method values evaluated at each drive's own p."""
        pairs = [(0.01, 0.005), (0.05, 0.02), (0.3, 0.1)]
        hazard, unc = aggregate_group_hazard(pairs)
        assert hazard == pytest.approx(sum(hazard_p_to_per_hour(p) for p, _ in pairs), rel=1e-12)
        expected = math.sqrt(sum(_delta(u, p) ** 2 for p, u in pairs))
        assert unc == pytest.approx(expected, rel=1e-12)

    def test_returns_pair_of_floats(self):
        """Callers unpack the result as ``(hazard, hazard_uncertainty)``."""
        out = aggregate_group_hazard([(0.02, 0.01)])
        assert len(out) == 2
        assert all(isinstance(x, float) for x in out)

    def test_nine_identical_drives_give_nine_times_one_drive(self):
        """The rate is additive over drives (9x one drive) and independent errors add in
        quadrature (sqrt(9) = 3x one drive's uncertainty)."""
        p, u = 0.04, 0.02
        one_h, one_u = aggregate_group_hazard([(p, u)])
        nine_h, nine_u = aggregate_group_hazard([(p, u)] * (GROUP_SIZE - 1))
        assert nine_h == pytest.approx(9 * one_h, rel=1e-12)
        assert nine_u == pytest.approx(3 * one_u, rel=1e-12)

    def test_order_of_drives_does_not_matter(self):
        """Both aggregates are symmetric in the drives."""
        pairs = [(0.01, 0.005), (0.05, 0.02), (0.3, 0.1), (0.002, 0.001)]
        forward = aggregate_group_hazard(pairs)
        backward = aggregate_group_hazard(pairs[::-1])
        assert forward == pytest.approx(backward, rel=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 5, 8, 9])
    def test_up_to_nine_entries_accepted(self, n):
        """Survivors can drop below nine as further drives fail."""
        aggregate_group_hazard([(0.02, 0.01)] * n)

    def test_zero_probability_drives_contribute_nothing(self):
        """A drive with p = 0 and sigma = 0 adds no rate and no uncertainty."""
        assert aggregate_group_hazard([(0.0, 0.0)]) == (0.0, 0.0)

    def test_empty_sequence_rejected(self):
        """With no surviving drives there is nothing to aggregate."""
        with pytest.raises(ContractViolation):
            aggregate_group_hazard([])

    @pytest.mark.parametrize("n", [GROUP_SIZE, GROUP_SIZE + 5])
    def test_more_than_survivors_rejected(self, n):
        """At most ``GROUP_SIZE - 1`` drives can be alive during a rebuild episode."""
        with pytest.raises(ContractViolation):
            aggregate_group_hazard([(0.02, 0.01)] * n)

    @pytest.mark.parametrize("bad", [(0.1,), (0.1, 0.1, 0.1), (), 0.1, None, "ab", b"ab"])
    def test_malformed_pair_rejected(self, bad):
        """Each entry must be exactly a (p_horizon, uncertainty) pair."""
        with pytest.raises(ContractViolation):
            aggregate_group_hazard([(0.02, 0.01), bad])

    @pytest.mark.parametrize(
        "bad, reason",
        [
            (None, "NoneType"),
            ("ab", "str"),
            ((0.1, 0.1, 0.1), "exactly 2"),
            ((0.1,), "exactly 2"),
            ((0.1, None), "uncertainty"),
            ((0.1, "x"), "uncertainty"),
            ((None, 0.1), "p_horizon"),
            (("x", 0.1), "p_horizon"),
            ((True, 0.1), "p_horizon"),
            ((np.bool_(True), 0.1), "p_horizon"),
            ((0.1, True), "uncertainty"),
        ],
        ids=repr,
    )
    def test_malformed_element_raises_contract_violation_not_type_error(self, bad, reason):
        """A caller that catches ``ContractViolation`` to report a clean boundary error
        would not catch a leaked ``TypeError``, which defeats the purpose of a named
        contract exception.  Each malformed element must therefore raise
        ``ContractViolation`` whose message names the element index and the reason."""
        pairs = [(0.02, 0.01), bad, (0.02, 0.01)]
        with pytest.raises(ContractViolation) as err:
            aggregate_group_hazard(pairs)
        msg = str(err.value)
        assert "[1]" in msg, f"message does not name the pair index: {msg!r}"
        assert reason in msg, f"message does not give the reason ({reason!r}): {msg!r}"

    def test_numpy_float_members_are_accepted(self):
        """numpy float64 and float32 scalars are real numbers and arrive naturally from
        array-based risk models, so they must not be rejected as malformed."""
        py = aggregate_group_hazard([(0.1, 0.05), (0.2, 0.03)])
        np64 = aggregate_group_hazard([(np.float64(0.1), np.float64(0.05)), (np.float64(0.2), np.float64(0.03))])
        assert np64 == pytest.approx(py, rel=1e-12)
        f32 = [(np.float32(0.1), np.float32(0.05)), (np.float32(0.2), np.float32(0.03))]
        got = aggregate_group_hazard(f32)
        expected = aggregate_group_hazard([(float(a), float(b)) for a, b in f32])
        # Computation runs in float64 whatever the input dtype, so float32 operands give
        # exactly the value obtained from their float64 equivalents.
        assert got == expected
        assert all(isinstance(x, float) for x in got)

    @pytest.mark.parametrize("bad_p", [-0.1, 1.0, 1.5, float("nan")])
    def test_bad_probability_error_names_index_and_field(self, bad_p):
        """A failing p_horizon is reported with its pair index and as the p_horizon
        (not the uncertainty), so a risk-model bug can be traced to one drive."""
        pairs = [(0.02, 0.01), (0.02, 0.01), (bad_p, 0.01), (0.02, 0.01)]
        with pytest.raises(ContractViolation) as err:
            aggregate_group_hazard(pairs)
        msg = str(err.value)
        assert "2" in msg and "p_horizon" in msg
        assert "uncertainty" not in msg.replace("p_horizon", "")
        assert "0 <= p < 1" in msg or "< 1" in msg

    @pytest.mark.parametrize("bad_u", [-0.1, 0.6, float("nan"), float("inf")])
    def test_bad_uncertainty_error_names_index_field_and_bound(self, bad_u):
        """A failing uncertainty is reported with its pair index, as the uncertainty,
        and with the [0, 0.5] bound it broke."""
        pairs = [(0.02, 0.01), (0.02, 0.01), (0.02, bad_u), (0.02, 0.01)]
        with pytest.raises(ContractViolation) as err:
            aggregate_group_hazard(pairs)
        msg = str(err.value)
        assert "2" in msg and "uncertainty" in msg
        assert "0.5" in msg

    def test_error_index_tracks_the_failing_position(self):
        """The index in the message is the position of the offending pair."""
        for position in range(4):
            pairs = [(0.02, 0.01)] * 4
            pairs[position] = (0.02, 0.9)
            with pytest.raises(ContractViolation, match=rf"\b{position}\b"):
                aggregate_group_hazard(pairs)

    def test_aggregate_uncertainty_is_not_built_from_the_probability_map(self):
        """Regression lock: a standard deviation has been pushed through a probability
        map if the aggregate uncertainty equals the root-sum-square of
        ``hazard_p_to_per_hour(sigma_i)``.  The correct value uses the delta method
        at each drive's own p; at p = 0.5, sigma = 0.1 the two differ by about 2x."""
        pairs = [(0.5, 0.1), (0.4, 0.1)]
        _, unc = aggregate_group_hazard(pairs)
        correct = math.sqrt(sum(_delta(u, p) ** 2 for p, u in pairs))
        wrong = math.sqrt(sum(hazard_p_to_per_hour(u) ** 2 for _, u in pairs))
        assert unc == pytest.approx(correct, rel=1e-12)
        assert unc != pytest.approx(wrong, rel=0.2), (
            "aggregate uncertainty matches the probability-map conversion: a standard "
            "deviation has been pushed through a probability map instead of the delta method"
        )

    def test_aggregate_output_is_a_valid_observation_input(self):
        """The aggregate pair is accepted by ``Observation`` and, at a group hazard well
        above the scale, saturates to exactly 1.0 in the policy vector."""
        hazard, unc = aggregate_group_hazard([(0.3, 0.05)] * 9)
        obs = make_obs(hazard=hazard, hazard_uncertainty=unc)
        obs.validate()
        vec = obs.to_vector()
        assert vec[OBS_FIELDS.index("hazard")] == 1.0
        low_h, low_u = aggregate_group_hazard([(0.001, 0.0005)] * 9)
        low = make_obs(hazard=low_h, hazard_uncertainty=low_u).to_vector()
        assert 0.0 < low[OBS_FIELDS.index("hazard")] < 1.0
        assert 0.0 < low[OBS_FIELDS.index("hazard_uncertainty")] < 1.0


# ---------------------------------------------------------------------------
# Return-type identity
# ---------------------------------------------------------------------------
#
# Every contract function annotated ``-> float`` must return a builtin ``float``
# whatever numeric type it was given.  Tolerance-based value checks cannot see a
# numpy scalar leaking out, because ``np.float32(0.5) == 0.5``.
#
# Functions are discovered from the module by their return annotation and the
# parametrisation below must cover every one of them (see
# ``test_every_float_returning_function_is_covered``), so a function added to the
# contract later cannot escape the check.

_FLOAT_RETURN_ANNOTATIONS = ("float", "tuple[float, float]")


def _discover_float_returning_functions():
    found = set()
    for name, fn in inspect.getmembers(C, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != C.__name__:
            continue
        ann = inspect.signature(fn).return_annotation
        ann_text = ann if isinstance(ann, str) else getattr(ann, "__name__", None) or str(ann)
        ann_text = ann_text.replace("typing.", "").replace("Tuple", "tuple")
        if ann is float or ann_text in _FLOAT_RETURN_ANNOTATIONS:
            found.add(name)
    return found


# name -> {dtype label: (converter, args as plain Python numbers)}.  Real-valued
# dtypes use non-integral values that are exactly representable in float32 so the
# call is valid at every precision; integer dtypes use integral values.
def _numeric_cases(real_args, int_args):
    cases = {
        "float": (float, real_args),
        "np.float64": (np.float64, real_args),
        "np.float32": (np.float32, real_args),
    }
    if int_args is not None:
        cases["int"] = (int, int_args)
        cases["np.int64"] = (np.int64, int_args)
    return cases


_RETURN_TYPE_CASES = {
    "hazard_p_to_per_hour": _numeric_cases((0.25, 720.0), (0, 720)),
    "hazard_uncertainty_to_per_hour": _numeric_cases((0.125, 0.25, 720.0), (0, 0, 720)),
    "kbps_to_mbps": _numeric_cases((1500.5,), (1500,)),
    "latency_degradation": _numeric_cases((15.5, 10.0), (15, 10)),
    "latency_violation": _numeric_cases((20.5, 10.0), (20, 10)),
    "compute_reward": _numeric_cases((1.5, 2.5, 3.5), (1, 2, 3)),
    "degraded_cost": {"bool": (bool, (True,)), "np.bool_": (np.bool_, (True,))},
    "action_to_mbps": {"int": (int, (2,))},
}
# aggregate_group_hazard takes a list of pairs and returns a tuple of two floats.
_AGGREGATE_CASES = _numeric_cases((0.25, 0.125), (0, 0))

_FUNCTIONS = {name: getattr(C, name) for name in list(_RETURN_TYPE_CASES) + ["aggregate_group_hazard"]}

_RETURN_TYPE_PARAMS = [
    pytest.param(name, label, id=f"{name}-{label}")
    for name, cases in _RETURN_TYPE_CASES.items()
    for label in cases
]
_AGGREGATE_PARAMS = [pytest.param(label, id=f"aggregate_group_hazard-{label}") for label in _AGGREGATE_CASES]

_WHY_RETURN_TYPE_MATTERS = (
    "a lane passing numpy scalars receives a different return type from the same "
    "contract function than a lane passing Python floats, and the difference "
    "propagates silently into logs, JSON serialisation (np.float32 is not "
    "JSON-serialisable) and the dtype of anything built from the result"
)


class TestReturnTypeIdentity:
    def test_every_float_returning_function_is_covered(self):
        """The check is driven by discovery: any public contract function annotated to
        return ``float`` (or a pair of floats) must appear in the parametrisation, so a
        function added later cannot silently escape the return-type check."""
        discovered = _discover_float_returning_functions()
        covered = set(_RETURN_TYPE_CASES) | {"aggregate_group_hazard"}
        assert discovered, "discovery found no float-returning functions; the annotation scan is broken"
        missing = discovered - covered
        assert not missing, f"float-returning contract functions with no return-type case: {sorted(missing)}"
        stale = covered - discovered
        assert not stale, f"return-type cases for functions that are not annotated to return float: {sorted(stale)}"

    @pytest.mark.parametrize("name, label", _RETURN_TYPE_PARAMS)
    def test_returns_builtin_float(self, name, label):
        """``type(result) is float``, not ``isinstance``: a numpy scalar can satisfy
        ``isinstance(x, float)`` (np.float64) or compare equal to one (np.float32), so
        only identity detects the leak."""
        conv, plain_args = _RETURN_TYPE_CASES[name][label]
        args = tuple(conv(a) for a in plain_args)
        result = _FUNCTIONS[name](*args)
        assert type(result) is float, (
            f"{name}({label} inputs) returned {type(result).__module__}.{type(result).__name__}, "
            f"not builtin float: {_WHY_RETURN_TYPE_MATTERS}"
        )

    @pytest.mark.parametrize("label", _AGGREGATE_PARAMS)
    def test_aggregate_group_hazard_returns_builtin_floats(self, label):
        """Both elements of the (hazard, hazard_uncertainty) pair are builtin floats."""
        conv, plain_args = _AGGREGATE_CASES[label]
        pairs = [tuple(conv(a) for a in plain_args)] * 3
        result = aggregate_group_hazard(pairs)
        assert type(result) is tuple and len(result) == 2
        for position, element in zip(("hazard", "hazard_uncertainty"), result):
            assert type(element) is float, (
                f"aggregate_group_hazard({label} inputs) returned {position} of type "
                f"{type(element).__module__}.{type(element).__name__}, not builtin float: "
                f"{_WHY_RETURN_TYPE_MATTERS}"
            )

    def test_result_values_do_not_depend_on_input_dtype(self):
        """Same numbers in different containers must give the same value, so a dtype
        change cannot also change the answer beyond float32 input rounding."""
        reference = hazard_p_to_per_hour(0.25, 720.0)
        assert hazard_p_to_per_hour(np.float64(0.25), np.float64(720.0)) == reference
        assert hazard_p_to_per_hour(np.float32(0.25), np.float32(720.0)) == reference
        assert hazard_p_to_per_hour(np.int64(0), np.int64(720)) == 0.0


class TestReplayDtypes:
    def test_keys_equal_replay_columns_in_order(self):
        """The dtype table is part of the replay schema: its keys are the six columns
        in schema order, not merely the same set, since order is what a positional
        reader of the file relies on."""
        assert tuple(REPLAY_DTYPES) == REPLAY_COLUMNS
        assert len(REPLAY_DTYPES) == len(REPLAY_COLUMNS) == 6

    @pytest.mark.parametrize("column", REPLAY_COLUMNS)
    def test_every_dtype_is_exactly_int64(self, column):
        """The source fields are integer counts.  A float column silently changes the
        format (and loses exactness above 2**53), so the schema pins int64 exactly,
        not any integer or numeric kind."""
        dtype = REPLAY_DTYPES[column]
        assert dtype == np.dtype("int64")
        assert dtype.kind == "i" and dtype.itemsize == 8


def _make_frame(pd, n=3, **columns):
    """Return a valid replay frame of ``n`` rows, with any column replaced by ``columns``."""
    data = {
        "t_seconds": np.arange(n, dtype="int64") * 300,
        "read_iops": np.arange(n, dtype="int64") + 10,
        "write_iops": np.arange(n, dtype="int64") + 20,
        "read_bw_kbps": np.arange(n, dtype="int64") + 30,
        "write_bw_kbps": np.arange(n, dtype="int64") + 40,
        "disk_usage_mb": np.arange(n, dtype="int64") + 50,
    }
    data.update(columns)
    return pd.DataFrame(data, columns=list(REPLAY_COLUMNS))


class TestReplayFrame:
    @pytest.fixture(autouse=True)
    def pd(self):
        return pytest.importorskip("pandas")

    def _violation(self, df):
        with pytest.raises(ContractViolation) as err:
            validate_replay_frame(df)
        return str(err.value)

    def test_valid_frame_passes_silently(self, pd):
        """A well-formed frame returns None; this guards the rejections below against
        passing only because everything is rejected."""
        assert validate_replay_frame(_make_frame(pd)) is None
        assert validate_replay_frame(_make_frame(pd, n=1)) is None
        assert validate_replay_frame(_make_frame(pd, n=50)) is None

    @pytest.mark.parametrize("bad", [{}, {"t_seconds": [0, 300]}, [[0, 1]], None, np.zeros((3, 6), dtype="int64")], ids=repr)
    def test_non_dataframe_is_rejected(self, bad):
        """Only a DataFrame carries the column names and dtypes the schema constrains."""
        msg = self._violation(bad)
        assert "DataFrame" in msg

    def test_wrong_column_order_is_rejected(self, pd):
        """Columns are positional in the schema; the same names in another order are a
        different file format."""
        df = _make_frame(pd)[list(REPLAY_COLUMNS[::-1])]
        msg = self._violation(df)
        assert "order" in msg

    def test_missing_column_is_rejected_and_named(self, pd):
        """The message names the column that is absent."""
        msg = self._violation(_make_frame(pd).drop(columns=["write_iops"]))
        assert "write_iops" in msg

    def test_extra_column_is_rejected_and_named(self, pd):
        """The message names the unexpected column."""
        df = _make_frame(pd)
        df["queue_depth"] = np.zeros(len(df), dtype="int64")
        msg = self._violation(df)
        assert "queue_depth" in msg

    def test_duplicate_column_name_is_rejected_and_named(self, pd):
        """A repeated name makes column lookup ambiguous."""
        df = _make_frame(pd)
        df = pd.concat([df, df[["read_iops"]]], axis=1)
        msg = self._violation(df)
        assert "read_iops" in msg and "duplicate" in msg

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint64", "bool"])
    def test_wrong_dtype_is_rejected_naming_column_expected_and_actual(self, pd, dtype):
        """Each wrong integer width, signedness, float or boolean dtype is a different
        format; the message names the column, the expected dtype and the actual one."""
        df = _make_frame(pd)
        df["read_bw_kbps"] = df["read_bw_kbps"].astype(dtype)
        msg = self._violation(df)
        assert "read_bw_kbps" in msg
        assert "int64" in msg, f"expected dtype missing from message: {msg!r}"
        assert dtype in msg, f"actual dtype missing from message: {msg!r}"

    def test_nullable_int64_without_nulls_is_a_dtype_mismatch(self, pd):
        """pandas' nullable ``Int64`` is not NumPy ``int64`` even with no missing value."""
        df = _make_frame(pd)
        df["read_iops"] = df["read_iops"].astype("Int64")
        msg = self._violation(df)
        assert "read_iops" in msg and "Int64" in msg and "int64" in msg

    def test_nan_in_float_column_is_reported_as_null(self, pd):
        """A NaN in a float column is a null, named with its column."""
        df = _make_frame(pd)
        df["write_iops"] = df["write_iops"].astype("float64")
        df.loc[1, "write_iops"] = np.nan
        msg = self._violation(df)
        assert "write_iops" in msg and "null" in msg.lower()

    def test_pd_na_in_nullable_int_column_is_reported_as_null(self, pd):
        """A ``pd.NA`` in an ``Int64`` column is a null, named with its column."""
        df = _make_frame(pd)
        df["disk_usage_mb"] = df["disk_usage_mb"].astype("Int64")
        df.loc[2, "disk_usage_mb"] = pd.NA
        msg = self._violation(df)
        assert "disk_usage_mb" in msg and "null" in msg.lower()

    def test_null_is_reported_as_null_not_as_dtype_mismatch(self, pd):
        """Check order: nulls are tested before dtypes.  A null coerces an integer
        column to float or object, so a dtype-first implementation would report the
        symptom (wrong dtype) instead of the cause (missing value), and the message is
        what someone debugging a broken file reads.  In both cases the column here has
        a non-int64 dtype *because of* the null, and the message must say null only."""
        for dtype, na in (("float64", np.nan), ("Int64", pd.NA), ("object", None)):
            df = _make_frame(pd)
            df["read_iops"] = df["read_iops"].astype(dtype)
            df.loc[1, "read_iops"] = na
            msg = self._violation(df)
            assert "null" in msg.lower(), f"{dtype}: null not reported: {msg!r}"
            assert "dtype" not in msg.lower(), f"{dtype}: null reported as a dtype mismatch: {msg!r}"

    def test_negative_value_is_rejected_with_column_and_row(self, pd):
        """Counts and sizes cannot be negative; the message gives the column and the row."""
        df = _make_frame(pd)
        df.loc[2, "write_bw_kbps"] = -7
        msg = self._violation(df)
        assert "write_bw_kbps" in msg
        assert re.search(r"\b2\b", msg), f"row index missing from message: {msg!r}"
        assert "negative" in msg.lower()

    def test_negative_value_row_index_tracks_position(self, pd):
        """The reported row follows the offending row rather than being a constant."""
        for row in (0, 1, 4):
            df = _make_frame(pd, n=5)
            df.loc[row, "read_iops"] = -1
            assert re.search(rf"\b{row}\b", self._violation(df))

    @pytest.mark.parametrize(
        "t_seconds",
        [
            np.array([300, 600, 900], dtype="int64"),  # does not start at 0
            np.array([0, 60, 120], dtype="int64"),  # wrong step
            np.array([0, 300, 300], dtype="int64"),  # duplicate timestamp
            np.array([0, 600, 300], dtype="int64"),  # not increasing
            np.array([0, 300, 900], dtype="int64"),  # gap
        ],
        ids=["bad_origin", "wrong_step", "duplicate", "not_increasing", "gap"],
    )
    def test_bad_time_grid_is_rejected(self, pd, t_seconds):
        """The time column must be the 300 s lattice from 0, as ``validate_replay_grid`` requires."""
        self._violation(_make_frame(pd, t_seconds=t_seconds))

    def test_empty_frame_is_rejected(self, pd):
        """A frame with the right columns and dtypes but no rows has no replay grid."""
        self._violation(_make_frame(pd, n=0))

    # -- ordering of the checks -------------------------------------------------

    def test_check_order_type_before_columns(self, pd):
        """The argument type is checked first; a dict is not inspected for columns."""
        assert "DataFrame" in self._violation({"anything": 1})

    def test_check_order_columns_before_nulls(self, pd):
        """A frame with a wrong column set and a null reports the columns."""
        df = _make_frame(pd).drop(columns=["write_iops"])
        df["read_iops"] = df["read_iops"].astype("float64")
        df.loc[0, "read_iops"] = np.nan
        msg = self._violation(df)
        assert "write_iops" in msg and "null" not in msg.lower()

    def test_check_order_nulls_before_dtypes_before_negatives(self, pd):
        """Nulls, then dtypes, then negatives: each earlier defect masks the later one."""
        df = _make_frame(pd)
        df["read_iops"] = df["read_iops"].astype("float64")
        df.loc[0, "read_iops"] = np.nan
        df["write_iops"] = df["write_iops"].astype("float64")  # wrong dtype as well
        df.loc[1, "write_bw_kbps"] = -1  # negative as well
        assert "null" in self._violation(df).lower()

        df = _make_frame(pd)
        df["write_iops"] = df["write_iops"].astype("float64")  # dtype defect, no null
        df.loc[1, "write_bw_kbps"] = -1  # negative as well
        msg = self._violation(df)
        assert "write_iops" in msg and "dtype" in msg.lower() and "negative" not in msg.lower()

    def test_check_order_negatives_before_grid(self, pd):
        """A negative value is reported before a time-grid defect."""
        df = _make_frame(pd, t_seconds=np.array([300, 600, 900], dtype="int64"))
        df.loc[1, "read_iops"] = -3
        msg = self._violation(df)
        assert "negative" in msg.lower() and "read_iops" in msg


class TestImportSurface:
    _PROBE = (
        "import sys; sys.path.insert(0, {root!r}); "
        "import contracts.contracts; "
        "sys.exit(1 if 'pandas' in sys.modules else 0)"
    )

    def _run(self, code):
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )

    def test_importing_the_contract_does_not_load_pandas(self):
        """pandas is imported lazily inside ``validate_replay_frame`` so that lanes
        which do not use pandas keep a numpy-and-stdlib-only import surface.  This runs
        in a fresh interpreter because this test session has already loaded pandas."""
        result = self._run(self._PROBE.format(root=str(Path(__file__).resolve().parent.parent)))
        assert result.returncode == 0, (
            "importing contracts.contracts loaded pandas (or failed): "
            f"{result.stdout!r} {result.stderr!r}"
        )

    def test_probe_detects_pandas_when_it_is_loaded(self):
        """Control: the same probe fails if pandas is imported first, so the check above
        is capable of failing rather than vacuously passing."""
        pytest.importorskip("pandas")
        code = "import pandas; " + self._PROBE.format(root=str(Path(__file__).resolve().parent.parent))
        assert self._run(code).returncode == 1


class TestUnitConversion:
    def test_known_values(self):
        """Replay bandwidth is in kB/s and observations are in MB/s."""
        assert kbps_to_mbps(1000) == 1.0
        assert kbps_to_mbps(0) == 0.0
        assert kbps_to_mbps(2500) == pytest.approx(2.5)
        assert isinstance(kbps_to_mbps(1000), float)

    def test_decimal_convention_is_deliberate(self):
        """The factor is 1000 (SI), not 1024: 1024 kB/s is 1.024 MB/s.  A binary
        factor would understate bandwidth by 2.4% and shift every normalised input."""
        assert kbps_to_mbps(1024) == pytest.approx(1.024)
        assert kbps_to_mbps(1024) != 1.0
        assert kbps_to_mbps(1000) != pytest.approx(1000 / 1024)


class TestReplayGrid:
    @pytest.mark.parametrize(
        "grid",
        [[0, 300, 600], (0, 300, 600), range(0, 3000, 300), [0, 300]],
    )
    def test_valid_grids_pass(self, grid):
        """A 300 s lattice starting at zero is the replay time base."""
        assert validate_replay_grid(grid) is None

    @pytest.mark.parametrize(
        "grid",
        [
            [300, 600],  # does not start at 0
            [0, 300, 300],  # duplicate
            [0, 300, 900],  # gap
            [0, 60, 120],  # wrong step
            [0, 600, 300],  # not increasing
            [-300, 0, 300],  # negative origin
            [0, 300, 600, 600],  # trailing duplicate
            [0, 300, 600, 500],  # step shrinks at the end
        ],
    )
    def test_invalid_grids_raise(self, grid):
        """Misaligned replay traces would pair load samples with the wrong control step."""
        with pytest.raises(ContractViolation):
            validate_replay_grid(grid)

    def test_empty_grid_raises(self):
        """An empty grid cannot start at 0, so it fails the origin requirement."""
        with pytest.raises(ContractViolation):
            validate_replay_grid([])

    def test_single_zero_is_valid(self):
        """``[0]`` starts at 0 and has no step to violate, so it is accepted."""
        assert validate_replay_grid([0]) is None


class _ConformingModel:
    """Smallest object satisfying the ``HazardModel`` call signature."""

    def __call__(self, features, horizon_hours=HORIZON_HOURS):
        return 0.05, 0.01


class TestHazardModelProtocol:
    def test_is_a_protocol(self):
        """The hazard model is an interface, not a concrete base class."""
        assert typing.Protocol in HazardModel.__mro__
        assert getattr(HazardModel, "_is_protocol", False)

    def test_signature(self):
        """Callers pass ``features`` positionally and may omit the horizon."""
        params = inspect.signature(HazardModel.__call__).parameters
        assert list(params)[:3] == ["self", "features", "horizon_hours"]
        assert params["horizon_hours"].default == HORIZON_HOURS

    def test_conforming_class_satisfies_protocol(self):
        """Structural conformance: a runtime-checkable protocol must accept any
        callable object, otherwise the check is asserted through behaviour."""
        model = _ConformingModel()
        if getattr(HazardModel, "_is_runtime_protocol", False):
            assert isinstance(model, HazardModel)
            assert not isinstance(object(), HazardModel)
        else:
            assert callable(model)

    def test_conforming_model_output_feeds_the_pipeline(self):
        """The returned (p_horizon, uncertainty) pair is converted to per-hour
        hazards and must yield a valid ``Observation`` with no further massaging."""
        model = _ConformingModel()
        p, u = model({"smart_5": 3.0})
        q = hazard_p_to_per_hour(p)
        q_unc = hazard_uncertainty_to_per_hour(u, p)
        obs = make_obs(hazard=q, hazard_uncertainty=q_unc)
        obs.validate()
        assert obs.to_vector()[OBS_FIELDS.index("hazard")] <= 1.0

    def test_horizon_argument_is_accepted_by_keyword(self):
        """The horizon is a keyword-capable parameter with the contract default."""
        model = _ConformingModel()
        assert model({}, horizon_hours=24.0) == model({}, 24.0)


def test_module_exports_are_importable_by_name():
    """The names listed in the specification are all attributes of the module."""
    for name in (
        "OBS_FIELDS", "OBS_SCALES", "Observation", "ContractViolation", "action_to_mbps",
        "INFO_KEYS", "validate_info", "TerminationReason", "RewardConfig",
        "latency_degradation", "latency_violation", "degraded_cost", "compute_reward", "hazard_p_to_per_hour", "hazard_uncertainty_to_per_hour", "aggregate_group_hazard", "GROUP_SIZE", "kbps_to_mbps",
        "HazardModel", "REPLAY_COLUMNS", "DISK_INDEX_COLUMNS", "validate_replay_grid",
    ):
        assert hasattr(C, name), name
