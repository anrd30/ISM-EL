# Baseline sweep: `profile_diurnal_long`, held-out fleet, sigma 0 to 1

Four runs of `python -m baselines.evaluate`, `--workload-csv
data_pipeline/profiles/profile_diurnal_long.csv --eval-fleet --seeds 20
--diagnostics`, one per `--hazard-error-sigma` in `{0, 0.25, 0.5, 1.0}`. This
is the first baseline comparison on the 14-day aggregate window with the
held-out-split fleet (`v1_eval_config`); every number here is reportable --
no train/eval leakage, no stale checkout, unlike `agent/evidence/upperbound-2026-11-06/`.

## Finding: every metric is identical across all four sigma values

`episodes.csv` in each of the four directories is identical to the others at
full floating-point precision, for every policy and every seed -- not just
`risk_aware_rule_based` vs `workload_only_throttle` (the degeneracy already
in `sim/LIMITATIONS.md` and `agent/T_DECISION.md`), but every policy against
itself across sigma, including `clip_fraction` and `p_failure_mean` for
`risk_aware_rule_based` and `risk_aware_rule_based k=1`.

This is not noise or a broken flag -- `--hazard-error-sigma` reaches
`SimConfig.hazard_error_sigma` (tested in `baselines/test_baselines.py`) and
the config label on each report (`eval-fleet,sigma=<value>`) confirms each
run used a different value. The mechanism is structural:

- `sim/scenario.py`'s `sample_fleet()` computes `true_hazard_per_hour =
  estimated_hazard_per_hour * multiplier`, where `multiplier` is the only
  place `hazard_error_sigma` enters. `true_hazard_per_hour` feeds exactly one
  thing: the per-interval stochastic failure draw (`fail_prob`).
- `sim/recovery_env.py`'s `_observation()` sets `hazard=self._hazard`, built
  by `_group_hazard()` from `self._fleet.pairs` -- the hazard model's raw
  `(p_horizon, uncertainty)` output, i.e. `estimated_hazard_per_hour`.
  `true_hazard_per_hour` is read nowhere in observation-building or in the
  safety filter's exposure/clip computation (`sim/safety_filter.py`); its
  only other reader is the diagnostic `_hazard_realised_ratio()`.

So neither the policies (all of which decide from `o.hazard`/
`o.hazard_uncertainty`) nor the filter can see the noise `hazard_error_sigma`
introduces. The one channel left -- an actual failure firing -- is a rare
event at these per-interval hazards (~1e-3 to 1e-4); with 20 seeds per run,
none fired in any of the four sweeps (`terminal_failure` is 0 everywhere),
so even that channel shows nothing at this seed count.

## Why this matters

This independently confirms, across the full sigma range rather than one
value, and on a clean (non-contaminated) setup rather than the upper-bound
run's old window and default train-split fleet, that the current filter +
baseline-policy design makes the hazard-input ablation inert by
construction, not by coincidence of one test. A sigma sweep cannot show
anything through this pipeline until either the filter/policy read the
noisy (true) hazard, or the seed count is raised enough for the rare-event
channel to register -- the filter-ablation work already proposed in
`agent/T_DECISION.md` is the direct way to make this knob observable.
