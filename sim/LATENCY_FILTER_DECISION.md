# Simulator decision record: an opt-in latency ceiling

Status: proposed. Not yet reviewed by the other lanes.

Evidence: `agent/evidence/latency-filter-2026-11-11/`. Code: `sim/latency_filter.py`,
`SimConfig.latency_filter` and `SimConfig.latency_filter_causal`, `sim/recovery_env.py` `step()`, tests in
`sim/test_latency_filter.py`.

## Prediction, and what happened

Stated before the first evaluation:

1. The peak-load catastrophe disappears for every policy.
2. The home-distribution ranking barely moves.

The first evaluation used a ceiling that reads the foreground load of the interval being decided
(`ceiling_oracle_fg`), which a controller deciding at the start of the interval does not have. A second,
strictly causal ceiling (`ceiling_causal_fg`) reads the previous interval's load, the value observation field
`fg_bw` carries. Both were evaluated, with the unfiltered numbers, in one report. This section was rewritten
after the second evaluation, not appended to.

| | `ceiling_oracle_fg` | `ceiling_causal_fg` |
| --- | --- | --- |
| Prediction 1, peak catastrophe gone for the forecast-PPO agents | Held: gap to the rule -5633 → -20.5 | Mostly held: -5633 → -312; violation 0.76 → 0.16; the deficit is 15 times the oracle ceiling's |
| Prediction 2, home ranking barely moves | **Failed**: PPO's lead over the rule +6.3 → +3.3; lead over the hand-built rule +2.6 → -0.3 | **Held**: lead +6.24 / +6.32 / +6.71 → +6.31 / +6.33 / +6.83; lead over the hand-built rule +2.6 → +2.6 to +3.2 |

**Reading.** The home-ranking shrinkage seen in the first evaluation was an artifact of the oracle ceiling's
foresight. It lifted the rule and the hand-built rule by 3.4 reward and the PPO agents by about 0.5, because it
supplies look-ahead the baselines lacked. The causal ceiling does not move the rule at all on any condition
(identical to the digit), moves the hand-built rule by at most 0.2 reward, and leaves the PPO ranking on `home`
where it was. The first evaluation's conclusion that PPO's lead over the hand-built rule disappears does
not survive. What the causal ceiling does change is the PPO agents' unsafe requests on out-of-distribution
workloads, partly.

## Decision

- **A new, separate filter.** `sim/latency_filter.py` clips the applied allocation down to the largest
  entry of `ACTIONS_MBPS` with `(fg + alloc) / BW_TOTAL_MBPS <= violation_rho_threshold(cfg)`
  (about 0.911). When no entry satisfies it the ceiling is the smallest action. The threshold is the
  existing `sim.config.violation_rho_threshold`, not re-derived.
- **Provisional placement.** It is a separate module, not part of `sim/safety_filter.py`, because that
  file's scope is the reliability budget. This is a provisional choice and is open to revision.
- **Two opt-in flags, mutually exclusive, both default False.** `latency_filter` passes the interval's own
  foreground load (`ceiling_oracle_fg`). `latency_filter_causal` passes the load of row `position - 1`, which is
  the `fg_bw` the policy's observation carried when it chose (`ceiling_causal_fg`). With both off every existing
  test, evaluation and evidence file reproduces unchanged; the new `info` keys appear only with a flag on, and a
  test checks bit-identical trajectories. The oracle variant's results were re-run after the causal change and
  matched the earlier ones exactly.
- **Order in `step()`:** safety filter, then latency filter, then the headroom cap that already applies.
- **Precedence: reliability wins.** The applied action is
  `max(reliability_floor, min(action_after_reliability_filter, latency_ceiling))`, where the floor is the
  smallest action the safety filter accepts at the current hazard and remaining work. The ceiling never
  lowers an action below that floor. This is slightly broader than "the ceiling cannot undo a reliability
  raise": it also stops the ceiling pushing an action that the safety filter left alone below the
  feasibility floor.
- **Visible in `info` (a flag on only):** `latency_filter_active`, `latency_filter_fg_source` (`oracle` or
  `causal`), `latency_filter_fg_mbps` (the load used), `alloc_after_reliability_filter_mbps`,
  `latency_ceiling_mbps`, `latency_ceiling_feasible`, `alloc_clipped_by_latency_filter`,
  `latency_filter_overridden_by_reliability` (the ceiling wanted a lower action and the floor kept it higher),
  `reliability_filter_clipped`, `both_filters_fired`.
- **Contract keys with a flag on.** `alloc_applied_mbps`, `p_failure_before_complete` and
  `violated_risk_budget` describe the final applied action. `alloc_clipped_by_filter` is true when the final
  action differs from the request, from either filter.
- **Flag plumbing:** `--latency-filter` and `--latency-filter-causal` in `agent/train.py`,
  `agent/evaluate_policy.py` and `baselines/evaluate.py` (mutually exclusive; the config label gains
  `,latency-filter` or `,latency-filter-causal`). The same flags for `agent/evaluate_learned.py` and
  `agent/train_residual.py`, which are on the T9 branch only, are in
  `agent/evidence/latency-filter-2026-11-11/t9_cli_flag.patch`.

## Evidence

Harness first: the `home` condition without a filter reproduces the committed forecast-ppo evidence exactly
(largest per-scenario difference 0 for all five policies). Policies: the rule, the hand-built seasonal rule and
the three `forecast_ppo_bc` checkpoints, trained without any filter. 20 paired scenarios. The conditions are a
set chosen from the assets present, **not** those of `agent/evaluate_ood.py`, which is not available here; no
phase-shift condition was run (see the evidence README). Every table below is in `report.md` with the
standard errors and the scenario counts.

Paired gap against the rule under the same variant, for the PPO agents (seeds 0 / 1 / 2) and the hand-built rule:

| Condition | No filter | `ceiling_oracle_fg` | `ceiling_causal_fg` |
| --- | --- | --- | --- |
| home, PPO | +6.24 / +6.32 / +6.71 | +3.35 / +3.43 / +3.30 | +6.31 / +6.33 / +6.83 |
| home, hand-built | +3.68 | +3.69 | +3.68 |
| peak, PPO (identical across seeds) | -5633 | -20.5 | -312 |
| peak, hand-built | +3.69 | +3.69 | +3.69 |
| bursty, PPO | -59 / -61 / -71 | +6.4 / +6.5 / +6.5 | -46 / -50 / -49 |
| bursty, hand-built | +14.3 | +3.2 | +14.1 |
| diurnal_generic, PPO | +9.4 / +9.4 / +8.6 | -0.3 / -0.4 / -0.6 | +9.6 / +9.5 / +10.9 |
| diurnal_generic, hand-built | +9.7 | +2.3 | +9.6 |
| degraded fleet, PPO | +4.1 / +3.7 / +4.8 | +3.0 / +3.7 / +3.6 | +4.3 / +3.9 / +4.9 |
| idle; sigma 0.0 and 0.5 | no effect / identical to home | same | same |

- **Peak, PPO agents:** mean reward -7441 / -1829 / -2121; violation fraction 0.76 / 0.14 / 0.16; latency-budget
  share 0.00 / 0.70 / 0.65. The causal ceiling removes most of the catastrophe. The remaining -312 comes from
  the lag: a load rise inside an interval is not seen by a ceiling set from the interval before. The rule and
  the hand-built rule were never in a catastrophe (-1808 and -1805, violation 0.132) and the ceiling does not
  touch them.
- **Bursty, generic diurnal:** the oracle ceiling's reversals (PPO behind the rule on bursty becoming ahead;
  PPO ahead on generic diurnal becoming level) do not appear with the causal ceiling. Under it PPO is still
  46 to 50 reward behind the rule on bursty (from 59 to 71), and still 9.5 to 10.9 ahead on generic diurnal. The
  out-of-distribution loss on bursty is only partly addressed.
- **The home extremes are not touched, by design and under either ceiling.** The worst single-episode reward
  is -2782.7 in all three columns. On scenarios 9, 17 and 19 the violated fraction equals the
  reliability-overrode fraction (rule: 0.46, 0.23, 0.19 under the oracle ceiling; 0.459, 0.229, 0.187 violated
  against 0.456, 0.229, 0.187 overridden under the causal one; PPO seed 0 within 0.003). Every violating interval
  is one where the reliability floor sat above the ceiling. The precedence decision, not the ceiling's
  precision, leaves those episodes unchanged.

## Rejected alternatives

- **Merging into `sim/safety_filter.py`.** Rejected for now, on scope: that file's contract is the risk
  budget. Provisional.
- **Latency wins over reliability.** Not built. It would likely remove the home-extreme violations above and
  would also breach the risk budget the safety filter enforces, with no measurement of that trade here.
- **Lowering `rho_max` to the threshold.** It would change the physical model, the `bw_available`
  observation and every existing result, which an opt-in flag avoids.
- **A ceiling from a forecast of the interval's load** instead of the previous interval's. Not built; the
  causal ceiling's residual cost on peak is the room such a forecast would work in.

## Limits

- **`ceiling_oracle_fg` is not deployable.** It reads the interval's own load, which the observation a policy
  sees at decision time does not carry. Its numbers show what look-ahead of that load is worth in this
  simulator and should not be reported as a deployable result. `ceiling_causal_fg` uses only values already in
  the observation, so it could be implemented by a controller that applies the clip at the start of an interval;
  whether a real data path can clip within an interval against the live load, which the oracle ceiling would
  need, is not something this simulator says.
- **The policies cannot see the ceiling.** The observation is unchanged, so a policy cannot learn to use it.
  The checkpoints were not retrained under either filter.
- **Conditions are substitutes** for those in `agent/evaluate_ood.py` (absent here); no phase-shift run.
- **Holdout use.** The `home` condition evaluates the same checkpoints on the same holdout window more than
  once, under a changed simulator. No selection was made from any of it.
- **Statistical unit.** 20 scenarios shared by every policy; the three PPO seeds are not independent
  evidence on `peak`, where their numbers coincide.
- **Interpretation not tested.** That the causal ceiling never binds for the rule because the rule's own
  latency logic already respects the observed load is consistent with the zero clip rate in
  `extreme_diagnostic.txt` and the identical rewards, but was not isolated by an experiment.
- **Latency model.** Uncalibrated (`sim/LIMITATIONS.md`); every figure is a property of the model.
