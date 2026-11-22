# Decision record: controllable and filter-forced scenarios in the evaluation harness

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes an existing default or the
environment's dynamics.

Code: `baselines/evaluate.py` (columns, `scenario_blocks`, `forced_masks`, `forced_seed_sets`,
`controllable_sensitivity`, `pareto_table`, `recovery_time_distribution`); tests in
`baselines/test_scenario_split.py`. Evidence: `agent/evidence/scenario-split-2026-11-19/`.

## Decision

- **Scenario flag.** `filter_forced` marks a scenario in which the safety filter raises a `fixed_20` rollout's
  applied action on at least 10% of the intervals (`FILTER_FORCED_CLIP_FRACTION`). Controllable scenarios are the
  rest. The flag is a property of the scenario, identical for every policy, and was fixed before any result was
  seen.
- **Sensitivity variants, reported beside every controllable figure.** `fixed_20` clip fraction at least 1% and
  at least 25%, and `fixed_20` latency cost above 50 (`forced_masks`, `controllable_sensitivity`).
- **What the primary flag measures.** It measures filter activity, not uncontrollability. Seed 3 separates the
  two: the filter raises its allocation on 42% of the intervals, and the latency consequence is small (`fixed_20`
  latency cost 21.8 on the holdout, 0.8 on altB).
- **New per-episode columns.** `latency_cost` (the sum of `alpha * D + gamma * V`), `exogenous_violations`,
  `data_loss_probability`, `fixed20_clip_fraction`, `fixed20_latency_cost`, `filter_forced`.
- **`exogenous_violations`.** The intervals in which the smallest action the safety filter allows would also have
  violated latency, on the policy's own trajectory. The floor is `apply_safety_filter(0, hazard, work)` with the
  hazard from `info["hazard_reported_per_hour"]` and the work from the observation; the delivered rate and the
  latency use the environment's formulas.
- **`data_loss_probability`.** The probability of two or more further failures, for a Poisson count whose mean is
  the true group hazard integrated over the realised episode. Labelled "simulated data-loss probability".
- **Summaries.** Three blocks (all, controllable, filter-forced), a Pareto table, and a recovery-time distribution
  with its tail (mean, p50, p90, p95, p99, max).

## Evidence

Holdout / altB, 20 scenarios, `v1,eval-fleet,sigma=0.25`, `ORACLE_lookahead` minus `risk_aware_rule_based`:

| Definition of controllable | Holdout n | Holdout gap | altB n | altB gap |
| --- | --- | --- | --- | --- |
| all scenarios | 20 | +11.850 +/- 1.496 (t 7.92) | 20 | +29.246 +/- 2.924 (t 10.00) |
| primary (10%) | 16 | +13.753 +/- 1.472 (t 9.35) | 16 | +33.165 +/- 2.716 (t 12.21) |
| clip fraction at least 1% | 13 | +13.248 +/- 1.618 (t 8.19) | 13 | +31.585 +/- 3.195 (t 9.89) |
| clip fraction at least 25% | 16 | +13.753 +/- 1.472 (t 9.35) | 16 | +33.165 +/- 2.716 (t 12.21) |
| `fixed_20` cost above 50 | 17 | +13.507 +/- 1.404 (t 9.62) | 17 | +32.296 +/- 2.695 (t 11.98) |

- **Flagged scenarios.** Primary: {3, 9, 17, 19} on both profiles. 1%: {2, 3, 9, 16, 17, 18, 19}. 25%: {3, 9, 17,
  19}. `fixed_20` cost above 50: {9, 17, 19} on both.
- **Concentration.** On the holdout, seeds 9, 17 and 19 carry 84.2% of the rule's total reward cost and 97.7% of
  its latency cost; on altB 61.8% and 80.4%.
- **Exogenous share.** The rule has 18.2 violations per episode on the holdout, 17.9 of them also under the
  smallest allowed action (98%); on altB 9.3 and 7.2 (77%).
- **Identical outcomes.** `risk_aware_rule_based` and `workload_only_throttle` have identical rewards on every
  scenario of both profiles.

## Decisions made during execution

- **Primary definition kept as registered.** It flags seed 3 in addition to the expected seeds. Rejected:
  adjusting the threshold to drop seed 3, which would change a definition fixed before the result.
- **Variant expectation did not match on altB.** The "`fixed_20` cost above 50" variant flags {9, 17, 19} on altB,
  not {9, 17} (seed 19 costs 145.5). Any split defined by the rule's total reward below -500 gives {9, 17}; that
  is a different definition and is not used here.
- **Hazard source corrected.** A first version read the hazard from the observation, which clips at 1.8e-4 per
  hour; on seeds 9 and 17 it counted 68 and 20 (holdout) against 179 and 91 realised for `fixed_20`. The count
  now uses the reported hazard of `info`; it matches the realised violations on all 20 scenarios of both
  profiles. The correction follows the registered intent ("after the safety filter"). Alternative rejected:
  keeping the clipped value and documenting a lower bound.
- **Scenario flags are computed once per workload, seed set and configuration** by a `fixed_20` rollout and
  cached; each `evaluate` call therefore adds at most one set of `fixed_20` rollouts.

## Limits

- **Counterfactual trajectory.** `exogenous_violations` takes the hazard and the remaining work from the
  policy's own trajectory, not from a rollout of the smallest action.
- **Latency filter.** With `latency_filter` options the counterfactual is the same, because the smallest allowed
  action is unchanged by the ceiling.
- **Data-loss probability.** The Poisson mean uses the group hazard of the realised episode and ignores the
  decline in survivors during the episode.
- **Selection of scenarios.** The flag uses a `fixed_20` rollout of the same scenario; a controller that changes
  the failure stream is not modelled (the failure stream is fixed by the seed).
- **Two profiles, 20 scenarios each.** The controllable subsets have 13 to 17 scenarios.
