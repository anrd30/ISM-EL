# Decision record: an exposure-weighted objective and the infeasible-budget variants (proposal)

Status: proposal, needs sign-off of all four teammates (it adds an optional reward mode and a filter option to the
simulator). Not yet reviewed by the other lanes. Nothing changes with the defaults (`risk_objective = None`,
`infeasible_budget_max_rate = True`). `contracts/` is not changed. Based on the fresh-populations branch (#38, which
contains #34, #35, #36 and #37) because the evaluation uses the scenario split, the optimum and the controller; the
specified base was the predictive-controller branch.

Code: `sim/config.py` (`RiskObjective`, `SimConfig.risk_objective`, `SimConfig.infeasible_budget_max_rate`),
`sim/recovery_env.py`, `baselines/risk_objective.py` (`calibrate_hazard_ref`, `HazardWeightedMixin`,
`HazardWeightedRule`), `baselines/dp_optimum.py` (weights and floors), `agent/evaluate_risk_objective.py`; tests in
`baselines/test_risk_objective.py` and `agent/test_evaluate_risk_objective.py`. Evidence:
`agent/evidence/risk-objective-2026-11-19/`.

## Decision

- **Objective.** The contract charges `beta * C` per degraded interval with `C = 1`. With `RiskObjective(h_ref)`
  the environment charges `beta * C * h / h_ref`, `h` the true group hazard at the start of the interval. `h_ref` is
  the mean initial true group hazard of the evaluation fleet over seeds 1000 to 1399: **1.0254e-4 per hour**
  (`calibrate_hazard_ref`), so the mean charge per degraded interval at the start of an episode is the contract's
  `beta = 0.1`. The weight multiplies the `c_degraded` argument of `contracts.compute_reward`.
- **Infeasible budget.** With `infeasible_budget_max_rate = False` the filter applies the requested action when no
  action meets the risk budget, instead of the largest; the exposure stays in `violated_risk_budget`. When the budget
  is feasible nothing changes.
- **Policies.** `HazardWeightedRule` and `HazardWeightedController` scale the rule's progress value (the reward it
  assigns to a MB/s of rebuild progress, standing in for the degraded-interval cost progress saves) by the observed
  hazard over `h_ref`; with the unforced option their risk floor is the smallest action when no action meets the
  budget at the inflated hazard. The optimum plans the same objective and the same filter option.

## Evidence

Holdout and altB, 20 scenarios, `v1,eval-fleet,sigma=0.25`, history rows 576, gains over `risk_aware_rule_based`
in the same environment (total reward under that environment's objective), standard errors across scenarios.

**1. Does the hazard change decisions?**

| Arm | Holdout, all (t) | Holdout, controllable | altB, all (t) | altB, controllable |
| --- | --- | --- | --- | --- |
| `workload_only_throttle` (risk-blind), objective | 0.000 (outcomes identical in 20 of 20 scenarios) | 0.000 | 0.000 (identical) | 0.000 |
| `hazard_weighted_rule`, objective | +0.524 +/- 0.072 (7.2) | +0.577 | +10.971 +/- 2.316 (4.7) | +13.570 |
| `predictive_controller`, objective | +8.114 +/- 1.170 (6.9) | +8.340 | +11.666 +/- 3.462 (3.4) | +14.727 |
| `hazard_weighted_controller`, objective | +8.465 +/- 1.205 (7.0) | +8.838 | +14.692 +/- 3.182 (4.6) | +18.974 |
| `DP_hindsight_optimum`, objective | +9.409 +/- 1.185 (7.9) | +9.845 | +26.998 +/- 2.868 (9.4) | +30.458 |

- **With the contract objective and with the exposure-weighted objective, the existing risk-aware rule and the
  risk-blind throttle give identical outcomes**: the rule's use of the hazard is the safety filter's, which the
  environment applies to both. The hazard changes a decision only through the new weighting: the hazard-weighted
  rule differs from the rule in all 20 scenarios and gains +0.52 (holdout) and +10.97 (altB).
- The weighting is worth little on the holdout (+0.35 for the controller over the unweighted controller) and +3.0 on
  altB.

**2. The infeasible-budget variants** (filter-forced seeds with an infeasible budget; `risk_budget_violated` is
true there): latency violation fraction and simulated data-loss probability (probability of two or more further
failures under the true group hazard over the realised episode), total reward under the objective.

| Seed | Environment, arm | Violation fraction | Data-loss probability | Total reward |
| --- | --- | --- | --- | --- |
| holdout 9 | forced, rule | 0.467 | 7e-5 | -2903.7 |
| holdout 9 | unforced, rule | 0.325 | 8e-5 | -2177.7 |
| holdout 9 | unforced, optimum | 0.000 | 1.1e-4 | -175.7 |
| holdout 17 | forced, rule | 0.236 | 9e-5 | -1549.7 |
| holdout 17 | unforced, rule | 0.236 | 9e-5 | -1549.7 |
| holdout 17 | unforced, optimum | 0.005 | 2.1e-4 | -278.2 |
| holdout 19 (budget feasible) | forced and unforced, rule and optimum | 0.187 and 0.203 | 2e-5 | -1409.6, -1402.6 |
| altB 9 | forced, rule | 0.230 | 8e-5 | -1472.3 |
| altB 9 | unforced, rule | 0.238 | 8e-5 | -1515.1 |
| altB 9 | unforced, optimum | 0.0035 | 1.5e-4 | -230.2 |
| altB 17 | forced, rule | 0.324 | 8e-5 | -1789.5 |
| altB 17 | unforced, rule | 0.324 | 8e-5 | -1789.5 |
| altB 17 | unforced, optimum | 0.000 | 1.3e-4 | -191.4 |

- **The trade.** Where the budget is infeasible, the exposure-weighted optimum removes almost all latency violations
  (0.23 to 0.49 of the intervals, 0.000 to 0.005 instead) and raises the simulated data-loss probability from about
  7e-5 to 9e-5 up to 1.1e-4 to 2.1e-4: an increase of 4e-5 to 1.3e-4 in absolute terms, in an environment where the
  data-loss probability is of order 1e-4 either way. Over all 20 scenarios the optimum's gain over the rule in the
  unforced environment is +172 +/- 115 (holdout) and +170 +/- 98 (altB), carried by these seeds.
- **No deployable arm gets it.** The unforced rule and controller keep violation fractions of 0.24 to 0.35 on these
  seeds; the hazard field of the observation saturates at 1.8e-4 per hour, so a policy with `k = 0` never sees an
  infeasible budget (the largest action is feasible at the saturated hazard), and the unforced floor is never
  triggered.
- **Over all scenarios,** mean violation fraction 0.0453 (rule) against 0.0472 (controller) and 0.0473 (optimum) on the
  holdout in the objective environment; data-loss probability 1e-5 for every arm.

## Decisions made during execution

- **The mode lives in `SimConfig`, not in `RewardConfig`.** `RewardConfig` is in `contracts/`, which may change only
  with a contract version bump and sign-off; the weight multiplies the `c_degraded` argument of the contract's
  `compute_reward`, so no contract name or value changes. Alternative rejected: editing `RewardConfig`.
- **True hazard in the reward, observed hazard in the policy.** The cost follows the realised risk; a policy sees
  only the (noisy, saturating) reported hazard.
- **Calibration seeds 1000 to 1399,** disjoint from the evaluation seeds 0 to 99.
- **Infeasible-budget variant as a filter option plus a policy floor,** because a policy cannot override a filter that
  raises its request; both are off by default.
- **Base branch.** The fresh-populations branch instead of the predictive-controller branch, because the evaluation
  needs the scenario split, the optimum and the controller.

## Limits

- **The data-loss probability is small in this simulator.** It is of order 1e-4 per episode under every arm; the
  exposure the risk budget bounds (0.5%) is the probability of one further failure, not of data loss. The objective's
  exposure term at the calibrated reference is weak compared with the latency terms (`gamma = 10` per violated
  interval), which is why relaxing the forced floor wins so much; a different weight would move the trade.
- **The weight, the reference and the data-loss definition are untuned assumptions,** like `beta`.
- **Saturating hazard observation.** The controller cannot see the infeasible budget (`OBS_SCALES["hazard"]` =
  1.8e-4); using the unclipped hazard would need an observation or contract change and was not tried.
- **The optimum is a hindsight reference** (`agent/OPTIMUM_DECISION.md`) and is not deployable.
- **Two profiles, 20 scenarios each,** and four filter-forced seeds.
